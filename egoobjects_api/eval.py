#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import logging
import math

import os
import time
from collections import defaultdict, OrderedDict
from multiprocessing import Pool
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pycocotools.mask as mask_utils
import torch
# from egodet.metric.metric import RecallAtPrecision

# from iopath.common.file_io import PathManager
# from iopath.fb.manifold import ManifoldPathHandler

from .egoobjects import EgoObjects, EgoObjectsMetaInfo

from .results import EgoObjectsResults


# pathmgr = PathManager()
# pathmgr.register_handler(ManifoldPathHandler())

logger = logging.getLogger(__name__)


def evaluate_img(
    det_type,
    img_id,
    class_id,
    area_ratio,
    background,
    lighting,
    difficulty,
    gt,
    dt,
    ious,
    params,
    img_nel,
):
    """Perform evaluation for single category and image."""
    if len(gt) == 0 and len(dt) == 0:
        return None

    # Add another filed _ignore to only consider anns satisfying the constraints.
    for g in gt:
        ignore = g["ignore"]
        if ignore == 0 and (area_ratio != "all" and area_ratio != g["area_ratio"]):
            ignore = 1
        if ignore == 0 and (background != "all" and background != g["background"]):
            ignore = 1
        if ignore == 0 and (lighting != "all" and lighting != g["lighting"]):
            ignore = 1
        if ignore == 0 and (
            difficulty != "all" and "difficulty" in g and difficulty != g["difficulty"]
        ):
            ignore = 1
        g["_ignore"] = ignore

    # Sort gt ignore last
    gt_idx = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
    gt = [gt[i] for i in gt_idx]
    # Sort dt highest score first
    dt_idx = np.argsort([-d["score"] for d in dt], kind="mergesort")
    dt = [dt[i] for i in dt_idx]
    # load computed ious
    ious = ious[:, gt_idx] if len(ious) > 0 else ious

    num_thrs = len(params.iou_thrs)
    num_gt = len(gt)
    num_dt = len(dt)
    # Array to store the "id" of the matched dt/gt
    gt_m = np.zeros((num_thrs, num_gt))
    dt_m = np.zeros((num_thrs, num_dt))

    gt_ig = np.array([g["_ignore"] for g in gt])
    dt_ig = np.zeros((num_thrs, num_dt))

    for iou_thr_idx, iou_thr in enumerate(params.iou_thrs):
        if len(ious) == 0:
            break

        for dt_idx, _dt in enumerate(dt):
            iou = min([iou_thr, 1 - 1e-10])
            # information about best match so far (m=-1 -> unmatched)
            # store the gt_idx which matched for _dt
            m = -1
            for gt_idx, _ in enumerate(gt):
                # if this gt already matched continue
                if gt_m[iou_thr_idx, gt_idx] > 0:
                    continue
                # if _dt matched to reg gt, and on ignore gt, stop
                if m > -1 and gt_ig[m] == 0 and gt_ig[gt_idx] == 1:
                    break
                # continue to next gt unless better match made
                if ious[dt_idx, gt_idx] < iou:
                    continue
                # if match successful and best so far, store appropriately
                iou = ious[dt_idx, gt_idx]
                m = gt_idx

            # No match found for _dt, go to next _dt
            if m == -1:
                continue

            # if gt to ignore for some reason update dt_ig.
            # Should not be used in evaluation.
            dt_ig[iou_thr_idx, dt_idx] = gt_ig[m]
            # _dt match found, update gt_m, and dt_m with "id"
            dt_m[iou_thr_idx, dt_idx] = gt[m]["id"]
            gt_m[iou_thr_idx, m] = _dt["id"]

    # We will ignore any unmatched detection if that category was
    # not exhaustively annotated in gt.
    class_id_key = "category_id" if det_type == "cat_det" else "instance_id"
    # dt_ig_mask = [
    #     d["area"] < area_rng[0]
    #     or d["area"] > area_rng[1]
    #     or d[class_id_key] in img_nel[d["image_id"]]
    #     for d in dt
    # ]
    dt_ig_mask = [
        d[class_id_key] in img_nel[d["image_id"]]
        or (area_ratio != "all" and d["area_ratio"] != area_ratio)
        or (background != "all" and d["background"] != background)
        or (lighting != "all" and d["lighting"] != lighting)
        or (difficulty != "all" and "difficulty" in d and d["difficulty"] != difficulty)
        for d in dt
    ]
    dt_ig_mask = np.array(dt_ig_mask).reshape((1, num_dt))  # 1 X num_dt
    dt_ig_mask = np.repeat(dt_ig_mask, num_thrs, 0)  # num_thrs X num_dt
    # Based on dt_ig_mask ignore any unmatched detection by updating dt_ig
    dt_ig = np.logical_or(dt_ig, np.logical_and(dt_m == 0, dt_ig_mask))

    # store results for given image and category
    return {
        "dt_ids": [d["id"] for d in dt],
        "dt_matches": dt_m,
        "dt_scores": [d["score"] for d in dt],
        "gt_ignore": gt_ig,
        "dt_ignore": dt_ig,
        "config": (img_id, class_id, area_ratio, background, lighting, difficulty),
    }


class EgoObjectsEval:
    def __init__(
        self,
        gt: EgoObjects,
        dt: EgoObjectsResults,
        num_processes: int = 1,
        max_dets: int = 100,
        eval_type: Tuple[str] = ("cat_det", "inst_det"),
    ):
        """
        Args:
            gt: ground truth
            dt: detection results
            num_processes: If 0, use single main process. If >0, use multiprocessing.Pool to
                do evaluate() using threads
            max_dets: maximal detections per image
        """
        assert num_processes >= 0

        self.gt = gt
        self.dt = dt
        self.num_processes = num_processes
        self.eval_type = eval_type

        self.eval_imgs = {}
        self.eval = {}
        self.gts = {}
        self.dts = {}
        self.results = {}
        self.ious = {}
        self.params = {
            "cat_det": CategoryDetectionParams(max_dets=max_dets),
            "inst_det": InstanceDetectionParams(max_dets=max_dets),
        }
        self.meta = EgoObjectsMetaInfo()
        self.freq_groups = {}
        self.img_nel = {}
        for det_type in self.eval_type:
            # per-image per-category evaluation results
            self.eval_imgs[det_type] = None
            self.eval[det_type] = {}  # accumulated evaluation results
            self.gts[det_type] = defaultdict(list)  # gt for evaluation
            self.dts[det_type] = defaultdict(list)  # dt for evaluation
            self.results[det_type] = OrderedDict()
            self.ious[det_type] = {}  # ious between all gts and dts

            self.params[det_type].img_ids = sorted(self.gt.get_img_ids())
            self.params[det_type].class_ids = sorted(self.gt.get_class_ids(det_type))

            logger.info(
                f"{det_type}, num class ids {len(self.params[det_type].class_ids)}"
            )

    def run(self, metric_filter):
        unique_metrics = {}
        for det_type in self.eval_type:
            unique_metrics = set(
                [
                    f"ar{x['ar']}-bg{x['bg']}-lt{x['lt']}-df{x['df']}"
                    for x in metric_filter[det_type]
                ]
            )
            self.evaluate(det_type, unique_metrics)
            self.accumulate(det_type, unique_metrics)
            self.summarize(det_type, metric_filter[det_type])

    def evaluate(
        self, det_type: str, unique_metrics: Set[str], multiprocessing: bool = False
    ):
        logger.info(f"Running per image evaluation for {det_type}.")

        start_time = time.time()

        class_ids = self.params[det_type].class_ids
        self._prepare(det_type)

        self.ious[det_type] = {
            (img_id, class_id): self.compute_iou(det_type, img_id, class_id)
            for img_id in self.params[det_type].img_ids
            for class_id in class_ids
        }

        logger.info(f"num_processes {self.num_processes}")

        # loop through images, area range, max detection number
        arg_tuples = []
        for class_id in class_ids:
            for area_ratio in self.params[det_type].area_rng_lbl:
                for bg in self.meta.background:
                    for lt in self.meta.lighting:
                        for df in self.params[det_type].difficulty:
                            for img_id in self.params[det_type].img_ids:
                                metric_tag = f"ar{area_ratio}-bg{bg}-lt{lt}-df{df}"
                                if metric_tag in unique_metrics:
                                    arg_tuples.append(
                                        (
                                            det_type,
                                            img_id,
                                            class_id,
                                            area_ratio,
                                            bg,
                                            lt,
                                            df,
                                            self.gts[det_type][img_id, class_id],
                                            self.dts[det_type][img_id, class_id],
                                            self.ious[det_type][img_id, class_id],
                                            self.params[det_type],
                                            self.img_nel[det_type],
                                        )
                                    )

        if self.num_processes > 1:
            with Pool(self.num_processes) as pool:
                self.eval_imgs[det_type] = pool.starmap(evaluate_img, arg_tuples)
        else:
            self.eval_imgs[det_type] = [evaluate_img(*x) for x in arg_tuples]

        elapsed_time = time.time() - start_time
        logger.info(f"Elapsed time of {det_type} evaluate(): {elapsed_time:.2f} sec")

    def accumulate(self, det_type: str, unique_metrics: Set[str]):
        """Accumulate per image evaluation results and store the result in
        self.eval[det_type].
        """
        logger.info(f"Accumulating evaluation results for {det_type}.")
        if self.eval_imgs[det_type] is None:
            logger.warning(f"Please run evaluate('{det_type}') first.")

        class_ids = self.params[det_type].class_ids
        cls_id_2_idx = {x: i for i, x in enumerate(class_ids)}

        num_thrs = len(self.params[det_type].iou_thrs)
        num_recalls = len(self.params[det_type].rec_thrs)
        num_classes = len(class_ids)
        num_area_rngs = len(self.params[det_type].area_rng)
        num_backgrounds = len(self.meta.background)
        num_lightings = len(self.meta.lighting)
        num_difficulties = len(self.params[det_type].difficulty)

        # -1 for absent classes
        precision = -np.ones(
            (
                num_thrs,
                num_recalls,
                num_classes,
                num_area_rngs,
                num_backgrounds,
                num_lightings,
                num_difficulties,
            )
        )
        recall = -np.ones(
            (
                num_thrs,
                num_classes,
                num_area_rngs,
                num_backgrounds,
                num_lightings,
                num_difficulties,
            )
        )
        # recall_at_precision = -np.ones(
        #     (
        #         num_thrs,
        #         num_classes,
        #         num_area_rngs,
        #         num_backgrounds,
        #         num_lightings,
        #         num_difficulties,
        #     )
        # )
        # recall_at_precision_metric = RecallAtPrecision(
        #     self.params[det_type].recall_at_precision_k
        # )

        # Initialize dt_pointers
        dt_pointers = {}
        for cls_idx in range(num_classes):
            dt_pointers[cls_idx] = {}
            for area_idx in range(num_area_rngs):
                dt_pointers[cls_idx][area_idx] = {}
                for bg_idx in range(num_backgrounds):
                    dt_pointers[cls_idx][area_idx][bg_idx] = {}
                    for lt_idx in range(num_lightings):
                        dt_pointers[cls_idx][area_idx][bg_idx][lt_idx] = {}
                        for df_idx in range(num_difficulties):
                            dt_pointers[cls_idx][area_idx][bg_idx][lt_idx][df_idx] = {}

        results = defaultdict(list)
        for res in self.eval_imgs[det_type]:
            if res is not None:
                img_id, class_id, area, background, lighting, difficulty = res["config"]
                cls_idx = cls_id_2_idx[class_id]
                area_idx = self.params[det_type].area_rng_lbl.index(area)
                bg_idx = self.meta.background.index(background)
                lt_idx = self.meta.lighting.index(lighting)
                df_idx = self.params[det_type].difficulty.index(difficulty)
                results[(cls_idx, area_idx, bg_idx, lt_idx, df_idx)].append(
                    (img_id, res)
                )

        for config, E in results.items():
            cls_idx, area_idx, bg_idx, lt_idx, df_idx = config
            E = [x[1] for x in E]

            # Append all scores: shape (N,)
            dt_scores = np.concatenate([e["dt_scores"] for e in E], axis=0)
            dt_ids = np.concatenate([e["dt_ids"] for e in E], axis=0)

            dt_idx = np.argsort(-dt_scores, kind="mergesort")
            dt_scores = dt_scores[dt_idx]
            dt_ids = dt_ids[dt_idx]

            dt_m = np.concatenate([e["dt_matches"] for e in E], axis=1)[:, dt_idx]
            dt_ig = np.concatenate([e["dt_ignore"] for e in E], axis=1)[:, dt_idx]

            gt_ig = np.concatenate([e["gt_ignore"] for e in E])
            # num gt anns to consider
            num_gt = np.count_nonzero(gt_ig == 0)

            if num_gt == 0:
                continue

            tps = np.logical_and(dt_m, np.logical_not(dt_ig))
            fps = np.logical_and(np.logical_not(dt_m), np.logical_not(dt_ig))

            tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
            fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)

            dt_pointers[cls_idx][area_idx][bg_idx][lt_idx][df_idx] = {
                "dt_ids": dt_ids,
                "tps": tps,
                "fps": fps,
            }

            for iou_thr_idx, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                tp = np.array(tp)
                fp = np.array(fp)
                num_tp = len(tp)
                rc = tp / num_gt
                if num_tp:
                    recall[iou_thr_idx, cls_idx, area_idx, bg_idx, lt_idx, df_idx] = rc[
                        -1
                    ]
                else:
                    recall[iou_thr_idx, cls_idx, area_idx, bg_idx, lt_idx, df_idx] = 0

                # np.spacing(1) ~= eps
                pr = tp / (fp + tp + np.spacing(1))
                pr = pr.tolist()

                # Replace each precision value with the maximum precision
                # value to the right of that recall level. This ensures
                # that the  calculated AP value will be less suspectable
                # to small variations in the ranking.
                for i in range(num_tp - 1, 0, -1):
                    if pr[i] > pr[i - 1]:
                        pr[i - 1] = pr[i]

                rec_thrs_insert_idx = np.searchsorted(
                    rc, self.params[det_type].rec_thrs, side="left"
                )

                pr_at_recall = [0.0] * num_recalls

                # we need to use "try-except" clause because for some high recall threshold,
                # the "pr_idx" == len(pr)
                try:
                    for _idx, pr_idx in enumerate(rec_thrs_insert_idx):
                        pr_at_recall[_idx] = pr[pr_idx]
                except BaseException:
                    pass
                precision[
                    iou_thr_idx, :, cls_idx, area_idx, bg_idx, lt_idx, df_idx
                ] = np.array(pr_at_recall)
                # Compute recall_at_precision below
                dt_ig_i = dt_ig[iou_thr_idx]
                dt_scores_i = dt_scores[np.logical_not(dt_ig_i)]
                dt_m_i = dt_m[iou_thr_idx, np.logical_not(dt_ig_i)]
                dt_true_i = np.greater(dt_m_i, 0)

                # recall_at_precision_metric.reset_state()
                # recall_at_precision_metric.update_state(
                #     torch.from_numpy(dt_true_i).to(torch.float32),
                #     torch.from_numpy(dt_scores_i).to(torch.float32),
                #     num_gt,
                # )
                # res = recall_at_precision_metric.result()
                # recall_at_precision[
                #     iou_thr_idx, cls_idx, area_idx, bg_idx, lt_idx, df_idx
                # ] = res

        self.eval[det_type] = {
            "params": self.params[det_type],
            "counts": [
                num_thrs,
                num_recalls,
                num_classes,
                num_area_rngs,
                num_backgrounds,
                num_lightings,
            ],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "recall": recall,
            "dt_pointers": dt_pointers,
            # "recall_at_precision": recall_at_precision,
        }

    def _summarize(
        self,
        det_type: str,
        summary_type: str,
        iou_thr: Optional[List[float]] = None,
        area_rng: str = "all",
        background: str = "all",
        lighting: str = "all",
        difficulty: str = "all",
        group: Optional[str] = None,
        topk=None,
    ):
        group_idx = None
        if group is not None and group != "all":
            if det_type == "cat_det":
                freq_group_idx = self.params[det_type].img_count_lbl.index(group)
                group_idx = self.freq_groups[det_type][freq_group_idx]
            else:
                assert group in {"seen", "unseen"}
                if group == "seen":
                    group_idx = self.inst_det_seen_unseen_cat_groups[0]
                else:
                    group_idx = self.inst_det_seen_unseen_cat_groups[1]

        aidx = np.array(
            [
                idx
                for idx, _area_rng in enumerate(self.params[det_type].area_rng_lbl)
                if _area_rng == area_rng
            ]
        )
        bidx = np.array(
            [
                idx
                for idx, _background in enumerate(self.meta.background)
                if _background == background
            ]
        )
        lidx = np.array(
            [
                idx
                for idx, _lighting in enumerate(self.meta.lighting)
                if _lighting == lighting
            ]
        )
        didx = np.array(
            [
                idx
                for idx, _difficulty in enumerate(self.params[det_type].difficulty)
                if _difficulty == difficulty
            ]
        )

        for idx in [aidx, bidx, lidx, didx]:
            if idx.size <= 0:
                return -1

        tidx = None
        if iou_thr is not None:
            iou_thr_to_idx = {
                x: i for i, x in enumerate(self.params[det_type].iou_thrs)
            }
            tidx = np.array([iou_thr_to_idx[x] for x in iou_thr]).astype(np.int64)

        if summary_type == "ap":
            s = self.eval[det_type]["precision"]
            if tidx is not None:
                s = s[tidx]
            if group_idx is not None:
                s = s[:, :, group_idx, aidx, bidx, lidx, didx]
            else:
                s = s[:, :, :, aidx, bidx, lidx, didx]
        elif summary_type == "ar":
            s = self.eval[det_type]["recall"]
            if tidx is not None:
                s = s[tidx]
            s = s[:, :, aidx, bidx, lidx, didx]
        elif summary_type == "r@p":
            s = self.eval[det_type]["recall_at_precision"]
            if tidx is not None:
                s = s[tidx]
            if group_idx is not None:
                s = s[:, group_idx, aidx, bidx, lidx, didx]
            else:
                s = s[:, :, aidx, bidx, lidx, didx]
            if topk is not None:
                sorted_s = -np.sort(-s, axis=1)
                s = sorted_s[:, :topk]
        else:
            raise ValueError(f"unknown summary type {summary_type}")

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        return mean_s

    def summarize(self, det_type: str, metric_filter: List[Dict]):
        if not self.eval[det_type]:
            raise RuntimeError("Please run accumulate() first.")

        logger.info(f"Summarize detection results for {det_type}.")

        max_dets = self.params[det_type].max_dets
        coco_iou_thres = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        coco_iou_thres_str = f"{coco_iou_thres[0]:0.2f}:{coco_iou_thres[-1]:0.2f}"

        results = self.results[det_type]

        for metric in metric_filter:
            key = "AP"
            for k in ["iou", "gr", "ar", "bg", "lt", "df"]:
                if k == "iou":
                    if metric[k] != "coco":
                        key = key + metric[k]
                elif metric[k] != "all":
                    key = key + "-" + f"{k}{metric[k]}"

            results[key] = {
                "title": "AP",
                "iou_thres": coco_iou_thres_str
                if metric["iou"] == "coco"
                else int(metric["iou"]) / 100.0,
                "area_rng": metric["ar"],
                "background": metric["bg"],
                "lighting": metric["lt"],
                "difficulty": metric["df"],
                "cat_group_name": metric["gr"],
                "value": self._summarize(
                    det_type,
                    "ap",
                    iou_thr=coco_iou_thres
                    if metric["iou"] == "coco"
                    else [int(metric["iou"]) / 100.0],
                    group=metric["gr"],
                    area_rng=metric["ar"],
                    background=metric["bg"],
                    lighting=metric["lt"],
                    difficulty=metric["df"],
                ),
            }

        # for metric in metric_filter:
        #     key = f"R@P{int(self.params[det_type].recall_at_precision_k * 100):02d}"
        #     for k in ["iou", "gr", "ar", "bg", "lt", "df"]:
        #         if k == "iou":
        #             if metric[k] != "coco":
        #                 key = key + "-" + metric[k]
        #         elif metric[k] != "all":
        #             key = key + "-" + f"{k}{metric[k]}"

        #     results[key] = {
        #         "title": "R@P",
        #         "iou_thres": coco_iou_thres_str
        #         if metric["iou"] == "coco"
        #         else int(metric["iou"]) / 100.0,
        #         "area_rng": metric["ar"],
        #         "background": metric["bg"],
        #         "lighting": metric["lt"],
        #         "difficulty": metric["df"],
        #         "cat_group_name": metric["gr"],
        #         "value": self._summarize(
        #             det_type,
        #             "r@p",
        #             iou_thr=coco_iou_thres
        #             if metric["iou"] == "coco"
        #             else [int(metric["iou"]) / 100.0],
        #             group=metric["gr"],
        #             area_rng=metric["ar"],
        #             background=metric["bg"],
        #             lighting=metric["lt"],
        #             difficulty=metric["df"],
        #         ),
        #     }

        key = f"AR50@{max_dets}"
        results[key] = {
            "title": "AR",
            "iou_thres": 0.5,
            "area_rng": "all",
            "background": "all",
            "lighting": "all",
            "cat_group_name": "all",
            "value": self._summarize(det_type, "ar", iou_thr=[0.5]),
        }

    def print_results(self):
        for det_type in self.eval_type:
            logger.info(f"print results for {det_type}")
            self._print_results(det_type)

    def _print_results(self, det_type: str):
        template = " {:<12} @[ IoU={:<9} | area={:>6s} | background={:>6s} | lighting={:>6s} | maxDets={:>3d} catIds={:>3s}] = {:0.3f}"

        for _key, value in self.results[det_type].items():
            max_dets = self.params[det_type].max_dets

            logger.info(
                template.format(
                    value["title"],
                    value["iou_thres"],
                    value["area_rng"],
                    value["background"],
                    value["lighting"],
                    max_dets,
                    value["cat_group_name"] + f"-top{value.get('topk', '')}",
                    value["value"],
                )
            )

    def _get_gt_dt(self, det_type, img_id, class_id):
        """Create gt, dt which are list of anns/dets."""
        gt = self.gts[det_type][img_id, class_id]
        dt = self.dts[det_type][img_id, class_id]
        return gt, dt

    def compute_iou(self, det_type: str, img_id: int, class_id: int):
        gt, dt = self._get_gt_dt(det_type, img_id, class_id)

        if len(gt) == 0 and len(dt) == 0:
            return []

        # Sort detections in decreasing order of score.
        idx = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in idx]

        ann_type = "bbox"
        gt = [g[ann_type] for g in gt]
        dt = [d[ann_type] for d in dt]

        # compute iou between each dt and gt region
        # will return array of shape len(dt), len(gt)
        iscrowd = [int(False)] * len(gt)
        ious = mask_utils.iou(dt, gt, iscrowd)
        return ious

    def _prepare(self, det_type: str):
        img_ids = self.params[det_type].img_ids
        class_ids = self.params[det_type].class_ids

        logger.info(f"{det_type}, len params img_ids {len(img_ids)}")
        logger.info(f"{det_type}, len params class_ids {len(class_ids)}")

        gts = self.gt.load_anns(
            det_type,
            self.gt.get_ann_ids(det_type, img_ids=img_ids, class_ids=class_ids),
        )
        dts = self.dt.load_anns(
            det_type,
            self.dt.get_ann_ids(det_type, img_ids=img_ids, class_ids=class_ids),
        )

        logger.info(f"{det_type}, len gts {len(gts)}")
        logger.info(f"{det_type}, len dts {len(dts)}")

        # set ignore flag
        for gt in gts:
            if "ignore" not in gt:
                gt["ignore"] = 0

        for gt in gts:
            class_key = "category_id" if det_type == "cat_det" else "instance_id"
            self.gts[det_type][gt["image_id"], gt[class_key]].append(gt)

            # asscociate image meta info to each gt
            img = self.gt.imgs[gt["image_id"]]
            img_meta = self.meta.video_id_to_setting[img["video_id"]]
            for key in ["background", "lighting"]:
                gt[key] = img_meta[key]

            area_ratio = math.sqrt(gt["area"] / (img["height"] * img["width"]))
            if area_ratio < 0.1:
                gt["area_ratio"] = "small"
            elif area_ratio < 0.2:
                gt["area_ratio"] = "medium"
            else:
                gt["area_ratio"] = "large"

            if det_type == "inst_det":
                # [Easy] -- register and detect on simple
                # [Medium] -- register on busy, detect on simple; register and detect on busy
                # [Hard] -- register on simple, detect on busy
                instance_id = gt["instance_id"]
                # the background for the instance that's used for registration
                if (
                    hasattr(self.gt.metadata, "instance_register_bg")
                    and instance_id in self.gt.metadata.instance_register_bg
                ):
                    register_bg = self.gt.metadata.instance_register_bg[instance_id]
                    query_bg = self.meta.video_id_to_setting[img["video_id"]][
                        "background"
                    ]
                    if register_bg == "simple" and query_bg == "simple":
                        gt["difficulty"] = "easy"
                    elif register_bg == "simple" and query_bg == "busy":
                        gt["difficulty"] = "hard"
                    else:
                        gt["difficulty"] = "medium"
                else:
                    logger.warning(f"instance_id={instance_id} is not registered!")

        # For federated dataset evaluation we will filter out all dt for an
        # image which belong to classes not present in gt and not present in
        # the negative list for an image. In other words detector is not penalized
        # for classes about which we don't have gt information about their
        # presence or absence in an image.
        img_data = self.gt.load_imgs(ids=self.params[det_type].img_ids)
        # per image map of classes not present in image
        neg_class_ids_key = (
            "neg_category_ids" if det_type == "cat_det" else "neg_instance_ids"
        )
        img_nl = {d["id"]: d[neg_class_ids_key] for d in img_data}
        # per image list of classes present in image
        img_pl = defaultdict(set)
        class_id_key = "category_id" if det_type == "cat_det" else "instance_id"
        for ann in gts:
            img_pl[ann["image_id"]].add(ann[class_id_key])
        # per image map of classes which have missing gt. For these
        # classes we don't penalize the detector for false positives.
        self.img_nel[det_type] = {
            d["id"]: d["not_exhaustive_category_ids"]
            if det_type == "cat_det" and "not_exhaustive_category_ids" in d
            else []
            for d in img_data
        }

        for dt in dts:
            img_id, class_id = dt["image_id"], dt[class_id_key]

            if class_id not in img_nl[img_id] and class_id not in img_pl[img_id]:
                continue

            # asscociate image meta info to each dt
            img = self.gt.imgs[dt["image_id"]]
            img_meta = self.meta.video_id_to_setting[img["video_id"]]
            for key in ["background", "lighting"]:
                dt[key] = img_meta[key]

            area_ratio = math.sqrt(dt["area"] / (img["height"] * img["width"]))
            if area_ratio < 0.1:
                dt["area_ratio"] = "small"
            elif area_ratio < 0.2:
                dt["area_ratio"] = "medium"
            else:
                dt["area_ratio"] = "large"

            if det_type == "inst_det":
                # [Easy] -- register and detect on simple
                # [Medium] -- register on busy, detect on simple; register and detect on busy
                # [Hard] -- register on simple, detect on busy
                instance_id = dt["instance_id"]
                # the background for the instance that's used for registration
                if (
                    hasattr(self.gt.metadata, "instance_register_bg")
                    and instance_id in self.gt.metadata.instance_register_bg
                ):
                    register_bg = self.gt.metadata.instance_register_bg[instance_id]
                    query_bg = self.meta.video_id_to_setting[img["video_id"]][
                        "background"
                    ]
                    if register_bg == "simple" and query_bg == "simple":
                        dt["difficulty"] = "easy"
                    elif register_bg == "simple" and query_bg == "busy":
                        dt["difficulty"] = "hard"
                    else:
                        dt["difficulty"] = "medium"
                else:
                    logger.warning(f"instance_id={instance_id} is not registered!")
            self.dts[det_type][img_id, class_id].append(dt)

        self.freq_groups[det_type] = self._prepare_freq_group(det_type)

        if det_type == "inst_det":
            self.inst_det_seen_unseen_cat_groups = (
                self._prepare_seen_unseen_cat_groups()
            )

    def _prepare_freq_group(self, det_type: str):
        freq_groups = [[] for _ in self.params[det_type].img_count_lbl]
        class_data = self.gt.load_classes(det_type, self.params[det_type].class_ids)
        for idx, _class_data in enumerate(class_data):
            if "frequency" in _class_data:
                frequency = _class_data["frequency"]
            else:
                # assign all sample to frequent group if not specified
                frequency = "frequent"
            freq_groups[self.params[det_type].img_count_lbl.index(frequency)].append(
                idx
            )

        return freq_groups

    def _prepare_seen_unseen_cat_groups(self):
        det_type = "inst_det"
        # 2 groups in total, including "seen" and "unseen" groups
        seen_unseen_groups = [[], []]
        class_data = self.gt.load_classes(det_type, self.params[det_type].class_ids)

        logger.info(f"num cat det categories {len(self.gt.classes['cat_det'])}")

        for idx, _class_data in enumerate(class_data):
            # Object categories consideted by category detection are common categories between
            # train and val split.
            group_id = (
                0
                if "category_id" in _class_data
                and _class_data["category_id"] in self.gt.classes["cat_det"]
                else 1
            )
            seen_unseen_groups[group_id].append(idx)

        for group_id, group in enumerate(seen_unseen_groups):
            logger.info(f"seen/unseen group_id {group_id}, group size {len(group)}")

        return seen_unseen_groups

    def get_results(self):
        return {det_type: self._get_results(det_type) for det_type in self.eval_type}

    def _get_results(self, det_type: str):
        if len(self.results[det_type]) == 0:
            logger.warning(f"{det_type} results is empty. Call run().")
        return self.results[det_type]

    def log_per_class_results(self, output_dir):
        if output_dir:
            det_type = "cat_det"
            iou_thres = 0.5
            rec_at_prec_k = int(self.params[det_type].recall_at_precision_k * 100)
            recall_at_prec = self.eval[det_type]["recall_at_precision"]
            precision = self.eval[det_type]["precision"]

            for area_rng in self.params[det_type].area_rng_lbl:
                aidx = [
                    idx
                    for idx, _area_rng in enumerate(self.params[det_type].area_rng_lbl)
                    if _area_rng == area_rng
                ][0]
                tidx = np.where(iou_thres == self.params[det_type].iou_thrs)[0].item()
                bidx = [
                    idx
                    for idx, _background in enumerate(self.meta.background)
                    if _background == "all"
                ][0]
                lidx = [
                    idx
                    for idx, _lighting in enumerate(self.meta.lighting)
                    if _lighting == "all"
                ][0]
                didx = [
                    idx
                    for idx, _difficulty in enumerate(self.params[det_type].difficulty)
                    if _difficulty == "all"
                ][0]

                # print per-category R@P stats
                cur_recall_at_prec = recall_at_prec[
                    tidx, :, aidx, bidx, lidx, didx
                ].reshape(-1)
                sort_idx = np.argsort(-cur_recall_at_prec)

                lines = []
                for idx in sort_idx.tolist():
                    r_at_p = cur_recall_at_prec[idx]
                    cat = self.gt.cats[det_type][self.params[det_type].class_ids[idx]]
                    lines.append(
                        "{},{},{},{:0.2f}".format(
                            cat["name"],
                            cat["image_count"] if "image_count" in cat else 0,
                            cat["instance_count"] if "instance_count" in cat else 0,
                            r_at_p,
                        )
                    )

                key = "R@P{}-{}-{}.csv".format(
                    rec_at_prec_k, int(iou_thres * 100), area_rng
                )
                with open(os.path.join(output_dir, key), "w") as h:
                    h.write("\n".join(lines))

                # print per-category AP50 stats
                cur_precision = precision[tidx, :, :, aidx, bidx, lidx, didx]
                cur_precision = np.mean(cur_precision, axis=0)
                sort_idx = np.argsort(-cur_precision)

                lines = []
                for idx in sort_idx.tolist():
                    ap50 = cur_precision[idx]
                    cat = self.gt.cats[det_type][self.params[det_type].class_ids[idx]]
                    lines.append(
                        "{},{},{},{:0.2f}".format(
                            cat["name"],
                            cat["image_count"] if "image_count" in cat else 0,
                            cat["instance_count"] if "instance_count" in cat else 0,
                            ap50,
                        )
                    )

                key = "AP{}-{}.csv".format(int(iou_thres * 100), area_rng)
                with open(os.path.join(output_dir, key), "w") as h:
                    h.write("\n".join(lines))


class CategoryDetectionParams:
    def __init__(self, max_dets: int = 100):
        """CategoryDetectionParams for EgoObjects evaluation API."""
        self.img_ids = []
        self.class_ids = []
        # np.arange causes trouble.  the data point on arange is slightly
        # larger than the true value
        # self.iou_thrs = np.linspace(
        #     0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        # )
        self.iou_thrs = np.array(
            [0.1, 0.25, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        )
        self.rec_thrs = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )
        self.max_dets = max_dets
        self.area_rng = [
            [0**2, 1e5**2],
            [0**2, 32**2],
            [32**2, 96**2],
            [96**2, 1e5**2],
        ]
        self.area_rng_lbl = ["all", "small", "medium", "large"]
        # self.use_cats = 1
        # We bin classes in three bins based how many images of the training
        # set the category is present in.
        # r: Rare    :  < 10
        # c: Common  : >= 10 and < 100
        # f: Frequent: >= 100
        self.img_count_lbl = ["rare", "common", "frequent"]
        self.difficulty = ["all"]

        self.topk_classes = [100, 200]
        self.recall_at_precision_k = 0.9


class InstanceDetectionParams:
    def __init__(self, max_dets: int = 100):
        """InstanceDetectionParams for EgoObjects evaluation API."""
        self.img_ids = []
        self.class_ids = []
        # np.arange causes trouble.  the data point on arange is slightly
        # larger than the true value
        # self.iou_thrs = np.linspace(
        #     0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        # )
        self.iou_thrs = np.array(
            [0.1, 0.25, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        )
        self.rec_thrs = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )
        self.max_dets = max_dets
        self.area_rng = [
            [0**2, 1e5**2],
            [0**2, 32**2],
            [32**2, 96**2],
            [96**2, 1e5**2],
        ]
        self.area_rng_lbl = ["all", "small", "medium", "large"]
        # self.use_cats = 1
        # We bin classes in three bins based how many images of the training
        # set the category is present in.
        # r: Rare    :  < 10
        # c: Common  : >= 10 and < 100
        # f: Frequent: >= 100
        self.img_count_lbl = ["rare", "common", "frequent"]
        self.difficulty = ["all", "easy", "medium", "hard"]

        self.topk_classes = [400, 800, 1200]
        self.recall_at_precision_k = 0.9
