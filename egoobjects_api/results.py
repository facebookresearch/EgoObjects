#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import defaultdict
from copy import deepcopy
from typing import List, Dict, Any

from .egoobjects import EgoObjects

logger = logging.getLogger(__name__)


class EgoObjectsResults(EgoObjects):
    def __init__(
        self,
        gt: EgoObjects,
        cat_det_dt_anns: List[Dict[str, Any]],
        inst_det_dt_anns: List[Dict[str, Any]],
        max_dets: int = 300,
    ):
        """Constructor for EgoObjects results.
        Args:
            gt: EgoObjects class instance
            cat_det_dt_anns: detected bounding boxes for category detection
            inst_det_dt_anns: detected bounding boxes for instance detection
            max_dets: max number of detections per image.
        """
        logger.info(f"num category detections {len(cat_det_dt_anns)}")
        logger.info(f"num instance detections {len(inst_det_dt_anns)}")

        self.dataset = deepcopy(gt.dataset)

        dt_anns = {}

        dt_anns["cat_det"] = (
            self.limit_detections_per_image(cat_det_dt_anns, max_dets)
            if max_dets >= 0
            else cat_det_dt_anns
        )
        dt_anns["inst_det"] = (
            self.limit_detections_per_image(inst_det_dt_anns, max_dets)
            if max_dets >= 0
            else inst_det_dt_anns
        )

        logger.info(
            f"after limit detections per image, len inst_det {len(dt_anns['inst_det'])}"
        )

        for _k, anns in dt_anns.items():
            if len(anns) > 0:
                assert "bbox" in anns[0]
            for id, ann in enumerate(anns):
                _x1, _y1, w, h = ann["bbox"]
                ann["area"] = w * h
                ann["id"] = id + 1

        self.annotations = dt_anns
        self._create_index(gt.metadata)

        # cat_det_dt_anns can be empty when we do not do category detection in the model.
        if len(cat_det_dt_anns) > 0:
            cat_det_img_ids_in_result = [ann["image_id"] for ann in cat_det_dt_anns]

            assert set(cat_det_img_ids_in_result) == (
                set(cat_det_img_ids_in_result) & set(self.get_img_ids())
            ), "Results do not correspond to current EgoObjects dataset."

    def limit_detections_per_image(self, anns, max_dets):
        img_ann = defaultdict(list)
        for ann in anns:
            img_ann[ann["image_id"]].append(ann)

        for img_id, _anns in img_ann.items():
            if len(_anns) <= max_dets:
                continue
            _anns = sorted(_anns, key=lambda ann: ann["score"], reverse=True)
            img_ann[img_id] = _anns[:max_dets]

        return [ann for anns in img_ann.values() for ann in anns]
