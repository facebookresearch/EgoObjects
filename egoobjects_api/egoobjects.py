#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
from collections import defaultdict
from copy import deepcopy
from typing import List, Optional, Dict, Any

from detectron2.data.catalog import Metadata

logger = logging.getLogger(__name__)


FILTER_OPTS = {
    # instance detection
    "egoobjects_unified_det_train": {},
    "egoobjects_unified_det_test_query": {
        "subset": "test",
    },
    "egoobjects_unified_det_val_query": {
        "subset": "val",
    },
    # category detection
    "egoobjects_cat_det_train": {
        "remove_non_category": True,
    },
    "egoobjects_cat_det_test": {
        "subset": "test",
        "remove_non_category": True,
    },
    "egoobjects_cat_det_val": {
        "subset": "val",
        "remove_non_category": True,
    },
    # instance detection with seen/unseen category
    "egoobjects_instdet_train": {},
    "egoobjects_instdet_test_query": {
        "subset": "test",
    },
    "egoobjects_instdet_val_query": {
        "subset": "val",
    },
}


def filter_annot(
    data,
    metadata,
    filter_opts,
):
    if filter_opts is None:
        filter_opts = {}

    valid_image_set = set([x["id"] for x in data["images"]])
    # filter according to easy/hard splits
    if "difficulty" in filter_opts and filter_opts["difficulty"]:
        selected_image_set = set(
            [
                x["id"]
                for x in data["images"]
                if x["difficulty"] == filter_opts["difficulty"]
            ]
        )
        valid_image_set = valid_image_set & selected_image_set

    # filter according to minival splits
    if "subset" in filter_opts and filter_opts["subset"]:
        selected_image_set = set(
            [x["id"] for x in data["images"] if x["subset"] == filter_opts["subset"]]
        )
        valid_image_set = valid_image_set & selected_image_set

    # filter out annotations/images without category_id field
    if "remove_non_category" in filter_opts and filter_opts["remove_non_category"]:
        if isinstance(metadata, Dict):
            cat_det_cats = metadata["cat_det_cats"]
        else:
            cat_det_cats = metadata.cat_det_cats
        kept_annot = []
        kept_image_id = set()
        kept_cat_ids = set([x["id"] for x in cat_det_cats])
        for anno in data["annotations"]:
            if (
                "category_id" in anno
                and anno["category_id"] in kept_cat_ids
                and anno["image_id"] in valid_image_set
            ):
                kept_annot.append(anno)
                kept_image_id.add(anno["image_id"])
        data["annotations"] = kept_annot
        data["images"] = [x for x in data["images"] if x["id"] in kept_image_id]
    else:
        kept_annot = []
        kept_image_id = set()
        for anno in data["annotations"]:
            if anno["image_id"] in valid_image_set:
                kept_annot.append(anno)
                kept_image_id.add(anno["image_id"])
        data["annotations"] = kept_annot
        data["images"] = [x for x in data["images"] if x["id"] in kept_image_id]

    return data


class EgoObjects:
    def __init__(
        self,
        annotation_path: str,
        metadata: Metadata,
        filter_opts: Any = None,
    ):
        """
        Args:
            annotation_path: location of annotation file
        """
        logger.info(f"annotation_path {annotation_path}")
        self.metadata = metadata

        with open(annotation_path, "r") as f:
            self.dataset = json.load(f)

        # is_valid_video_id = self._valid_video_ids()
        # if not is_valid_video_id:
        #     self._replace_video_ids()

        # filter the dataset accordingly
        self.dataset = filter_annot(
            self.dataset,
            metadata,
            filter_opts,
        )

        assert (
            type(self.dataset) == dict
        ), f"Annotation file format {type(self.dataset)} not supported."

        self.annotations = {
            "cat_det": [
                deepcopy(ann)
                for ann in self.dataset["annotations"]
                if "category_id" in ann
            ],
            "inst_det": [
                deepcopy(ann)
                for ann in self.dataset["annotations"]
                if "instance_id" in ann
            ],
        }

        logger.info(f"num cat det instances {len(self.annotations['cat_det'])}")
        logger.info(f"num inst det instances {len(self.annotations['inst_det'])}")

        self._create_index(metadata)

    def _valid_video_ids(self):
        """dummy check on whether video ids lie in existing video_id """
        video_ids_in_setting = {"01", "02", "03", "04", "05", "06", "07", "08", "09", "10"}
        video_ids_in_dataset = set([img['video_id'] for img in self.dataset['images']])
        return video_ids_in_dataset.issubset(video_ids_in_setting)

    def _replace_video_ids(self):
        """
        To align the `video_id` for *ego-object dataset towards existing dataset.
        Rules:
            {'1', '2', '3'} maps into {'01', '02', '03'},
            while others {'V1', 'V2', 'V26', 'V28', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8'} map into {'04'}
        """
        star_to_existing_vid_mapping = {'1': '01', '2': '02', '3': '03'} # hard-coded replacement for easy videos
        mapping_key_set = set(star_to_existing_vid_mapping.keys())
        for img in self.dataset['images']:
            if img['video_id'] in mapping_key_set:
                img.update({'video_id': star_to_existing_vid_mapping[img['video_id']]})
            else:
                img.update({'video_id':'04'}) # hard-coded replacement for complex videos

        logger.info("VideoID in *ego-object updated to match existing dataset.")
        return

    def _prepare_neg_instance_ids(self):
        img_id_2_instance_id = {
            img_id: {self.anns["inst_det"][ann_id]["instance_id"] for ann_id in ann_ids}
            for img_id, ann_ids in self.img_ann_map["inst_det"].items()
        }
        for img_id, img in self.imgs.items():
            if img_id in img_id_2_instance_id:
                img["neg_instance_ids"] = self.instance_ids.difference(
                    img_id_2_instance_id[img_id]
                )
            else:
                img["neg_instance_ids"] = self.instance_ids

    def _prepare_neg_cat_ids(self):
        img_id_2_category_id = {
            img_id: {self.anns["cat_det"][ann_id]["category_id"] for ann_id in ann_ids}
            for img_id, ann_ids in self.img_ann_map["cat_det"].items()
        }
        for img_id, img in self.imgs.items():
            if "neg_category_ids" in img:
                continue
            elif img_id in img_id_2_category_id:
                img["neg_category_ids"] = self.category_ids.difference(
                    img_id_2_category_id[img_id]
                )
            else:
                img["neg_category_ids"] = self.category_ids

    def _create_index(self, metadata):
        logger.info("Creating index.")

        self.cat_id_2_cat = {c["id"]: c for c in metadata.categories}
        self.imgs = {img["id"]: img for img in self.dataset["images"]}
        self.anns = defaultdict(dict)
        self.img_ann_map = {"cat_det": defaultdict(list), "inst_det": defaultdict(list)}
        for det_type, anns in self.annotations.items():
            for ann in anns:
                self.anns[det_type][ann["id"]] = ann
                self.img_ann_map[det_type][ann["image_id"]].append(ann["id"])

            logger.info(
                f"{det_type}, len img_ann_map {len(self.img_ann_map[det_type])}"
            )

        # self.category_ids = {x["id"] for x in metadata.cat_det_cats}
        self.category_ids = {ann["category_id"] for ann in self.annotations["cat_det"]}
        self.instance_ids = {ann["instance_id"] for ann in self.annotations["inst_det"]}

        self._prepare_neg_instance_ids()
        self._prepare_neg_cat_ids()

        self.cats = {
            "cat_det": {c["id"]: c for c in metadata.cat_det_cats},
            "inst_det": {c["id"]: c for c in metadata.inst_det_cats},
        }

        self.classes = {
            "cat_det": {c["id"]: c for c in metadata.cat_det_cats},
            "inst_det": {},
        }
        for _i, ann in enumerate(self.annotations["inst_det"]):
            inst_id = ann["instance_id"]
            cat_id = ann["category_id"] if "category_id" in ann else None

            if inst_id not in self.classes["inst_det"]:
                inst_dict = {"id": inst_id}
                if cat_id is not None:
                    if "frequency" in self.cat_id_2_cat[cat_id]:
                        frequency = self.cat_id_2_cat[cat_id]["frequency"]
                    else:
                        # assign all sample to frequent group if not specified
                        frequency = "frequent"
                    inst_dict.update(
                        {
                            "category_id": cat_id,
                            "frequency": frequency,
                        }
                    )

                self.classes["inst_det"][inst_id] = inst_dict

            else:
                if cat_id is not None:
                    assert self.classes["inst_det"][inst_id]["category_id"] == cat_id

        logger.info(f"num total images: {len(self.imgs)}")
        for det_type in ["cat_det", "inst_det"]:
            logger.info(f"num images for {det_type}: {len(self.img_ann_map[det_type])}")
            logger.info(f"num annotations for {det_type} {len(self.anns[det_type])}")
            logger.info(
                f"num object categories of {det_type} {len(self.cats[det_type])}"
            )
            logger.info(f"num classes of {det_type} {len(self.classes[det_type])}")

        logger.info("Index created.")

    def get_img_ids(self):
        """Get all img ids.

        Returns:
            ids (int array): integer array of image ids
        """
        return list(self.imgs.keys())

    def get_class_ids(self, det_type: str):
        """Get all category ids of category detection.
        Args:
            det_type: detection type. Choices {"cat_det", "inst_det"}
        Returns:
            ids: integer array of category ids
        """
        return list(self.classes[det_type].keys())

    def get_ann_ids(
        self,
        det_type: str,
        img_ids: Optional[List[int]] = None,
        class_ids: Optional[List[int]] = None,
    ) -> List[int]:
        """Get ann ids that satisfy given filter conditions.

        Args:
            det_type: detection type. Choices {"cat_det", "inst_det"}
            img_ids: get anns for given imgs
            class_ids: get anns for given class ids, which are category ids for "cat_det"
                and instance ids for "inst_det"
        Returns:
            ids: integer array of ann ids
        """
        assert det_type in self.img_ann_map
        anns = []
        if img_ids is not None:
            for img_id in img_ids:
                if img_id in self.img_ann_map[det_type]:
                    anns.extend(
                        [
                            self.anns[det_type][ann_id]
                            for ann_id in self.img_ann_map[det_type][img_id]
                        ]
                    )
        else:
            anns = self.annotations[det_type]
        # return early if no more filtering required
        if class_ids is None:
            return [ann["id"] for ann in anns]

        class_ids = set(class_ids)

        ann_ids = [
            _ann["id"]
            for _ann in anns
            if _ann["category_id" if det_type == "cat_det" else "instance_id"]
            in class_ids
        ]

        return ann_ids

    def _load_helper(self, _dict, ids):
        if ids is None:
            return list(_dict.values())
        else:
            return [_dict[id] for id in ids]

    def load_anns(self, det_type: str, ids: Optional[List[int]] = None):
        """Load anns with the specified ids. If ids=None load all anns.

        Args:
            det_type: detection type. Choices {"cat_det", "inst_det"}
            ids: integer array of annotation ids

        Returns:
            anns: loaded annotation objects
        """
        return self._load_helper(self.anns[det_type], ids)

    def load_classes(self, det_type: str, ids: Optional[List[int]] = None):
        """Load classes  with the specified ids.
        If ids=None load all classes.

        Args:
            det_type: detection type. Choices {"cat_det", "inst_det"}
            ids: integer array of class ids

        Returns:
            classes: loaded class dicts
        """
        return self._load_helper(self.classes[det_type], ids)

    def load_imgs(self, ids: Optional[List[int]] = None):
        """Load categories with the specified ids. If ids=None load all images.

        Args:
            ids: integer array of image ids

        Returns:
            imgs: loaded image dicts
        """
        return self._load_helper(self.imgs, ids)


class EgoObjectsMetaInfo:
    def __init__(self):
        self.video_id_to_setting = {
            "01": {
                "distance": "near",
                "camera motion": "horizontal",
                "background": "simple",
                "lighting": "bright",
            },
            "02": {
                "distance": "medium",
                "camera motion": "horizontal",
                "background": "simple",
                "lighting": "bright",
            },
            "03": {
                "distance": "near",
                "camera motion": "horizontal",
                "background": "simple",
                "lighting": "dim",
            },
            "04": {
                "distance": "medium",
                "camera motion": "horizontal",
                "background": "busy",
                "lighting": "bright",
            },
            "05": {
                "distance": "far",
                "camera motion": "horizontal",
                "background": "busy",
                "lighting": "bright",
            },
            "06": {
                "distance": "medium",
                "camera motion": "vertical",
                "background": "busy",
                "lighting": "bright",
            },
            "07": {
                "distance": "medium",
                "camera motion": "diagonal",
                "background": "busy",
                "lighting": "bright",
            },
            "08": {
                "distance": "near",
                "camera motion": "horizontal",
                "background": "busy",
                "lighting": "dim",
            },
            "09": {
                "distance": "medium",
                "camera motion": "horizontal",
                "background": "busy",
                "lighting": "dim",
            },
            "10": {
                "distance": "far",
                "camera motion": "horizontal",
                "background": "busy",
                "lighting": "dim",
            },
        }

        self.background = ["all", "simple", "busy"]
        self.lighting = ["all", "bright", "dim"]
