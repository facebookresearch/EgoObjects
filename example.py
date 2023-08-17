#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import unittest
from copy import deepcopy

import numpy as np
from detectron2.utils.logger import create_small_table
from detectron2.data import MetadataCatalog
from egoobjects_api.eval import EgoObjectsEval
from egoobjects_api.results import EgoObjectsResults
from egoobjects_api.egoobjects import EgoObjects, FILTER_OPTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

gt_json_file = "./data/EgoObjectsV1_unified_eval.json"
metadata_json_file = "./data/EgoObjectsV1_unified_metadata.json"

metric_filter = {}

# NOTE on legends
# iou     -- the IOU threshold for computing metircs, "coco" refers to the averaging for IOU = [0.5, 0.55, ..., 0.95]
# gr      -- the grouping for categories, for category detection it could in ["all", "frequent", "common", "rare"]
#             for instance detection, it could be in ["all", "seen", "unseen"]
# ar      -- area ratio for the gt, it can be in ["all", "small", "medium", "large"]
# bg      -- background, choice in ["all", "busy", "simple"]
# lt      -- lighting condition, choice in ["all", "bright", "dim"]
# df      -- the difficulty for the test sample, only used for instance detection, as we already explicitly splited
#             the validation set into an easy one and a hard one. It's all filled with "all"
metric_filter["cat_det"] = [
    {"iou": "coco", "gr": "all", "ar": "all", "bg": "all", "lt": "all", "df": "all"},
    {"iou": "50", "gr": "all", "ar": "all", "bg": "all", "lt": "all", "df": "all"},
    {"iou": "75", "gr": "all", "ar": "all", "bg": "all", "lt": "all", "df": "all"},

    {"iou": "50", "gr": "frequent", "ar": "all", "bg": "all", "lt": "all", "df": "all"},
    {"iou": "50", "gr": "common", "ar": "all", "bg": "all", "lt": "all", "df": "all"},
    {"iou": "50", "gr": "rare", "ar": "all", "bg": "all", "lt": "all", "df": "all"},

    {"iou": "50", "gr": "all", "ar": "large", "bg": "all", "lt": "all", "df": "all"},
    {"iou": "50", "gr": "all", "ar": "medium", "bg": "all", "lt": "all", "df": "all"},
    {"iou": "50", "gr": "all", "ar": "small", "bg": "all", "lt": "all", "df": "all"},

    {"iou": "50", "gr": "all", "ar": "all", "bg": "all", "lt": "bright", "df": "all"},
    {"iou": "50", "gr": "all", "ar": "all", "bg": "all", "lt": "dim", "df": "all"},
    {"iou": "50", "gr": "all", "ar": "all", "bg": "simple", "lt": "all", "df": "all"},
    {"iou": "50", "gr": "all", "ar": "all", "bg": "busy", "lt": "all", "df": "all"},
]

metric_filter["inst_det"] = [
    {"iou": "coco", "gr": "all", "ar": "all", "bg": "all", "lt": "all", "df": "all"},
    {"iou": "50", "gr": "all", "ar": "all", "bg": "all", "lt": "all", "df": "all"},
    {"iou": "75", "gr": "all", "ar": "all", "bg": "all", "lt": "all", "df": "all"},

    {"iou": "50", "gr": "seen", "ar": "all", "bg": "all", "lt": "all", "df": "all"},
    {"iou": "50", "gr": "unseen", "ar": "all", "bg": "all", "lt": "all", "df": "all"},

    {"iou": "50", "gr": "all", "ar": "large", "bg": "all", "lt": "all", "df": "all"},
    {"iou": "50", "gr": "all", "ar": "medium", "bg": "all", "lt": "all", "df": "all"},
    {"iou": "50", "gr": "all", "ar": "small", "bg": "all", "lt": "all", "df": "all"},

    {"iou": "50", "gr": "all", "ar": "all", "bg": "all", "lt": "bright", "df": "all"},
    {"iou": "50", "gr": "all", "ar": "all", "bg": "all", "lt": "dim", "df": "all"},
    {"iou": "50", "gr": "all", "ar": "all", "bg": "simple", "lt": "all", "df": "all"},
    {"iou": "50", "gr": "all", "ar": "all", "bg": "busy", "lt": "all", "df": "all"},
]


def get_egoobjects_meta(metadata_path: str):
    """
    return metadata dictionary with 4 keys:
        cat_det_cats
        inst_det_cats
        cat_det_cat_id_2_cont_id
        cat_det_cat_names
    """
    with open(metadata_path, "r") as fp:
        metadata = json.load(fp)

    cat_det_cat_id_2_name = {cat["id"]: cat["name"] for cat in metadata["cat_det_cats"]}
    cat_det_cat_ids = sorted([cat["id"] for cat in metadata["cat_det_cats"]])
    cat_det_cat_id_2_cont_id = {cat_id: i for i, cat_id in enumerate(cat_det_cat_ids)}
    cat_det_cat_names = [cat_det_cat_id_2_name[cat_id] for cat_id in cat_det_cat_ids]

    metadata["cat_det_cat_id_2_cont_id"] = cat_det_cat_id_2_cont_id
    metadata["cat_det_cat_names"] = cat_det_cat_names
    return metadata

def main():
    dataset_name = "EgoObjects"
    metadata = get_egoobjects_meta(metadata_json_file)
    MetadataCatalog.get(dataset_name).set(**metadata)
    metadata = MetadataCatalog.get(dataset_name)

    split = "egoobjects_unified_det_val_query"
    gt = EgoObjects(gt_json_file, metadata, filter_opts=FILTER_OPTS[split])

    # dummy category detection predictions
    dt_cat = [
        deepcopy(ann)
        for ann in gt.dataset["annotations"]
        if "category_id" in ann and np.random.uniform() > 0.1
    ]

    logger.info(f"len dt_cat {len(dt_cat)}")

    for dt_box in dt_cat:
        cx, cy, w, h = dt_box["bbox"]
        w = np.random.randint(int(w * 0.5), w)
        h = np.random.randint(int(h * 0.5), h)
        image_id = dt_box["image_id"]
        category_id = dt_box["category_id"]

        dt_box["bbox"] = [cx, cy, w, h]
        dt_box["area"] = w * h
        dt_box["image_id"] = image_id
        dt_box["category_id"] = category_id
        dt_box["score"] = np.random.rand(1)[0]

    # dummy instance detection predictions
    dt_inst = [
        deepcopy(ann)
        for ann in gt.dataset["annotations"]
        if "instance_id" in ann and np.random.uniform() > 0.2
    ]

    logger.info(f"len dt_inst {len(dt_inst)}")

    for dt_box in dt_inst:
        cx, cy, w, h = dt_box["bbox"]
        w = np.random.randint(int(w * 0.5), w)
        h = np.random.randint(int(h * 0.5), h)
        image_id = dt_box["image_id"]
        instance_id = dt_box["instance_id"]

        dt_box["bbox"] = [cx, cy, w, h]
        dt_box["area"] = w * h
        dt_box["image_id"] = image_id
        dt_box["instance_id"] = instance_id
        dt_box["score"] = np.random.rand(1)[0]

    dt = EgoObjectsResults(gt, dt_cat, dt_inst)
    evaluator = EgoObjectsEval(gt, dt, num_processes=32)
    evaluator.run(metric_filter)
    evaluator.print_results()

    results = evaluator.get_results()
    for det_type in ["cat_det", "inst_det"]:
        logger.info(f"{det_type} results")
        one_result = results[det_type]
        one_result = {metric: float(one_result[metric]["value"] * 100) for metric in one_result.keys()}
        for _name, metric in one_result.items():
            assert metric < 100.0
        logger.info(create_small_table(one_result))

if __name__ == "__main__":
    main()