# inference/commenter.py
from typing import List, Dict
import yaml
import os

def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def intersection_area(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def generate_comments(fire_dets: List[Dict], other_dets: List[Dict],
                      cfg_path="config.yaml", comments_path="comments.yaml"):
    """
    fire_dets: detections labeled as fire_extinguisher
    other_dets: detections from textile model
    returns: list of comments (strings)
    """
    cfg = load_yaml(cfg_path)
    comments_cfg = load_yaml(comments_path)

    o_thresh = float(cfg["heuristics"].get("overlap_threshold", 0.10))
    results = []

    if not fire_dets:
        # no extinguisher detected, just return other object comments
        for o in other_dets:
            results.append(comments_cfg.get(o["label"], comments_cfg.get("default", "")))
        return list(dict.fromkeys(results))  # deduplicate

    for f in fire_dets:
        fbox = f["bbox"]
        f_area = box_area(fbox)
        obstructed = False

        for o in other_dets:
            obox = o["bbox"]
            inter = intersection_area(fbox, obox)
            if f_area > 0 and inter / f_area >= o_thresh:
                obstructed = True
                results.append(comments_cfg.get(o["label"], comments_cfg.get("default", "")))

        if obstructed:
            results.append(comments_cfg["fire_extinguisher"]["obstructed"])
        else:
            results.append(comments_cfg["fire_extinguisher"]["accessible"])

    # deduplicate while preserving order
    return list(dict.fromkeys(results))
