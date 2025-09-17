# inference/detector.py
from ultralytics import YOLO
import yaml
import os
import numpy as np

class ModelWrapper:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, image_path, conf=0.25, iou=0.45):
        results = self.model.predict(source=image_path, conf=conf, iou=iou, verbose=False)
        # results is a list; take first result for single image
        r = results[0]
        detections = []
        # r.boxes.xyxyn or r.boxes.xyxy, r.boxes.conf, r.boxes.cls
        boxes = r.boxes
        if boxes is None:
            return detections
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        names = self.model.model.names if hasattr(self.model, "model") else {}
        for b, c, cl in zip(xyxy, confs, cls):
            x1, y1, x2, y2 = b.tolist()
            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(c),
                "class_id": int(cl),
                "label": names.get(cl, str(cl))
            })
        return detections

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_models(config_path="config.yaml"):
    cfg = load_config(config_path)
    fire_path = cfg["models"]["fire"]["path"]
    textile_path = cfg["models"]["textile"]["path"]
    if not os.path.exists(fire_path):
        raise FileNotFoundError(f"Fire model not found at {fire_path}. Place your weights there.")
    if not os.path.exists(textile_path):
        raise FileNotFoundError(f"Textile model not found at {textile_path}. Place your weights there.")
    fire_model = ModelWrapper(fire_path)
    textile_model = ModelWrapper(textile_path)
    return cfg, fire_model, textile_model
