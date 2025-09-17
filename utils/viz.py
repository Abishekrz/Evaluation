# utils/viz.py
import cv2
import os

def draw_boxes(image_path, detections, out_path):
    """
    detections: list of dicts with keys 'bbox', 'label', 'conf'
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image {image_path} not found or unreadable.")
    for d in detections:
        x1,y1,x2,y2 = map(int, d["bbox"])
        label = f"{d.get('label','')}: {d.get('conf',0):.2f}"
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        (w,h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - 18), (x1 + w + 6, y1), (0,255,0), -1)
        cv2.putText(img, label, (x1+3, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)
    return out_path
