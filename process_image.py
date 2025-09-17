# process_image.py
import argparse
from inference.detector import load_models
from inference.commenter import generate_comments
from utils.viz import draw_boxes

def to_simple(dets):
    # convert to unified format
    out = []
    for d in dets:
        out.append({
            "bbox": d["bbox"],
            "label": d["label"],
            "conf": d["conf"]
        })
    return out

def main(image_path, out_path="static/results/annotated.jpg"):
    cfg, fire_model, textile_model = load_models()
    fire_dets = fire_model.predict(image_path)
    textile_dets = textile_model.predict(image_path)
    # filter only fire extinguisher detections labeled as configured name
    fire_label_name = cfg["models"]["fire"]["class_name"]
    fires = [d for d in fire_dets if d["label"] == fire_label_name or fire_label_name in d["label"].lower()]
    # run comments
    comments = generate_comments(fires, textile_dets, cfg_path="config.yaml")
    # prepare combined detections for viz
    combined = fires + textile_dets
    draw_boxes(image_path, combined, out_path)
    print("Annotated image saved to:", out_path)
    print("Comments:")
    for c in comments:
        print("-", c)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--out", default="static/results/annotated.jpg", help="Output annotated image path")
    args = parser.parse_args()
    main(args.image, args.out)
