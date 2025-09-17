# Flask app entry point
# app.py
import os
from flask import Flask, request, render_template, url_for
from inference.detector import load_models
from inference.commenter import generate_comments
from utils.viz import draw_boxes
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "static/results"
ALLOWED = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

cfg, fire_model, textile_model = load_models()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return "No file part", 400
        file = request.files["image"]
        if file.filename == "":
            return "No selected file", 400
        if file and allowed_file(file.filename):
            fname = secure_filename(file.filename)
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
            file.save(save_path)

            fire_dets = fire_model.predict(save_path)
            textile_dets = textile_model.predict(save_path)
            fire_label_name = cfg["models"]["fire"]["class_name"]
            fires = [
                d for d in fire_dets
                if d["label"] == fire_label_name or fire_label_name in d["label"].lower()
            ]

            comments = generate_comments(fires, textile_dets)

            combined = fires + textile_dets
            os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)
            out_file = os.path.join(app.config["RESULT_FOLDER"], f"annotated_{fname}")
            draw_boxes(save_path, combined, out_file)

            # ✅ Normalize relative path for Flask
            rel_out = os.path.relpath(out_file, "static").replace("\\", "/")

            return render_template(
                "result.html",
                image_url=url_for("static", filename=rel_out),
                comments=comments
            )
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
