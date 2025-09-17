# evaluation

Local VS Code project to run two YOLO models (fire extinguisher + textile/object detector) on uploaded images and generate comments such as "fire extinguisher is not accessible".

## Setup

1. Clone / copy this folder to your machine.
2. Create a Python virtual environment and install deps:
   ```bash
   python -m venv venv
   source venv/bin/activate        # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
