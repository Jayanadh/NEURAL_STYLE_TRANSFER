# 🎨 Neural Style Transfer & Segmentation Experiments

This repository contains three Jupyter notebooks exploring **neural style transfer (NST)** combined with modern **segmentation** techniques:

1. **deep50_vgg199_back.ipynb** — Classic VGG‑19 style transfer pipeline (content/style loss, feature maps, multi‑layer style aggregation).  
2. **Mask -rcnn and vgg19.ipynb** — Foreground‑aware NST using **Mask R‑CNN** to isolate subjects, then applying style transfer selectively.  
3. **Sam and vgg19.ipynb** — Prompt‑free object masks via **Segment Anything (SAM)**, then applying VGG‑19 style to masked regions for clean composites.

The goal is to compare subject‑aware stylization vs global stylization and produce high‑quality artworks while preserving critical structure.

---

## 📂 Project Structure

```
.
├─ deep50_vgg199_back.ipynb
├─ Mask -rcnn and vgg19.ipynb
├─ Sam and vgg19.ipynb
├─ requirements.txt
├─ README.md
└─ outputs/                 # put your generated images here (added to git)
   ├─ nst_vgg19_sample.jpg
   ├─ nst_maskrcnn_sample.jpg
   └─ nst_sam_sample.jpg
```

> Create the `outputs/` folder and drop your result images there. They will render on GitHub automatically.

---

## ⚙️ Environment & Installation

> Python 3.9+ recommended. GPU (CUDA) will speed things up but is optional for inference‑only runs.

### 1) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> If `segment-anything` fails to import, install from source:
```bash
pip uninstall -y segment-anything
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 3) (Optional) Colab
The notebooks contain `google.colab` helpers for file upload. You can also run locally in Jupyter:
```bash
pip install notebook
jupyter notebook
```

---

## ▶️ How to Run

### A) VGG‑19 Global Style Transfer (`deep50_vgg199_back.ipynb`)
1. Open the notebook and set **content** and **style** image paths.  
2. Run cells to compute VGG‑19 features, style/content losses, and optimize the output image.  
3. Save results into `outputs/nst_vgg19_<desc>.jpg`.

### B) Mask‑aware NST via **Mask R‑CNN** (`Mask -rcnn and vgg19.ipynb`)
1. Provide the content image (person/object).  
2. The notebook runs `maskrcnn_resnet50_fpn` to get foreground masks.  
3. Apply VGG‑19 style transfer **only on masked regions** (or different strengths per mask).  
4. Save to `outputs/nst_maskrcnn_<desc>.jpg`.

### C) SAM‑guided NST (`Sam and vgg19.ipynb`)
1. Load the content image; the notebook uses SAM for high‑quality object masks.  
2. Compose per‑mask style transfer or blend multiple styles.  
3. Save to `outputs/nst_sam_<desc>.jpg`.

---

## 🖼️ Output Gallery (auto‑renders on GitHub)

> Replace file names after you add your images to the `outputs/` directory.

| Global VGG‑19 | Mask R‑CNN + VGG‑19 | SAM + VGG‑19 |
| --- | --- | --- |
| ![VGG19](outputs/nst_vgg19_sample.jpg) | ![MaskRCNN](outputs/nst_maskrcnn_sample.jpg) | ![SAM](outputs/nst_sam_sample.jpg) |

You can add more images—GitHub will display them as long as paths are correct.

---

## 🧠 Notes & Tips

- **Torch / TorchVision versions** matter for detection models (Mask R‑CNN) and may require compatible CUDA.  
- If you see missing ops errors for `torchvision` (e.g., `nms`), reinstall a **matching** torch/torchvision pair.  
- **Mediapipe** is available for face/hands/pose; if unused, you can remove it from requirements to slim the env.  

---

## 📜 License
Add a license (MIT or similar) if you plan to share/extend. If you already have a `LICENSE`, it will apply.

---

## 🙌 Acknowledgements
- PyTorch & TorchVision model zoo (VGG‑19, Mask R‑CNN, DeepLabV3)
- Meta AI **Segment Anything** (SAM)
- OpenCV / NumPy / Matplotlib for preprocessing & visualization
