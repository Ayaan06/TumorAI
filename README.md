# TumorAI - Brain Tumor Type Detection Demo (Not for Medical Use)

This repository walks through an end-to-end workflow for fine-tuning a ResNet-18 classifier on the Kaggle brain tumor MRI dataset and serving predictions with Gradio. The pipeline is tested on Windows with an NVIDIA RTX 3060 using Python and PyTorch. **All predictions are strictly for educational/demo purposes and must never be used to make medical decisions.** A lightweight REST API is planned so others can interact with the model remotely—until then, follow the instructions below to run everything yourself.

---

## 1. Dataset Overview

- Source: <https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri>
- Classes: `glioma_tumor`, `meningioma_tumor`, `pituitary_tumor`, `no_tumor`

Expected folder structure after extracting the dataset:

```
TumorAI/
  image_sets/
    Training/
      glioma_tumor/
      meningioma_tumor/
      pituitary_tumor/
      no_tumor/
    Testing/
      glioma_tumor/
      meningioma_tumor/
      pituitary_tumor/
      no_tumor/
```

The training script relies on `torchvision.datasets.ImageFolder`, so keep the directory layout exactly as shown or override it with the CLI flags `--train-dir` and `--val-dir`.

---

## 2. Technology Stack

| Component           | Details                                                                 |
|--------------------|-------------------------------------------------------------------------|
| Language            | Python 3                                                                |
| Deep Learning       | PyTorch, Torchvision                                                    |
| Data Handling       | Pillow, optional OpenCV                                                 |
| UI / Serving        | Gradio dashboard                                                        |
| Model Architecture  | ResNet-18 pretrained on ImageNet, fine-tuned for four classes           |
| Hardware Support    | CUDA GPU (RTX 3060) preferred; CPU fallback works albeit slower         |
| Environment         | Python virtual environment `.venv` in project root                      |

---

## 3. Environment Setup

Create and activate a project-specific virtual environment, then install dependencies (Torch build pinned to CUDA 12.1 for the RTX 3060):

```powershell
cd C:\projects\Tumor\TumorAI
python -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pillow opencv-python gradio
```

> Tip: Save these commands to a PowerShell script or VS Code task to recreate the environment quickly.

---

## 4. Training Pipeline (`train_tumor_model.py`)

### 4.1 Data Loading

- Training data: `image_sets/Training`
- Validation data: `image_sets/Testing`
- `ImageFolder` infers class names from subdirectories and prints the mapping along with dataset sizes.
- DataLoaders use batch size 32, shuffle only for training, and pin memory whenever CUDA is available.

### 4.2 Transforms

- Training: resize 224x224 ? random horizontal flip ? random ±10° rotation ? tensor conversion ? normalize with ImageNet mean/std.
- Validation: resize 224x224 ? tensor ? normalize with ImageNet mean/std.

### 4.3 Model Definition

- Loads ResNet-18 with ImageNet weights (`models.ResNet18_Weights.IMAGENET1K_V1`).
- Replaces the final fully connected layer to output four logits.
- Moves the model to `cuda` if available, otherwise `cpu`.

### 4.4 Training Loop

- Loss: `torch.nn.CrossEntropyLoss`
- Optimizer: `torch.optim.Adam` with default learning rate `1e-4`
- Epochs: default 10 (configurable via CLI flag)
- Prints per-epoch train/val loss and accuracy.
- Saves `tumor_model.pth` whenever validation accuracy improves; the checkpoint stores `model_state` and `class_names`.

### 4.5 Running Training

```powershell
python train_tumor_model.py --epochs 10 --batch-size 32 --lr 1e-4
```

Optional flags:

- `--train-dir` / `--val-dir` to point to alternative folders
- `--epochs`, `--batch-size`, `--lr` to tweak hyperparameters

Training typically takes a few minutes on an RTX 3060 for 10 epochs. CPU execution works but is slower.

---

## 5. Gradio Inference App (`app.py`)

### 5.1 Loading the Model

- Reads `tumor_model.pth`.
- Reconstructs ResNet-18 with a four-class classifier head.
- Uses the same preprocessing transforms as validation.

### 5.2 Prediction Function

- Accepts an uploaded MRI (RGB).
- Converts to tensor, runs inference, applies softmax to obtain probabilities.
- Returns a probability distribution (`gr.Label`) and a Markdown summary.

### 5.3 Launching the UI

```powershell
python app.py
```

Open the URL printed in the console (default `http://127.0.0.1:7860`). Upload an MRI image to inspect probabilities. The UI reiterates the **Not for Medical Use** warning and highlights that a REST API is coming soon.

---

## 6. Troubleshooting & Tips

- **Checkpoint missing:** Run `python train_tumor_model.py` before `python app.py`.
- **Dataset layout errors:** Verify the `image_sets/Training` and `image_sets/Testing` folders exist or override with CLI flags.
- **Performance tweaks:** Experiment with stronger augmentations, longer training, schedulers, or TensorBoard logging.
- **Determinism:** Set seeds for Python/NumPy/PyTorch and enable `torch.backends.cudnn.deterministic = True` if you need reproducible runs.
- **Resource usage:** Batch size 32 fits on an RTX 3060; lower it if you encounter CUDA OOM errors.

---

## 7. Safety Disclaimer

This project is an educational demonstration only. The model is not validated for clinical use, and predictions must **never** influence medical decisions. Always consult qualified medical professionals for diagnosis or treatment.

---

## 8. Quick Command Reference

```powershell
# 1. Setup
python -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pillow opencv-python gradio

# 2. Train
python train_tumor_model.py

# 3. Launch UI
python app.py
```

Follow these steps from the project root (`C:\projects\Tumor\TumorAI`) to reproduce the full workflow end-to-end.

---

## 9. API Roadmap

- REST API endpoints will expose the prediction logic for remote clients.
- Planned features: token-based authentication, request logging, optional batching, and example integration notebooks.
- Until the API ships, distribute the project files or self-host the Gradio app if someone else needs to try the model.
