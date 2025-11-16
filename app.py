"""Gradio demo for 4-class brain tumor type detection (Not for medical use)."""

from __future__ import annotations

import os
from typing import Dict, Tuple

import gradio as gr
import torch
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = "tumor_model.pth"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
 

def load_model(path: str) -> Tuple[torch.nn.Module, list[str], torch.device]:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Checkpoint {path} not found. Run train_tumor_model.py before launching the app."
        )

    checkpoint = torch.load(path, map_location="cpu")
    class_names = checkpoint["class_names"]

    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, class_names, device


TRANSFORM = build_transforms()
MODEL, CLASS_NAMES, DEVICE = load_model(MODEL_PATH)


def predict(image: Image.Image) -> Tuple[Dict[str, float], str]:
    if image is None:
        raise ValueError("Please upload an MRI image.")

    image = image.convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu()

    prob_dict = {cls: float(probs[idx]) for idx, cls in enumerate(CLASS_NAMES)}
    sorted_items = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    summary_lines = [f"**{cls}**: {prob*100:.2f}%" for cls, prob in sorted_items]
    summary = "\n".join(summary_lines)
    return prob_dict, summary


THEME = (
    gr.themes.Soft(primary_hue="orange", secondary_hue="slate", neutral_hue="gray")
    .set(body_background_fill="#0c1524", body_text_color="#f7f7f7")
)

CUSTOM_CSS = """
#hero {
  background: linear-gradient(90deg, #171f33, #0f1626);
  border-radius: 12px;
  padding: 1.2rem 1.6rem;
}
#hero h1 {
  margin-bottom: 0.3rem;
}
#tip-card {
  border: 1px solid #27324a;
  background: #111a2c;
  border-radius: 8px;
  padding: 0.6rem;
  font-size: 0.9rem;
}
.panel {
  background: #111b2f;
  border-radius: 12px;
  padding: 1rem;
  border: 1px solid #1f2a3f;
}
#action-btn {
  font-size: 1.1rem;
  height: 3rem;
}
"""

with gr.Blocks(
    theme=THEME,
    css=CUSTOM_CSS,
    title="Brain Tumor Type Detection (Not for Medical Use)",
) as demo:
    gr.HTML(
        """
        <div id="hero">
          <h1>Brain Tumor Type Detection (Not for Medical Use)</h1>
          <p>This demo estimates probabilities for glioma, meningioma, pituitary, or no tumor
          classes from MRI slices. A lightweight REST API endpoint is coming soon for remote
          integrations. Until then, follow the README to run everything yourself.</p>
        </div>
        """
    )
    with gr.Row():
        with gr.Column(scale=1, elem_classes=["panel"]):
            image_input = gr.Image(
                type="pil",
                label="Upload MRI Image",
                height=360,
            )
            gr.HTML(
                '<div id="tip-card">Tip: use images from '
                '`image_sets/Testing/<class>` to validate the model.</div>'
            )
        with gr.Column(scale=1, elem_classes=["panel"]):
            prob_output = gr.Label(label="Class probabilities")
            summary_output = gr.Markdown(value="Confidence details will appear here.")
    analyze_btn = gr.Button("Analyze MRI", variant="primary", elem_id="action-btn")
    analyze_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[prob_output, summary_output],
    )
    with gr.Accordion("How this works", open=False):
        gr.Markdown(
            "- Images are resized to 224x224 and normalized with ImageNet statistics.\n"
            "- A ResNet-18 backbone fine-tuned on four MRI tumor classes produces logits.\n"
            "- Outputs are softmax probabilities shown above.\n"
            "- Everything runs on your local device; never rely on this for medical care."
        )

if __name__ == "__main__":
    demo.queue().launch()
