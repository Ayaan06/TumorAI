"""Training pipeline for 4-class brain tumor classification (Not for medical use)."""

from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

TRAIN_DIR = os.path.join("image_sets", "Training")
VAL_DIR = os.path.join("image_sets", "Testing")
MODEL_PATH = "tumor_model.pth"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Create data transforms for training and validation."""
    train_tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tfms, val_tfms


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, Dict[str, int], int, int]:
    """Load datasets from disk and wrap them in dataloaders."""
    train_tfms, val_tfms = build_transforms()
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_tfms)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_tfms)

    print("Detected classes:", train_dataset.classes)
    print("Class-to-index mapping:", train_dataset.class_to_idx)
    print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return (
        train_loader,
        val_loader,
        train_dataset.class_to_idx,
        len(train_dataset),
        len(val_dataset),
    )


def build_model(num_classes: int) -> nn.Module:
    """Load ResNet-18 with ImageNet weights and customize the classifier."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct / max(labels.size(0), 1)


def train(
    train_dir: str,
    val_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, val_loader, class_to_idx, train_count, val_count = create_dataloaders(
        train_dir=train_dir, val_dir=val_dir, batch_size=batch_size
    )

    class_names = list(class_to_idx.keys())
    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        train_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size_actual = labels.size(0)
            epoch_train_loss += loss.item() * batch_size_actual
            epoch_train_acc += accuracy_from_logits(outputs, labels) * batch_size_actual
            train_samples += batch_size_actual

        train_loss = epoch_train_loss / max(train_samples, 1)
        train_acc = epoch_train_acc / max(train_samples, 1)

        model.eval()
        val_loss_total = 0.0
        val_acc_total = 0.0
        val_samples = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                batch_size_actual = labels.size(0)
                val_loss_total += loss.item() * batch_size_actual
                val_acc_total += accuracy_from_logits(outputs, labels) * batch_size_actual
                val_samples += batch_size_actual

        val_loss = val_loss_total / max(val_samples, 1)
        val_acc = val_acc_total / max(val_samples, 1)

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {"model_state": model.state_dict(), "class_names": class_names}
            torch.save(best_state, MODEL_PATH)
            print(
                f"  -> Saved best model to {MODEL_PATH} "
                f"(Val Acc: {best_val_acc:.4f}, Train samples: {train_count}, Val samples: {val_count})"
            )

    if best_state is None:
        raise RuntimeError("Training completed without saving a checkpoint. Check data setup.")

    print(f"Training finished. Best validation accuracy: {best_val_acc:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 4-class brain tumor classifier (Not for medical use)."
    )
    parser.add_argument("--train-dir", default=TRAIN_DIR, help="Path to training folder.")
    parser.add_argument("--val-dir", default=VAL_DIR, help="Path to validation folder.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.train_dir) or not os.path.isdir(args.val_dir):
        raise FileNotFoundError(
            "Training or validation directory missing. "
            "Ensure image_sets/Training and image_sets/Testing exist."
        )
    train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
