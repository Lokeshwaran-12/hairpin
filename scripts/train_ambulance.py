"""
Ambulance YOLOv8 Fine-Tuning Script:
Trains YOLOv8n on the converted ambulance_yolo dataset using transfer learning.
Saves the best fine-tuned model checkpoint to models/yolov8n_ambulance.pt.
Usage:
    python scripts/train_ambulance.py --epochs 15 --imgsz 416 --batch 8
"""
import os
import sys
import shutil
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO


def train_ambulance_model(
    data_yaml: str = "data/datasets/ambulance_yolo/data.yaml",
    base_weights: str = "yolov8n.pt",
    epochs: int = 15,
    imgsz: int = 416,
    batch: int = 8,
    device: str = "auto",
    output_model: str = "models/yolov8n_ambulance.pt"
):
    project_root = Path(__file__).resolve().parent.parent
    data_yaml_path = project_root / data_yaml
    output_path = project_root / output_model

    if not data_yaml_path.exists():
        print(f"Error: {data_yaml_path} does not exist.")
        print("Please run 'python scripts/prepare_ambulance_dataset.py' first.")
        sys.exit(1)

    # Determine device
    if device == "auto":
        target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        target_device = device

    print("=" * 70)
    print("      YOLOV8 AMBULANCE & EMERGENCY VEHICLE FINE-TUNING")
    print("=" * 70)
    print(f" Dataset YAML:      {data_yaml_path}")
    print(f" Base Weights:      {base_weights}")
    print(f" Target Device:     {target_device} (CUDA Available: {torch.cuda.is_available()})")
    print(f" Image Size:        {imgsz}x{imgsz}")
    print(f" Batch Size:        {batch}")
    print(f" Epochs:            {epochs}")
    print(f" Final Output:      {output_path}")
    print("=" * 70)

    # Create models directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load base model for transfer learning
    model = YOLO(base_weights)

    # Execute training
    runs_dir = project_root / "runs" / "train" / "ambulance_finetune"
    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=target_device,
        project=str(project_root / "runs" / "train"),
        name="ambulance_finetune",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        patience=5,
        save=True,
        verbose=True
    )

    # Find best.pt from run
    best_pt = project_root / "runs" / "train" / "ambulance_finetune" / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, output_path)
        print(f"\n[Success] Best fine-tuned checkpoint copied to: {output_path}")
    else:
        last_pt = project_root / "runs" / "train" / "ambulance_finetune" / "weights" / "last.pt"
        if last_pt.exists():
            shutil.copy2(last_pt, output_path)
            print(f"\n[Notice] Copied last checkpoint to: {output_path}")

    # Run quick validation
    print("\nRunning Validation Evaluation...")
    val_model = YOLO(str(output_path))
    val_results = val_model.val(data=str(data_yaml_path), imgsz=imgsz, device=target_device)

    print("\n" + "=" * 70)
    print("      FINE-TUNING COMPLETED")
    print("=" * 70)
    print(f" Model saved at: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Ambulance Dataset")
    parser.add_argument("--data", type=str, default="data/datasets/ambulance_yolo/data.yaml", help="Path to data.yaml")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Base pretrained weights")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=416, help="Input image dimension")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cpu', 'cuda:0', 'auto')")
    parser.add_argument("--output", type=str, default="models/yolov8n_ambulance.pt", help="Output path for best model")

    args = parser.parse_args()
    train_ambulance_model(
        data_yaml=args.data,
        base_weights=args.weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        output_model=args.output
    )


if __name__ == "__main__":
    main()
