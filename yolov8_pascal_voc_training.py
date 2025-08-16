#!/usr/bin/env python3
"""
YOLOv8 Training Program for Pascal VOC Dataset
This program converts Pascal VOC format to YOLO format and trains a YOLOv8 model
"""

import os
import xml.etree.ElementTree as ET
import yaml
import shutil
from pathlib import Path
import argparse
from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


class VOCToYOLOConverter:
    """Convert Pascal VOC format to YOLO format"""

    def __init__(self, voc_path, output_path, class_names=None):
        self.voc_path = Path(voc_path)
        self.output_path = Path(output_path)
        self.class_names = class_names or self.get_class_names()
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

    def get_class_names(self):
        """Extract class names from XML annotations"""
        classes = set()
        annotations_path = self.voc_path / "Annotations"

        for xml_file in annotations_path.glob("*.xml"):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for obj in root.findall("object"):
                class_name = obj.find("name").text
                classes.add(class_name)

        return sorted(list(classes))

    def convert_bbox(self, size, bbox):
        """Convert Pascal VOC bbox to YOLO format"""
        dw = 1.0 / size[0]
        dh = 1.0 / size[1]
        x = (bbox[0] + bbox[2]) / 2.0
        y = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)

    def convert_annotation(self, xml_file, output_file):
        """Convert single XML annotation to YOLO format"""
        tree = ET.parse(xml_file)
        root = tree.getroot()

        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)

        annotations = []
        for obj in root.findall("object"):
            difficult = obj.find("difficult")
            if difficult is not None and int(difficult.text) == 1:
                continue

            cls = obj.find("name").text
            if cls not in self.class_to_idx:
                continue

            cls_id = self.class_to_idx[cls]
            xmlbox = obj.find("bndbox")
            bbox = (
                float(xmlbox.find("xmin").text),
                float(xmlbox.find("ymin").text),
                float(xmlbox.find("xmax").text),
                float(xmlbox.find("ymax").text),
            )

            yolo_bbox = self.convert_bbox((w, h), bbox)
            annotations.append(f"{cls_id} {' '.join(map(str, yolo_bbox))}")

        with open(output_file, "w") as f:
            f.write("\n".join(annotations))

    def convert_dataset(self, train_ratio=0.8, val_ratio=0.1):
        """Convert entire Pascal VOC dataset to YOLO format"""
        # Create output directory structure
        for split in ["train", "val", "test"]:
            (self.output_path / split / "images").mkdir(parents=True, exist_ok=True)
            (self.output_path / split / "labels").mkdir(parents=True, exist_ok=True)

        # Get all image files
        images_path = self.voc_path / "JPEGImages"
        annotations_path = self.voc_path / "Annotations"

        image_files = (
            list(images_path.glob("*.jpg"))
            + list(images_path.glob("*.jpeg"))
            + list(images_path.glob("*.png"))
        )

        # Split dataset
        train_files, temp_files = train_test_split(
            image_files, test_size=1 - train_ratio, random_state=42
        )
        val_files, test_files = train_test_split(
            temp_files,
            test_size=val_ratio / (val_ratio + (1 - train_ratio - val_ratio)),
            random_state=42,
        )

        splits = {"train": train_files, "val": val_files, "test": test_files}

        for split, files in splits.items():
            for img_file in files:
                # Copy image
                dst_img = self.output_path / split / "images" / img_file.name
                shutil.copy2(img_file, dst_img)

                # Convert annotation
                xml_file = annotations_path / f"{img_file.stem}.xml"
                if xml_file.exists():
                    label_file = (
                        self.output_path / split / "labels" / f"{img_file.stem}.txt"
                    )
                    self.convert_annotation(xml_file, label_file)

        # Create classes.txt
        with open(self.output_path / "classes.txt", "w") as f:
            f.write("\n".join(self.class_names))

        print(f"Dataset converted successfully!")
        print(f"Classes: {self.class_names}")
        print(f"Train: {len(splits['train'])} images")
        print(f"Val: {len(splits['val'])} images")
        print(f"Test: {len(splits['test'])} images")

        return self.class_names


class YOLOv8Trainer:
    """YOLOv8 training class"""

    def __init__(self, dataset_path, model_size="n"):
        self.dataset_path = Path(dataset_path)
        self.model_size = model_size
        self.model = None

    def create_config(self, class_names):
        """Create YAML configuration file for YOLOv8"""
        config = {
            "path": str(self.dataset_path.absolute()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(class_names),
            "names": class_names,
        }

        config_path = self.dataset_path / "dataset.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        return config_path

    def train(
        self,
        config_path,
        epochs=100,
        imgsz=640,
        batch_size=16,
        lr0=0.01,
        weight_decay=0.0005,
        save_period=10,
    ):
        """Train YOLOv8 model"""

        # Initialize model
        self.model = YOLO(f"yolov8{self.model_size}.pt")

        # Training parameters
        train_params = {
            "data": str(config_path),
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch_size,
            "lr0": lr0,
            "weight_decay": weight_decay,
            "save_period": save_period,
            "patience": 50,
            "device": "mps",  # Automatically select GPU if available (use auto) for pc platform
            "workers": 8,
            "project": "runs/train",
            "name": f"yolov8{self.model_size}_custom",
            "exist_ok": True,
            "pretrained": True,
            "optimizer": "auto",
            "close_mosaic": 10,
            "resume": False,
            "amp": True,  # Automatic Mixed Precision
            "fraction": 1.0,
            "profile": False,
            "freeze": None,
            "multi_scale": False,
            "overlap_mask": True,
            "mask_ratio": 4,
            "dropout": 0.0,
            "val": True,
            "split": "val",
            "save_json": False,
            "save_hybrid": False,
            "conf": None,
            "iou": 0.7,
            "max_det": 300,
            "half": False,
            "dnn": False,
            "plots": True,
            "source": None,
            "vid_stride": 1,
            "stream_buffer": False,
            "visualize": False,
            "augment": False,
            "agnostic_nms": False,
            "classes": None,
            "retina_masks": False,
            "embed": None,
            "show": False,
            "save_frames": False,
            "save_txt": False,
            "save_conf": False,
            "save_crop": False,
            "show_labels": True,
            "show_conf": True,
            "show_boxes": True,
            "line_width": None,
        }

        # Start training
        print("Starting YOLOv8 training...")
        results = self.model.train(**train_params)

        print("Training completed!")
        return results

    def validate(self):
        """Validate trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        results = self.model.val()
        return results

    def export_model(self, format="onnx"):
        """Export trained model to different formats"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        self.model.export(format=format)
        print(f"Model exported to {format} format")


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Pascal VOC dataset")
    parser.add_argument(
        "--voc_path", type=str, required=True, help="Path to Pascal VOC dataset"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for YOLO format dataset",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="YOLOv8 model size",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0005, help="Weight decay"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Training set ratio"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1, help="Validation set ratio"
    )
    parser.add_argument(
        "--convert_only",
        action="store_true",
        help="Only convert dataset, skip training",
    )
    parser.add_argument("--classes", nargs="+", help="Custom class names")

    args = parser.parse_args()

    # Convert Pascal VOC to YOLO format
    print("Converting Pascal VOC to YOLO format...")
    converter = VOCToYOLOConverter(args.voc_path, args.output_path, args.classes)
    class_names = converter.convert_dataset(args.train_ratio, args.val_ratio)

    if args.convert_only:
        print("Dataset conversion completed. Skipping training.")
        return

    # Train YOLOv8 model
    print("Initializing YOLOv8 trainer...")
    trainer = YOLOv8Trainer(args.output_path, args.model_size)

    # Create configuration
    config_path = trainer.create_config(class_names)

    # Start training
    results = trainer.train(
        config_path=config_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
    )

    # Validate model
    print("Validating trained model...")
    val_results = trainer.validate()

    # Export model
    print("Exporting model to ONNX format...")
    trainer.export_model("onnx")

    print("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()

# Example usage:
"""
# Basic usage
python yolov8_pascal_voc_training.py --voc_path /path/to/pascal_voc --output_path /path/to/yolo_dataset

# Advanced usage with custom parameters
python yolov8_pascal_voc_training.py \
    --voc_path /path/to/pascal_voc \
    --output_path /path/to/yolo_dataset \
    --model_size m \
    --epochs 200 \
    --batch_size 32 \
    --imgsz 640 \
    --lr0 0.01 \
    --weight_decay 0.0005 \
    --train_ratio 0.8 \
    --val_ratio 0.1

# Convert dataset only
python yolov8_pascal_voc_training.py \
    --voc_path /path/to/pascal_voc \
    --output_path /path/to/yolo_dataset \
    --convert_only

# Custom class names
python yolov8_pascal_voc_training.py \
    --voc_path /path/to/pascal_voc \
    --output_path /path/to/yolo_dataset \
    --classes person car bicycle
"""
