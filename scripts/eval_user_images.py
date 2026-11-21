"""
Comprehensive Diagnostic Evaluator for User-Uploaded Test Images:
Runs YOLODetector and raw model heads on all 5 images, generates annotated outputs,
and logs detailed breakdown of successes, misses, and false alarms.
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from src.detection.yolo_detector import YOLODetector, Detection


def evaluate_user_images():
    base_dir = Path(r"C:/Users/shahiil/.gemini/antigravity-cli/brain/f00582fe-b629-403f-9102-43fcc86e0af8/.user_uploaded")
    out_dir = Path("runs/user_eval")
    if not out_dir.exists():
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    image_files = [f"uploaded_media_{i}_1795266614869.jpg" for i in range(5)]

    detector = YOLODetector(
        weights_path="yolov8n.pt",
        ambulance_weights="models/yolov8n_ambulance.pt",
        enable_fine_tuned_ambulance=True,
        confidence_threshold=0.20,
        ambulance_conf=0.50
    )

    print("=" * 75)
    print("      COMPREHENSIVE DIAGNOSTIC EVALUATION OF 5 USER TEST IMAGES")
    print("=" * 75)

    results_summary = []

    for idx, fname in enumerate(image_files):
        img_path = base_dir / fname
        if not img_path.exists():
            print(f"Skipping {fname} (not found)")
            continue

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # 1. Base YOLO predictions
        base_res = detector.model.predict(source=img, conf=0.20, verbose=False)[0]
        base_counts = {}
        for b in base_res.boxes:
            cid = int(b.cls[0].item())
            cname = detector.COCO_VEHICLE_CLASSES.get(cid, str(cid))
            base_counts[cname] = base_counts.get(cname, 0) + 1

        # 2. Raw Fine-tuned Ambulance model predictions
        raw_amb_boxes = []
        if detector.ambulance_model_loaded:
            amb_res = detector.ambulance_model.predict(source=img, conf=0.15, verbose=False)[0]
            for ab in amb_res.boxes:
                c = float(ab.conf[0].item())
                b = ab.xyxy[0].cpu().numpy().astype(int).tolist()
                raw_amb_boxes.append((b, c))

        # 3. Full hybrid detector
        detections = detector.detect(img)
        emerg_dets = [d for d in detections if d.is_emergency]
        cars = [d for d in detections if d.class_name == "car"]
        trucks = [d for d in detections if d.class_name == "truck"]
        buses = [d for d in detections if d.class_name == "bus"]

        # Render bounding boxes
        annotated = YOLODetector.draw_bounding_boxes(img, detections)
        out_file = out_dir / f"eval_image_{idx}.jpg"
        cv2.imwrite(str(out_file), annotated)

        print(f"\n[IMAGE #{idx}] {fname} ({w}x{h})")
        print(f"  Base COCO vehicles:    Cars={len(cars)}, Trucks={len(trucks)}, Buses={len(buses)}")
        print(f"  Raw Fine-tuned Heads:  {len(raw_amb_boxes)} detections (conf >= 0.15)")
        for b, c in raw_amb_boxes:
            print(f"     -> Box={b}, Confidence={c*100:.1f}%")
        print(f"  Final Output:          {len(detections)} total vehicles, {len(emerg_dets)} AMBULANCES")
        for ed in emerg_dets:
            print(f"     * AMBULANCE: Box={ed.box}, Conf={ed.confidence*100:.1f}%, Type={ed.emergency_type}")
        print(f"  Saved Annotated View:  {out_file}")

        results_summary.append({
            "image_idx": idx,
            "filename": fname,
            "total_vehicles": len(detections),
            "ambulances_detected": len(emerg_dets),
            "ambulance_boxes": [ed.box for ed in emerg_dets],
            "raw_amb_count": len(raw_amb_boxes)
        })

    print("\n" + "=" * 75)
    print("                  EVALUATION RUN COMPLETE")
    print("=" * 75)
    return results_summary


if __name__ == "__main__":
    evaluate_user_images()
