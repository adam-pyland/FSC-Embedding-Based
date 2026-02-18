import os
import glob
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
import numpy as np
from tqdm import tqdm


def compute_polygon_iou(poly1_pts, poly2_pts):
    """Compute IoU between two polygons (list of (x,y) tuples)."""
    poly1 = Polygon(poly1_pts)
    poly2 = Polygon(poly2_pts)
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    if union == 0:
        return 0.0
    return inter / union

def evaluate_yolo_obb_polygon(pred_dir, gt_dir, iou_thresh=0.5):
    tp = 0
    fp = 0
    fn = 0
    all_scores = []
    all_matches = []

    pred_files = glob.glob(os.path.join(pred_dir, "*.txt"))

    for pred_path in tqdm(pred_files):
        base = os.path.basename(pred_path).replace(".txt", "")
        gt_path = os.path.join(gt_dir, base + ".xml")
        if not os.path.exists(gt_path):
            continue

        # --- Parse GT polygons ---
        tree = ET.parse(gt_path)
        root = tree.getroot()
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)

        gt_polygons = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            if "vehicle" not in name.lower():  # filter only vehicle class
                continue

            polygon = obj.find("polygon")
            pts = []
            for i in range(4):
                x = float(polygon.find(f"x{i}").text)
                y = float(polygon.find(f"y{i}").text)
                pts.append((x, y))
            gt_polygons.append(pts)

        matched = [False] * len(gt_polygons)

        # --- Parse predictions ---
        with open(pred_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            cls = int(parts[0])
            if cls != 0:
                continue  # only vehicle class

            coords = list(map(float, parts[1:9]))  # 4 points
            conf = float(parts[9])

            # Convert normalized to pixel coordinates
            pred_pts = []
            for i in range(0, 8, 2):
                x = coords[i] * width
                y = coords[i+1] * height
                pred_pts.append((x, y))

            best_iou = 0
            best_idx = -1
            for i, gt_poly in enumerate(gt_polygons):
                iou = compute_polygon_iou(pred_pts, gt_poly)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou >= iou_thresh and not matched[best_idx]:
                tp += 1
                matched[best_idx] = True
                all_matches.append(1)
            else:
                fp += 1
                all_matches.append(0)

            all_scores.append(conf)

        fn += matched.count(False)

    # --- Precision, Recall ---
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    # --- Average Precision (AP) ---
    sorted_indices = np.argsort(-np.array(all_scores))
    sorted_matches = np.array(all_matches)[sorted_indices]
    cum_tp = np.cumsum(sorted_matches)
    cum_fp = np.cumsum(1 - sorted_matches)

    precisions = cum_tp / (cum_tp + cum_fp + 1e-8)
    recalls = cum_tp / (tp + fn + 1e-8)
    ap = np.trapezoid(precisions, recalls)  # numeric integration

    return precision, recall, ap


pred_dir = "/home/adamm/Documents/FSOD/codes/FSC-Embedding-Based-Satellites/FSC-Embedding-Based-Satellites/Results_with_SAHI/FAIR1M_17_2_26/labels"
gt_dir   = "/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Filtered_Dataset/VOC_Annotations/Base_and_Big_Cargo_Vehicles_Hor_Boxes/Val/"
iou_thresh=0.25
precision, recall, ap = evaluate_yolo_obb_polygon(pred_dir, gt_dir, iou_thresh=iou_thresh)

print(f"Vehicle class Precision: {precision:.4f}")
print(f"Vehicle class Recall: {recall:.4f}")
print(f"Vehicle class AP@{iou_thresh*100}: {ap:.4f}")