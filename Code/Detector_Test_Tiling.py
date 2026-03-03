import os
import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO
from tqdm import tqdm

try:
    from shapely.geometry import Polygon
except ImportError:
    print("Warning: 'shapely' library is not installed. Evaluation will fail. Run 'pip install shapely'.")

def compute_polygon_iou(poly1_pts, poly2_pts):
    """
    Computes Intersection over Union (IoU) for two polygons using Shapely.
    poly_pts should be a list or array of points [[x1, y1], [x2, y2], ...]
    """
    p1 = Polygon(poly1_pts)
    p2 = Polygon(poly2_pts)
    
    # Fix invalid geometry if self-intersecting
    if not p1.is_valid: p1 = p1.buffer(0)
    if not p2.is_valid: p2 = p2.buffer(0)
        
    if not p1.intersects(p2):
        return 0.0
    try:
        inter_area = p1.intersection(p2).area
        union_area = p1.union(p2).area
        return inter_area / union_area if union_area > 0 else 0.0
    except Exception:
        return 0.0

def evaluate_predictions(all_preds_dict, label_dir, iou_thresh=0.25):
    """
    Evaluates stitched predictions against ground truth labels.
    Calculates TP, FP, FN, Precision, Recall, and F1-Score.
    """
    print(f"\nStarting Evaluation (IoU Threshold: {iou_thresh})...")
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for img_file, img_data in tqdm(all_preds_dict.items(), desc="Evaluation"):
        preds = img_data['preds']
        img_w = img_data['width']
        img_h = img_data['height']
        
        # Load Ground Truths
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)
        
        gt_polys = []
        gt_classes =[]
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 9:
                        cls = int(parts[0])
                        # The dataset has normalized coordinates[x1 y1 x2 y2 x3 y3 x4 y4]
                        coords = list(map(float, parts[1:9]))
                        pts = np.array(coords).reshape(-1, 2)
                        
                        # Un-normalize back to absolute pixels
                        pts[:, 0] *= img_w
                        pts[:, 1] *= img_h
                        
                        gt_polys.append(pts)
                        gt_classes.append(cls)
        
        # Sort predictions by confidence score descending
        preds = sorted(preds, key=lambda x: x['score'], reverse=True)
        matched_gt_indices = set()
        
        for pred in preds:
            pred_poly = pred['poly']
            pred_cls = pred['class']
            
            best_iou = 0.0
            best_gt_idx = -1
            
            # Find the best matching ground truth box
            for gt_idx, (gt_poly, gt_cls) in enumerate(zip(gt_polys, gt_classes)):
                if gt_idx in matched_gt_indices:
                    continue  # Already matched
                if pred_cls != gt_cls:
                    continue  # Class mismatch
                
                iou = compute_polygon_iou(pred_poly, gt_poly)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    
            if best_iou >= iou_thresh:
                total_tp += 1
                matched_gt_indices.add(best_gt_idx)
            else:
                total_fp += 1
                
        # Any ground truth not matched is a False Negative
        total_fn += len(gt_polys) - len(matched_gt_indices)
        
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print("\n" + "="*50)
    print("                 EVALUATION METRICS                 ")
    print("="*50)
    print(f"Total True Positives (TP) : {total_tp}")
    print(f"Total False Positives (FP): {total_fp}")
    print(f"Total False Negatives (FN): {total_fn}")
    print("-" * 50)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1_score:.4f}")
    print("="*50 + "\n")


def process_image_with_tiling(image_path, model, output_dir, tile_size=1024, overlap=0.20):
    """
    Splits a large image into tiles, runs YOLO inference, stitches predictions,
    applies safe NMS to prevent deleting dense vehicles, and saves visualization.
    Returns: Final stitched predictions, image width, and image height.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}. Skipping...")
        return None, 0, 0

    img_h, img_w = img.shape[:2]
    stride = int(tile_size * (1 - overlap))
    thickness = max(2, int(max(img_h, img_w) / 1500))
    
    # print(f"\nProcessing Image: {os.path.basename(image_path)}")
    # print(f"Original size: {img_w}x{img_h}. Tiling with {tile_size}x{tile_size} and {overlap*100:.0f}% overlap...")

    all_boxes =[]     
    all_scores = []    
    all_classes = []   
    all_raw_preds =[] 
    is_obb = False     

    y_starts = list(range(0, img_h, stride))
    x_starts = list(range(0, img_w, stride))

    # 1. Tile the image and predict
    for y in y_starts:
        for x in x_starts:
            y1, y2 = y, min(y + tile_size, img_h)
            x1, x2 = x, min(x + tile_size, img_w)

            if y2 - y1 < tile_size and img_h >= tile_size:
                y1 = max(0, img_h - tile_size)
                y2 = y1 + tile_size
            if x2 - x1 < tile_size and img_w >= tile_size:
                x1 = max(0, img_w - tile_size)
                x2 = x1 + tile_size

            tile = img[y1:y2, x1:x2]
            if tile.size == 0:
                continue

            results = model.predict(tile, imgsz=tile_size, conf=0.001, verbose=False)
            result = results[0]

            # 2. Extract predictions and shift to the global coordinate system
            if hasattr(result, 'obb') and result.obb is not None:
                is_obb = True
                if len(result.obb) > 0:
                    obbs = result.obb.xyxyxyxy.cpu().numpy()
                    confs = result.obb.conf.cpu().numpy()
                    cls_ids = result.obb.cls.cpu().numpy()

                    for obb, conf, cls_id in zip(obbs, confs, cls_ids):
                        global_obb = obb + np.array([x1, y1])
                        min_x, min_y = np.min(global_obb, axis=0)
                        max_x, max_y = np.max(global_obb, axis=0)
                        
                        max_x = max(min_x + 1, max_x)
                        max_y = max(min_y + 1, max_y)
                        
                        all_boxes.append([min_x, min_y, max_x, max_y])
                        all_scores.append(conf)
                        all_classes.append(cls_id)
                        all_raw_preds.append(global_obb)
                        
            elif hasattr(result, 'boxes') and result.boxes is not None:
                if len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    cls_ids = result.boxes.cls.cpu().numpy()

                    for box, conf, cls_id in zip(boxes, confs, cls_ids):
                        global_box = box + np.array([x1, y1, x1, y1])
                        x_min, y_min, x_max, y_max = global_box
                        
                        x_max = max(x_min + 1, x_max)
                        y_max = max(y_min + 1, y_max)
                        
                        all_boxes.append([x_min, y_min, x_max, y_max])
                        all_scores.append(conf)
                        all_classes.append(cls_id)
                        all_raw_preds.append(global_box)

    # print(f"Total raw detections across all tiles: {len(all_boxes)}")

    final_preds =[]

    # 3. Global Non-Maximum Suppression (NMS)
    if len(all_boxes) > 0:
        boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
        classes_tensor = torch.tensor(all_classes, dtype=torch.float32)

        keep_indices = torchvision.ops.batched_nms(boxes_tensor, scores_tensor, classes_tensor, iou_threshold=0.65)
        # print(f"Detections remaining after global NMS: {len(keep_indices)}")

        # 4. Visualize and Collect Finalized predictions
        for idx in keep_indices:
            idx = int(idx)
            raw_pred = all_raw_preds[idx]
            cls_id = int(all_classes[idx])
            score = float(all_scores[idx])
            
            color = (0, 255, 0) if cls_id % 2 == 0 else (255, 0, 0) 
            
            if is_obb:
                pts = np.float32(raw_pred).reshape((-1, 2))
                cv2.polylines(img, [np.int32(pts)], isClosed=True, color=color, thickness=thickness)
            else:
                x_min, y_min, x_max, y_max = map(float, raw_pred)
                pts = np.array([[x_min, y_min], [x_max, y_min],[x_max, y_max], [x_min, y_max]])
                cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)

            # Store mapping for evaluation
            final_preds.append({
                'poly': pts,
                'class': cls_id,
                'score': score
            })
    else:
        print("No detections found! Try lowering the 'conf' threshold in model.predict().")

    # 5. Save the final annotated image
    os.makedirs(output_dir, exist_ok=True)
    img_name = os.path.basename(image_path)
    save_path = os.path.join(output_dir, f"stitched_{img_name}")
    cv2.imwrite(save_path, img)
    # print(f"Saved annotated image to: {save_path}")

    return final_preds, img_w, img_h


def main():
    # --- CONFIGURATION FLAGS ---
    EVALUATE_RESULTS = True  # Set to False to skip evaluation
    # ---------------------------

    weights_path = '/home/adamm/Documents/FSOD/codes/FSC-Embedding-Based-Satellites/FSC-Embedding-Based-Satellites/Outputs/YOLO26_Train_FAIR1M/fair1m_vehicle_training/weights/best.pt'
    output_dir = '/home/adamm/Documents/FSOD/codes/FSC-Embedding-Based-Satellites/FSC-Embedding-Based-Satellites/Outputs/Test_YOLOn_TrOn_FAIR1M_On_DOTA'
    input_dir = '/home/adamm/Documents/FSOD/Data/DOTA/DOTA_Vehicles/val/images/'
    
    label_dir = "/home/adamm/Documents/FSOD/Data/DOTA/DOTA_Vehicles/val/labels/"
    iou_thresh = 0.25

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_dir):
        print(f"Directory not found: {input_dir}")
        return

    print(f"Loading YOLO Model from {weights_path}...")
    model = YOLO(weights_path)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
    image_files =[f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"No valid image files found in {input_dir}.")
        return

    print(f"Found {len(image_files)} images. Starting batch processing...")

    all_predictions_dict = {}

    for img_file in tqdm(image_files, desc="Prediction"):
        full_image_path = os.path.join(input_dir, img_file)
        
        preds, img_w, img_h = process_image_with_tiling(
            image_path=full_image_path,
            model=model,
            output_dir=output_dir,
            tile_size=1024,
            overlap=0.20
        )
        
        if preds is not None:
            all_predictions_dict[img_file] = {
                'preds': preds,
                'width': img_w,
                'height': img_h
            }
        
    print("\n=============================================")
    print(f"Batch processing complete! All results are in:\n{output_dir}")
    print("=============================================")

    # 6. Optional Evaluation Step
    if EVALUATE_RESULTS:
        if not os.path.exists(label_dir):
            print(f"\nWarning: Label directory not found at {label_dir}. Evaluation skipped.")
        else:
            evaluate_predictions(all_predictions_dict, label_dir, iou_thresh)


if __name__ == '__main__':
    main()