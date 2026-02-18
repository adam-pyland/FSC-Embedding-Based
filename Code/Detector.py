from ultralytics import YOLO
import argparse
from datetime import datetime
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import torch
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)
print("Working directory:", os.getcwd())

current_date = datetime.now().strftime("%d_%-m_%y")

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # Ensure this points to your .pt file now
    parser.add_argument("--weights", type=str, default="models/Yolo_Trained_on_Cars.pt", help="initial weights path")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence thr")
    parser.add_argument("--source", type=str, default="/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Filtered_Dataset/Images/Val/", help="inference source")
    
    parser.add_argument("--file_list", type=str, default="/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Filtered_Dataset/Image_lists/Val_only_Vehicles_No_BCV.txt", help="Path to TXT file")
    parser.add_argument("--xml_dir", type=str, default="/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Filtered_Dataset/VOC_Annotations/Base_and_Big_Cargo_Vehicles_Hor_Boxes/Val/", help="Path to XML annotations")
    parser.add_argument("--tile", type=bool, default=True, help="If True, tile large images.")

    parser.add_argument("--iou", type=float, default=0.2, help="iou thr")
    parser.add_argument("--batch", type=int, default=1, help="total batch size")
    parser.add_argument("--imgsz", type=int, default=1920, help="inference size")
    parser.add_argument("--device", default="", help="cuda device")
    parser.add_argument("--project", default="runs", help="save to project/name")
    parser.add_argument("--name", default="FAIR1M_Val_Preds", help="save to project/name")
    
    # Visualization args (We handle these manually now, but keeping args doesn't hurt)
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--save_txt", type=bool, default=True)
    parser.add_argument("--iou_thresh", type=float, default=0.25) 

    return parser.parse_known_args()[0] if known else parser.parse_args()

def calculate_polygon_iou(poly1, poly2, img_shape=(10000, 10000)):
    mask1 = np.zeros(img_shape, dtype=np.uint8)
    mask2 = np.zeros(img_shape, dtype=np.uint8)
    pts1 = np.array(poly1, np.int32).reshape((-1, 1, 2))
    pts2 = np.array(poly2, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask1, [pts1], 1)
    cv2.fillPoly(mask2, [pts2], 1)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def get_gt_polygons(xml_path):
    polygons = []
    if not os.path.exists(xml_path): return polygons
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            poly = obj.find('polygon')
            if poly is not None:
                pts = []
                for i in range(4):
                    pts.append([float(poly.find(f'x{i}').text), float(poly.find(f'y{i}').text)])
                polygons.append(np.array(pts))
            else:
                bnd = obj.find('bndbox')
                if bnd is not None:
                    xmin = float(bnd.find('xmin').text)
                    ymin = float(bnd.find('ymin').text)
                    xmax = float(bnd.find('xmax').text)
                    ymax = float(bnd.find('ymax').text)
                    polygons.append(np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]))
    except: pass
    return polygons

# --- CHANGED: Removed project/name args here to stop internal saving ---
def run_inference(model, img_bw, inference_sz, conf, iou, use_tiling=False):
    h, w = img_bw.shape[:2]
    all_preds = []

    # 1. STANDARD INFERENCE
    if not use_tiling or (h <= inference_sz and w <= inference_sz):
        # --- CHANGED: Force save=False, save_txt=False ---
        results = model.track(img_bw, conf=conf, iou=iou, imgsz=inference_sz, 
                              verbose=False, persist=True, save=False, save_txt=False)
        
        if len(results) > 0:
            if results[0].obb is not None:
                return list(zip(results[0].obb.xyxyxyxy.cpu().numpy(), results[0].obb.conf.cpu().numpy()))
            elif results[0].boxes is not None:
                boxes = results[0].boxes.data.cpu().numpy()
                for b in boxes:
                    # Convert box to poly format
                    all_preds.append((np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]), b[4]))
        return all_preds

    # 2. TILED INFERENCE
    step = inference_sz 
    for y in range(0, h, step):
        for x in range(0, w, step):
            h_crop = min(inference_sz, h - y)
            w_crop = min(inference_sz, w - x)
            crop = img_bw[y : y + h_crop, x : x + w_crop]
            
            if h_crop < inference_sz or w_crop < inference_sz:
                pad_h = inference_sz - h_crop
                pad_w = inference_sz - w_crop
                crop = cv2.copyMakeBorder(crop, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))

            # --- CHANGED: Force save=False ---
            results = model.track(crop, conf=conf, iou=iou, imgsz=inference_sz, 
                                  verbose=False, persist=True, save=False)
            
            if len(results) > 0:
                if results[0].obb is not None:
                    polys = results[0].obb.xyxyxyxy.cpu().numpy()
                    confs = results[0].obb.conf.cpu().numpy()
                    for p_poly, p_conf in zip(polys, confs):
                        p_poly[:, 0] += x
                        p_poly[:, 1] += y
                        all_preds.append((p_poly, p_conf))
                elif results[0].boxes is not None:
                    boxes = results[0].boxes.data.cpu().numpy()
                    for box in boxes:
                        poly = np.array([
                            [box[0]+x, box[1]+y], [box[2]+x, box[1]+y], 
                            [box[2]+x, box[3]+y], [box[0]+x, box[3]+y]
                        ])
                        all_preds.append((poly, box[4]))

    return all_preds

if __name__ == '__main__':
    opt = parse_opt()
    model = YOLO(opt.weights, task='obb')
    
    # --- CHANGED: Setup One Clean Output Directory ---
    run_name = opt.name + '_' + current_date
    save_dir = os.path.join(opt.project, run_name)
    labels_dir = os.path.join(save_dir, "labels")
    images_dir = os.path.join(save_dir, "images")
    
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    # Auto-detect Training Size
    model_imgsz = opt.imgsz
    try:
        if hasattr(model, 'overrides') and 'imgsz' in model.overrides:
            model_imgsz = model.overrides['imgsz']
    except: pass
    inference_sz = model_imgsz if opt.tile else opt.imgsz
    
    print(f"Output Directory: {save_dir}")
    print(f"Mode: {'Tiling' if opt.tile else 'Standard Resize'} | Size: {inference_sz}")

    if opt.file_list and os.path.exists(opt.file_list):
        with open(opt.file_list, 'r') as f:
            file_names = [line.strip() for line in f.readlines() if line.strip()]
    else:
        print("Please provide a valid --file_list txt file.")
        exit()

    all_tp_fp = [] 
    total_gt_objects = 0

    print(f"Processing {len(file_names)} images...")

    for fname in tqdm(file_names):
        img_path = os.path.join(opt.source, fname + ".jpg") 
        xml_path = os.path.join(opt.xml_dir, fname + ".xml")

        if not os.path.exists(img_path): continue

        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_bw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # --- CHANGED: Simplified Call (Removed project/name passing) ---
        preds = run_inference(model, img_bw, inference_sz, opt.conf, opt.iou, use_tiling=opt.tile)

        # --- CHANGED: Manual Saving for BOTH .txt and .jpg ---
        if len(preds) > 0:
            # 1. Save Text Labels
            txt_path = os.path.join(labels_dir, fname + ".txt")
            with open(txt_path, "w") as f:
                for poly, conf in preds:
                    flat_poly = poly.flatten() 
                    line = f"0 " + " ".join([f"{coord:.2f}" for coord in flat_poly]) + f" {conf:.4f}\n"
                    f.write(line)
            
            # 2. Save Image with Detections
            if opt.save: # Check if user wants images saved
                img_vis = img.copy()
                for poly, conf in preds:
                    # Draw Polygon (Green, thickness 2)
                    pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img_vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                    
                    # Optional: Draw Score
                    label = f"{conf:.2f}"
                    cv2.putText(img_vis, label, (pts[0][0][0], pts[0][0][1] - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                out_img_path = os.path.join(images_dir, fname + ".jpg")
                cv2.imwrite(out_img_path, img_vis)

        # --- EVALUATION (Unchanged) ---
        gt_polys = get_gt_polygons(xml_path)
        total_gt_objects += len(gt_polys)
        mask_shape = img_bw.shape[:2]
        detected_gt_indices = set()
        
        for pred_poly, conf in preds:
            best_iou = 0
            best_gt_idx = -1
            for i, gt_poly in enumerate(gt_polys):
                iou_val = calculate_polygon_iou(pred_poly, gt_poly, img_shape=mask_shape)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = i
            
            if best_iou >= opt.iou_thresh and best_gt_idx not in detected_gt_indices:
                all_tp_fp.append((conf, 1)) 
                detected_gt_indices.add(best_gt_idx)
            else:
                all_tp_fp.append((conf, 0))

    # --- SAVE METRICS TO TXT ---
    metrics_path = os.path.join(save_dir, "evaluation_metrics.txt")
    
    with open(metrics_path, "w") as f:
        f.write(f"Evaluation Run: {current_date}\n")
        f.write(f"Weights: {opt.weights}\n")
        f.write(f"ImgSz: {inference_sz} | Tiling: {opt.tile}\n")
        f.write("-" * 30 + "\n")
        
        if total_gt_objects > 0:
            all_tp_fp.sort(key=lambda x: x[0], reverse=True)
            tps = np.cumsum([x[1] for x in all_tp_fp])
            fps = np.cumsum([1 - x[1] for x in all_tp_fp])
            recalls = tps / total_gt_objects
            precisions = tps / (tps + fps + 1e-16)
            ap = np.trapz(precisions, recalls)
            
            print(f"\nResults saved to: {metrics_path}")
            print(f"Precision: {precisions[-1]:.4f}")
            print(f"Recall:    {recalls[-1]:.4f}")
            print(f"mAP@{opt.iou_thresh}:    {ap:.4f}")
            
            f.write(f"Total Images:     {len(file_names)}\n")
            f.write(f"Total GT Objects: {total_gt_objects}\n")
            f.write(f"Total Detections: {len(all_tp_fp)}\n")
            f.write(f"Precision:        {precisions[-1]:.4f}\n")
            f.write(f"Recall:           {recalls[-1]:.4f}\n")
            f.write(f"mAP@{opt.iou_thresh}:           {ap:.4f}\n")
        else:
            print("No GT objects found.")
            f.write("No Ground Truth objects found to evaluate.\n")