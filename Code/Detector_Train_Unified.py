import os
import cv2
import glob
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from shapely.geometry import Polygon

# ---------------------------------------------------------
# 1. TRAINING FUNCTION
# ---------------------------------------------------------
def train_yolo26_obb():
    print("--- Starting/Resuming YOLO26 OBB Training ---")
    
    # Paths
    project_dir = "/Data/Projects/Satellite/code/adamm/FSC-Embedding-Based/Outputs/"
    name = "YOLO26_Unified_Training_Results"
    data_yaml = "/Data/Projects/Satellite/data/Satellite-Data-Unified/dataset.yaml"
    
    # Path to check if training was interrupted
    last_weights = os.path.join(project_dir, name, "weights", "last.pt")
    
    if os.path.exists(last_weights):
        # ---------------------------------------------------------
        # RESUME INTERRUPTED TRAINING
        # ---------------------------------------------------------
        print(f"Found interrupted training. Resuming from: {last_weights}")
        model = YOLO(last_weights)
        
        # When resuming, you only need to pass resume=True. 
        # YOLO remembers the epochs, batch size, and imgsz from the previous run.
        model.train(resume=True)
        
    else:
        # ---------------------------------------------------------
        # START FRESH TRAINING
        # ---------------------------------------------------------
        print("No previous run found. Starting fresh training...")
        
        # NOTE: If "yolo26n-obb.pt" is an official model, it will auto-download.
        # If it's a custom file, make sure it is in the current working directory.
        model = YOLO("yolo26n-obb.pt") 
        
        model.train(
            data=data_yaml,
            epochs=100,
            imgsz=1024,
            batch=8,           # Adjust based on your GPU VRAM
            project=project_dir,
            name=name,
            device="0",        # Set to 'cpu' if no GPU is available
            workers=8
        )
    
    # Return the path to the best weights
    best_weights = os.path.join(project_dir, name, "weights", "best.pt")
    return best_weights

# ---------------------------------------------------------
# 2. TILING & STITCHING UTILITIES
# ---------------------------------------------------------
def compute_polygon_iou(box1, box2):
    """Calculate Intersection over Union for two oriented bounding boxes."""
    try:
        poly1, poly2 = Polygon(box1), Polygon(box2)
        # Fix invalid geometries if any
        if not poly1.is_valid: poly1 = poly1.buffer(0)
        if not poly2.is_valid: poly2 = poly2.buffer(0)
        
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
        return inter_area / union_area if union_area > 0 else 0.0
    except Exception:
        return 0.0

def nms_obb(boxes, confs, classes, iou_thresh=0.25):
    """Apply Non-Maximum Suppression to stitched OBBs across tile boundaries."""
    if len(boxes) == 0:
        return [], [], []
    
    indices = np.argsort(confs)[::-1]
    keep_boxes, keep_confs, keep_classes = [], [],[]
    
    while len(indices) > 0:
        curr_idx = indices[0]
        keep_boxes.append(boxes[curr_idx])
        keep_confs.append(confs[curr_idx])
        keep_classes.append(classes[curr_idx])
        
        if len(indices) == 1:
            break
            
        rest_indices = indices[1:]
        ious = np.array([compute_polygon_iou(boxes[curr_idx], boxes[i]) for i in rest_indices])
        
        # Keep boxes that do not overlap heavily with the current highest confidence box
        indices = rest_indices[ious <= iou_thresh]
        
    return keep_boxes, keep_confs, keep_classes

def get_tiles(image_shape, tile_size=1024, overlap=256):
    """Generate tile coordinates with sliding window overlap."""
    H, W = image_shape[:2]
    stride = tile_size - overlap
    tiles =[]
    
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1, x1 = y, x
            y2, x2 = y1 + tile_size, x1 + tile_size
            
            # Shift back if the tile exceeds image boundaries
            if y2 > H:
                y2 = H
                y1 = max(0, H - tile_size)
            if x2 > W:
                x2 = W
                x1 = max(0, W - tile_size)
                
            tiles.append({'y1': y1, 'y2': y2, 'x1': x1, 'x2': x2})
            
    # Remove duplicates if the image was smaller than tile_size
    return [dict(t) for t in {tuple(d.items()) for d in tiles}]

# ---------------------------------------------------------
# 3. INFERENCE & STITCHING EXECUTION
# ---------------------------------------------------------
def process_val_images(weights_path, val_images_dir, output_dir, tile_size=1024, overlap=256):
    print("\n--- Starting Tiled Inference & Stitching on Validation Set ---")
    model = YOLO(weights_path)
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = glob.glob(os.path.join(val_images_dir, "*.*"))
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        H, W = img.shape[:2]
        tiles = get_tiles(img.shape, tile_size, overlap)
        
        all_boxes, all_confs, all_classes = [], [],[]
        
        for t in tiles:
            tile_img = img[t['y1']:t['y2'], t['x1']:t['x2']]
            
            # Run inference on the tile
            results = model.predict(tile_img, imgsz=tile_size, verbose=False)
            result = results[0]
            
            if result.obb is not None:
                # Get corner coordinates shape (N, 4, 2)
                obbs = result.obb.xyxyxyxy.cpu().numpy()
                confs = result.obb.conf.cpu().numpy()
                cls = result.obb.cls.cpu().numpy()
                
                for i in range(len(obbs)):
                    # Shift coordinates back to the global image space
                    shifted_obb = obbs[i] + np.array([t['x1'], t['y1']])
                    all_boxes.append(shifted_obb)
                    all_confs.append(confs[i])
                    all_classes.append(cls[i])
                    
        # Apply NMS to remove duplicates across overlapping boundaries
        final_boxes, final_confs, final_classes = nms_obb(all_boxes, all_confs, all_classes)
        
        # Save results in Normalized YOLO OBB format
        base_name = Path(img_path).stem
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        
        with open(txt_path, 'w') as f:
            for box, c in zip(final_boxes, final_classes):
                # Normalize by Image Width and Height
                norm_box = box.astype(float)
                norm_box[:, 0] /= W
                norm_box[:, 1] /= H
                
                # Flatten (x1 y1 x2 y2 x3 y3 x4 y4)
                coords_str = " ".join([f"{pt:.6f}" for pt in norm_box.flatten()])
                f.write(f"{int(c)} {coords_str}\n")
                
        print(f"Processed & Stitched: {Path(img_path).name} -> Found {len(final_boxes)} objects.")

if __name__ == "__main__":
    # 1. Train the model
    best_model_path = train_yolo26_obb()
    
    print(f"Training complete! Best weights are located at: {best_model_path}")
    
    # 2. Run tile-and-stitch inference on Validation dataset
    val_dir = "/Data/Projects/Satellite/data/Satellite-Data-Unified/val-full-imgs/images"
    stitch_output_dir = "/Data/Projects/Satellite/code/adamm/FSC-Embedding-Based/Outputs/YOLO26_Unified_Training_Results/stitched_val_labels"
    
    process_val_images(best_model_path, val_dir, stitch_output_dir)
    print(f"\nCompleted! Stitched validation labels saved to: {stitch_output_dir}")