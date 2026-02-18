import argparse
import os
import cv2
from tqdm import tqdm
from datetime import datetime
from ultralytics import YOLO

# SAHI imports
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
# os.chdir(parent_dir) # Optional depending on your folder structure

current_date = datetime.now().strftime("%d_%-m_%y")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="models/Yolo_Trained_on_Cars.pt", help="weights path")
    parser.add_argument("--source", type=str, default="Input_Images/Vehicles_Base_Class", help="image folder path")
    parser.add_argument("--conf", type=float, default=0.0002, help="confidence threshold") # Higher than 0.01 is usually better for SAHI
    parser.add_argument("--device", default="cuda:0", help="cuda device")
    parser.add_argument("--project", default=f"{parent_dir}/Results_with_SAHI", help="save results path")
    parser.add_argument("--name", default="FAIR1M", help="experiment name")
    
    # SAHI Specific Args
    parser.add_argument("--slice_size", type=int, default=512, help="Size of the crop")
    parser.add_argument("--overlap_ratio", type=float, default=0.2, help="Overlap between slices (0.2 = 20%)")
    
    return parser.parse_args()

def save_sahi_predictions_as_txt(prediction_result, output_dir, file_name, img_width, img_height):
    """
    Converts SAHI prediction results to the format expected by your eval script:
    class x1 y1 x2 y2 x3 y3 x4 y4 conf (Normalized)
    """
    txt_path = os.path.join(output_dir, file_name + ".txt")
    
    with open(txt_path, "w") as f:
        for obj in prediction_result.object_prediction_list:
            cls_id = obj.category.id
            score = obj.score.value
            
            # Get Bounding Box (XYXY)
            bbox = obj.bbox # [minx, miny, maxx, maxy]
            minx, miny, maxx, maxy = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
            
            # Convert HBB (Horizontal BBox) to 4 points (Polygon style)
            # Top-Left, Top-Right, Bottom-Right, Bottom-Left
            points = [
                (minx, miny),
                (maxx, miny),
                (maxx, maxy),
                (minx, maxy)
            ]
            
            # Normalize coordinates (0-1)
            norm_coords = []
            for (x, y) in points:
                nx = x / img_width
                ny = y / img_height
                # Clip to ensure valid range
                nx = max(0, min(1, nx))
                ny = max(0, min(1, ny))
                norm_coords.extend([nx, ny])
            
            # Format line: class p1x p1y p2x p2y p3x p3y p4x p4y conf
            line_parts = [str(cls_id)] + [f"{c:.6f}" for c in norm_coords] + [f"{score:.6f}"]
            f.write(" ".join(line_parts) + "\n")

def run():
    opt = parse_opt()
    
    # Paths setup
    run_name = f"{opt.name}_{current_date}"
    save_dir = os.path.join(opt.project, run_name)
    labels_dir = os.path.join(save_dir, "labels")
    vis_dir = os.path.join(save_dir, "visuals")
    
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    print(f"Initializing SAHI with model: {opt.weights}")
    
    # 1. Initialize SAHI Model Wrapper
    # Use 'yolov8' model_type for Ultralytics YOLOv11 (it shares the same codebase)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8', 
        model_path=opt.weights,
        confidence_threshold=opt.conf,
        device=opt.device
    )

    # Get list of images
    supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = [os.path.join(opt.source, f) for f in os.listdir(opt.source) if f.lower().endswith(supported_ext)]

    print(f"Starting inference on {len(image_files)} images...")

    for img_path in tqdm(image_files):
        # Read image to get dimensions
        image = cv2.imread(img_path)
        if image is None: 
            continue
        height, width = image.shape[:2]
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # 2. Perform Sliced Inference
        result = get_sliced_prediction(
            img_path,
            detection_model,
            slice_height=opt.slice_size,
            slice_width=opt.slice_size,
            overlap_height_ratio=opt.overlap_ratio,
            overlap_width_ratio=opt.overlap_ratio,
            perform_standard_pred=True, # Also detect on full image to catch large objects
            verbose=0
        )

        # 3. Save Text Labels (Normalized 8-point format)
        save_sahi_predictions_as_txt(result, labels_dir, base_name, width, height)

        # 4. Save Visualization (Optional)
        # SAHI's built-in visualization
        visual_path = os.path.join(vis_dir, base_name + ".png")
        result.export_visuals(export_dir=vis_dir, file_name=base_name)

    print(f"Done! Results saved to {save_dir}")

if __name__ == "__main__":
    run()