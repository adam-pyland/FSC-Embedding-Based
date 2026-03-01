from ultralytics import YOLO
import os
import shutil
import yaml

def run_inference(model, images_dir, output_dir):
    print("\n--- 1. Running Inference (Generating Visuals & Predicted Text files) ---")
    results = model.predict(
        source=images_dir,
        imgsz=1024,          
        conf=0.25,           
        save=True,           
        save_txt=True,       
        project=output_dir,  
        name="visualizations", 
        exist_ok=True,
        line_width=1,        # Thin boxes and small text
        show_conf=False      # Hides the confidence score for cleaner visuals
    )
    predicted_path = os.path.join(output_dir, "visualizations")
    print(f"✅ Visualizations and your model's predicted labels saved to: {predicted_path}")


def evaluate_metrics(model, base_data_dir, output_dir):
    print("\n--- 2. Evaluating Metrics (mAP, Precision, Recall) against Ground Truth ---")
    
    # Define paths
    images_dir = os.path.join(base_data_dir, "images")
    gt_labels_src = os.path.join(base_data_dir, "results", "labels")
    yolo_labels_dest = os.path.join(base_data_dir, "labels")
    
    # 1. Structure the Ground Truth for YOLO
    # YOLO automatically looks for a 'labels' folder adjacent to the 'images' folder.
    # We copy the ground truth from 'results/labels' to 'labels' so YOLO can find them.
    if not os.path.exists(yolo_labels_dest):
        print(f"Setting up ground truth folder for YOLO at: {yolo_labels_dest}")
        shutil.copytree(gt_labels_src, yolo_labels_dest, dirs_exist_ok=True)
    else:
        print("Ground truth folder already structured for YOLO evaluation.")

    # 2. Generate a temporary dataset.yaml for the evaluation
    yaml_path = os.path.join(base_data_dir, "eval_dataset.yaml")
    yaml_content = {
        "path": base_data_dir,
        "val": "images",         # Validation images folder
        "names": {0: "vehicle"}  # Class dictionary
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    # 3. Run YOLO validation to calculate metrics
    # Note: We do not set 'conf=0.25' here. mAP calculation requires testing ALL confidence levels 
    # to generate an accurate Precision-Recall curve. YOLO handles this automatically.
    print("Calculating metrics...")
    metrics = model.val(
        data=yaml_path,
        split='val',
        imgsz=1024,
        batch=16,
        project=output_dir,
        name="evaluation_metrics",
        exist_ok=True,
        plots=True # Automatically saves Precision-Recall curves and confusion matrices
    )

    # 4. Extract and print the specific metrics you requested
    print("\n" + "="*40)
    print("       FINAL EVALUATION METRICS       ")
    print("="*40)
    
    try:
        # OBB (Oriented Bounding Box) properties
        precision = metrics.obb.mp    # Mean Precision
        recall = metrics.obb.mr       # Mean Recall
        map50 = metrics.obb.map50     # mAP at IoU=0.50
        map50_95 = metrics.obb.map    # mAP at IoU=0.50:0.95
        
        print(f"Precision:         {precision:.4f}")
        print(f"Recall:            {recall:.4f}")
        print(f"mAP@50:            {map50:.4f}")
        print(f"mAP@50-95:         {map50_95:.4f}")
    except AttributeError:
        # Fallback just in case your model trained as standard horizontal boxes
        precision = metrics.box.mp
        recall = metrics.box.mr
        map50 = metrics.box.map50
        map50_95 = metrics.box.map
        
        print(f"Precision:         {precision:.4f}")
        print(f"Recall:            {recall:.4f}")
        print(f"mAP@50:            {map50:.4f}")
        print(f"mAP@50-95:         {map50_95:.4f}")

    print("="*40)
    metrics_path = os.path.join(output_dir, "evaluation_metrics")
    print(f"Detailed metric plots (PR-curves, F1-curves) saved to: {metrics_path}")


def main():
    # Define exact directories based on your paths
    model_path = "/home/adamm/Documents/FSOD/codes/FSC-Embedding-Based-Satellites/FSC-Embedding-Based-Satellites/Outputs/YOLO26_Train_FAIR1M/fair1m_vehicle_training/weights/best.pt"
    base_data_dir = "/home/adamm/Documents/FSOD/Data/Tagged_Data"
    images_dir = os.path.join(base_data_dir, "images")
    output_dir = os.path.join(base_data_dir, "results_after_training")

    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    # Step 1: Draw bounding boxes, make them thin, and save your model's raw text file predictions
    run_inference(model, images_dir, output_dir)
    
    # Step 2: Compare to the Ground Truth model's results to calculate mAP, Precision, and Recall
    # evaluate_metrics(model, base_data_dir, output_dir)

if __name__ == '__main__':
    main()