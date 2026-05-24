import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from collections import defaultdict
import cv2
import colorsys

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ==========================================
# Global Constants
# ==========================================

# Class Mapping (Original YOLO labels)
CLASS_MAP = {
    '0': 'ExtremelyLongHeavyDuty', '1': 'LongHeavyDuty', '2': 'HeavyDuty',
    '3': 'MediumStandard', '4': 'MediumSmall', '5': 'Small',
    '6': 'ExtremelyLongHeavyDutyTraileronly', '7': 'HeavyDutyTractorTruck',
    '8': 'CementMixerTrucks', '9': 'Bulldozers', '10': 'MobileCranes',
    '11': 'Forklifts', '12': 'TruckTractor', '13': 'Other'
}

# The classes that were skipped during training
EXCLUDED_CLASSES = {'HeavyDutyTractorTruck', 'ExtremelyLongHeavyDuty', 'Forklifts'}

# Generate unique, distinct colors for each class (in BGR format for OpenCV)
CLASS_COLORS = {}
classes_list = list(CLASS_MAP.values())
for i, cls in enumerate(classes_list):
    hue = i / len(classes_list)
    # Convert HSV to RGB, scale to 255, then swap to BGR for OpenCV
    r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.8, 0.9)]
    CLASS_COLORS[cls] = (b, g, r)


# ==========================================
# 1. Neural Network Definition
# ==========================================
class MLP_PyTorch(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP_PyTorch, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.dropout1 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(p=0.4)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = self.dropout1(h1)
        h2 = F.relu(self.fc2(h1))
        h2_drop = self.dropout2(h2)
        logits = self.fc3(h2_drop)
        return logits, h2

# ==========================================
# 2. Main Evaluation Pipeline
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MLP Model")
    
    # Base Data Path
    parser.add_argument('--base_dir', type=str, default='/home/adamm/Documents/FSOD/Data/Lavyanut/Images/test', help="Base directory for test data")
    
    # Model Output Directories
    parser.add_argument('--train_output_dir', type=str, default='./Output_MLP_Training', help="Directory where model was saved during training")
    parser.add_argument('--train_model_dir', type=str, default='models/MLP', help="Directory where model was saved during training")
    
    # Evaluation Output Directory
    parser.add_argument('--eval_output_dir', type=str, default='./Output_MLP_Evaluation', help="Directory to save evaluation results")
    
    # Metric
    parser.add_argument('--distance_metric', type=str, default='cosine', choices=['l2', 'cosine', 'logits'], help="Must match what was used in training")
    
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- Setup Dynamic Paths ---
    TEST_EMB_DIR = os.path.join(args.base_dir, 'Obj_Embs')
    TEST_IMG_DIR = os.path.join(args.base_dir, 'images')
    TEST_LBL_DIR = os.path.join(args.base_dir, 'labels')

    MODELS_DIR = args.train_model_dir
    PROTOTYPES_DIR = os.path.join(args.train_output_dir, 'prototypes')

    IMG_RESULTS_DIR = os.path.join(args.eval_output_dir, 'Img_Results')

    os.makedirs(args.eval_output_dir, exist_ok=True)
    os.makedirs(IMG_RESULTS_DIR, exist_ok=True)

    # --- Load Scaler and Label Encoder ---
    le_path = os.path.join(MODELS_DIR, 'label_encoder.joblib')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
    
    if not os.path.exists(le_path) or not os.path.exists(scaler_path):
        print("Error: Could not find trained Scaler or LabelEncoder. Run training script first.")
        return

    le = load(le_path)
    scaler = load(scaler_path)
    num_classes = len(le.classes_)
    print(f"Loaded Label Encoder ({num_classes} classes) and Scaler.")

    # --- Load Test Data ---
    test_files = glob.glob(os.path.join(TEST_EMB_DIR, '*.npy'))
    if not test_files:
        print(f"Error: No embeddings found in {TEST_EMB_DIR}")
        return

    X_test, y_test_true, filenames = [], [], []
    
    print("Loading Test Embeddings...")
    for f in test_files:
        filename = os.path.basename(f)
        
        found_cls = None
        for cls in CLASS_MAP.values():
            if f"_{cls}_" in filename:
                found_cls = cls
                break
                
        if found_cls and found_cls not in EXCLUDED_CLASSES:
            emb = np.load(f).flatten()
            X_test.append(emb)
            y_test_true.append(found_cls)
            filenames.append(filename)

    X_test = np.array(X_test)
    X_test_scaled = scaler.transform(X_test)
    y_test_encoded = le.transform(y_test_true)

    # --- Load Model ---
    model_path = os.path.join(MODELS_DIR, 'best_mlp.pth')
    model = MLP_PyTorch(X_test.shape[1], num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    centers_tensor = None
    if args.distance_metric in ['l2', 'cosine']:
        centers_list = []
        for cls_name in le.classes_:
            center_path = os.path.join(PROTOTYPES_DIR, f"{cls_name}_center.npy")
            centers_list.append(np.load(center_path))
        centers_tensor = torch.FloatTensor(np.array(centers_list)).to(device)
        centers_tensor = F.normalize(centers_tensor, p=2, dim=1) if args.distance_metric == 'cosine' else centers_tensor

    # --- Perform Inference ---
    print("\nRunning Inference...")
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    with torch.no_grad():
        logits, hidden = model(X_test_tensor)
        
        if args.distance_metric == 'logits':
            scores = logits
            probabilities = F.softmax(scores, dim=1)
            confidences, preds_encoded = torch.max(probabilities, 1)
        elif args.distance_metric == 'cosine':
            hidden_norm = F.normalize(hidden, p=2, dim=1)
            scores = torch.mm(hidden_norm, centers_tensor.t())
            # For cosine, use a softmax with temperature to get a 0-1 probability curve
            probabilities = F.softmax(scores * 5.0, dim=1) 
            confidences, preds_encoded = torch.max(probabilities, 1)
        else: # L2
            scores = -torch.cdist(hidden, centers_tensor, p=2.0)
            probabilities = F.softmax(scores, dim=1)
            confidences, preds_encoded = torch.max(probabilities, 1)

    preds_encoded = preds_encoded.cpu().numpy()
    confidences = confidences.cpu().numpy()
    y_pred_names = le.inverse_transform(preds_encoded)

    # --- Dictionary for Image Mapping ---
    predictions_dict = {}
    for i in range(len(filenames)):
        lookup_key = filenames[i].replace('.npy', '')
        predictions_dict[lookup_key] = (y_pred_names[i], confidences[i])

    # --- Calculate & Save Metrics ---
    acc = accuracy_score(y_test_true, y_pred_names)
    print(f"\n✅ Overall Accuracy: {acc * 100:.2f}%\n")
    
    report = classification_report(y_test_true, y_pred_names, digits=4)
    print(report)
    
    with open(os.path.join(args.eval_output_dir, 'Metrics.txt'), 'w') as f:
        f.write(f"Overall Accuracy: {acc * 100:.2f}%\n")
        f.write(f"Prediction Mode: {args.distance_metric.upper()}\n\n")
        f.write(report)

    # --- Plot Confusion Matrices ---
    print("Generating Confusion Matrices...")
    
    # 1. Absolute Count Confusion Matrix
    cm = confusion_matrix(y_test_true, y_pred_names, labels=le.classes_)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Test Set Confusion Matrix (Absolute Counts)\nDistance: {args.distance_metric.upper()}', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(args.eval_output_dir, 'Confusion_Matrix.png'), dpi=300)
    plt.close() # Close figure to free up memory

    # 2. Normalized (Percentage) Confusion Matrix
    cm_normalized = confusion_matrix(y_test_true, y_pred_names, labels=le.classes_, normalize='true')
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1,
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Normalized Test Set Confusion Matrix (Row Percentages)\nDistance: {args.distance_metric.upper()}', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(args.eval_output_dir, 'Confusion_Matrix_Normalized.png'), dpi=300)
    plt.close()

    # --- Draw Oriented Bounding Boxes on Original Images ---
    print("\nDrawing Object Predictions on Original Images...")
    
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    test_images = [img for img in os.listdir(TEST_IMG_DIR) if os.path.splitext(img)[1].lower() in valid_extensions]
    
    for img_name in test_images:
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        txt_path = os.path.join(TEST_LBL_DIR, f"{base_name}.txt")
        
        if not os.path.exists(txt_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_h, img_w = img.shape[:2]
        class_counts = defaultdict(int)
        
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                    
                class_id = parts[0]
                true_class_name = CLASS_MAP.get(class_id, "Unknown")
                class_counts[true_class_name] += 1
                count = class_counts[true_class_name]
                
                # Check if this object was evaluated (not excluded)
                lookup_key = f"{base_name}_{true_class_name}_{count}"
                
                # Extract OBB coordinates
                coords = [float(p) for p in parts[1:9]]
                pts = np.array([
                    [coords[0] * img_w, coords[1] * img_h],
                    [coords[2] * img_w, coords[3] * img_h],
                    [coords[4] * img_w, coords[5] * img_h],
                    [coords[6] * img_w, coords[7] * img_h]
                ], np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                # Determine Box Color and Label
                if lookup_key in predictions_dict:
                    pred_cls, conf = predictions_dict[lookup_key]
                    color = CLASS_COLORS.get(pred_cls, (255, 255, 255)) 
                    label_text = f"{pred_cls} {conf:.2f}"
                else:
                    color = (128, 128, 128)
                    label_text = f"Excluded ({true_class_name})"
                
                # Draw the Polygon (OBB)
                cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
                
                # ==========================================
                # NEW: Fixed, uniform, small text scaling
                # ==========================================
                min_x, min_y = np.min(pts[:, 0, 0]), np.min(pts[:, 0, 1])
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.45  # Fixed, small scale for ALL objects
                thickness = 1      # Thin line thickness to keep text clean
                
                (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
                
                # Draw a filled rectangle behind the text for readability
                cv2.rectangle(img, (min_x, min_y - text_h - 4), (min_x + text_w, min_y), color, -1)
                
                # Make text black or white depending on the background brightness
                brightness = (color[2] * 299 + color[1] * 587 + color[0] * 114) / 1000
                text_color = (0, 0, 0) if brightness > 125 else (255, 255, 255)
                
                # Write the text
                cv2.putText(img, label_text, (min_x, min_y - 2), font, font_scale, text_color, thickness)
        
        # Save Output Image
        save_img_path = os.path.join(IMG_RESULTS_DIR, img_name)
        cv2.imwrite(save_img_path, img)

    print(f"\n✅ Finished! Evaluation outputs saved to {args.eval_output_dir}")

if __name__ == "__main__":
    main()