import os
import glob
import numpy as np
from joblib import load

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# Global Configuration
# ==========================================

# Select your evaluation metric: 'l2', 'cosine', or 'logits'
DISTANCE_METRIC = 'cosine'

# Set the target class to track (e.g., 'Trailer'). 
# If you want to track ALL classes collectively, set this to None.
TARGET_EVAL_CLASS = 'Trailer'

# Directories (Change these if they differ from your environment)
TRAIN_BASE_DIR = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Vehicle_Dataset/Image_Obj_Embs/train/base_class'
TRAIN_NOVEL_DIR = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Vehicle_Dataset/Image_Obj_Embs/train/novel_class_few_shot_trailer/'

VAL_BASE_DIR = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Vehicle_Dataset/Image_Obj_Embs/val/base_class'
VAL_NOVEL_DIR = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Vehicle_Dataset/Image_Obj_Embs/val/novel_class'

# Path to the directory where the MLP model, scaler, and label encoder are saved
LOSS_COMBINATION = 'focal_center'
SAVED_MODEL_DIR = f"models/MLP-Pytorch-Few-Shots-{LOSS_COMBINATION}-Loss-TOP3-{DISTANCE_METRIC}-{'Distance' if DISTANCE_METRIC == 'logits' else 'Logits'}-Based"

# Output directory for the text files
OUTPUT_DIR = f"Outputs/TopK_Evaluation_Lists-{LOSS_COMBINATION}-Loss-{DISTANCE_METRIC}-{'Distance' if DISTANCE_METRIC == 'logits' else 'Logits'}-Based"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALL_CLASSES =[
    'Bus', 'Dump Truck', 'Tractor', 
    'Truck Tractor', 'Excavator', 
    'Cargo Truck', 'Trailer'
]
SAFE_CLASS_NAMES =[cls.replace(" ", "_") for cls in ALL_CLASSES]

# ==========================================
# 1. PyTorch Model Definition
# ==========================================

class MLP_PyTorch(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP_PyTorch, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        logits = self.fc3(h2)
        return logits, h2
    
# ==========================================
# 2. Evaluation Helpers
# ==========================================

def compute_train_centers(model, dataloader, num_classes, device):
    """Calculates prototype centers required for 'cosine' or 'l2' metrics."""
    model.eval()
    centers = torch.zeros(num_classes, 256).to(device)
    counts = torch.zeros(num_classes).to(device)
    
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            _, h2 = model(features)
            centers.index_add_(0, labels, h2)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float))
            
    counts = counts.clamp(min=1e-8)
    centers = centers / counts.unsqueeze(1)
    return centers

def compute_prototype_scores(h2, centers, metric='cosine'):
    """Calculates distance/similarity to prototypes (Higher score = closer)."""
    if metric.lower() == 'cosine':
        h2_norm = F.normalize(h2, p=2, dim=1)
        centers_norm = F.normalize(centers, p=2, dim=1)
        scores = torch.mm(h2_norm, centers_norm.t()) 
    elif metric.lower() == 'l2':
        dist = torch.cdist(h2, centers, p=2.0)
        scores = -dist # Negate so highest value is the closest distance
    else:
        raise ValueError("Metric must be 'l2' or 'cosine'")
    return scores

def load_features_and_filenames(directory, X_list, y_list, filenames_list):
    if not os.path.exists(directory):
        print(f"Warning: Directory does not exist -> {directory}")
        return
        
    files = glob.glob(os.path.join(directory, '*.npy'))
    for f in files:
        filename = os.path.basename(f)
        for safe_cls in SAFE_CLASS_NAMES:
            if f"_{safe_cls}_" in filename:
                embedding = np.load(f).flatten()
                X_list.append(embedding)
                y_list.append(safe_cls.replace("_", " ")) 
                filenames_list.append(filename)
                break

# ==========================================
# 3. Main Evaluation Pipeline
# ==========================================

def main():
    print("="*50)
    print("Starting Top-K Ground Truth Evaluation")
    print(f"Prediction Mode: {DISTANCE_METRIC.upper()}")
    print(f"Target Class to Track: {TARGET_EVAL_CLASS if TARGET_EVAL_CLASS else 'ALL CLASSES'}")
    print("="*50)

    # 1. Load Scaler and Label Encoder
    scaler_file = os.path.join(SAVED_MODEL_DIR, 'saved_scaler_NO_CARS_VANS.joblib')
    le_file = os.path.join(SAVED_MODEL_DIR, 'saved_le_NO_CARS_VANS.joblib')
    best_model_file = os.path.join(SAVED_MODEL_DIR, 'best_mlp_NO_CARS_VANS.pth')

    if not (os.path.exists(scaler_file) and os.path.exists(le_file) and os.path.exists(best_model_file)):
        print(f"Error: Could not find model, scaler, or LabelEncoder in {SAVED_MODEL_DIR}")
        return

    scaler = load(scaler_file)
    le = load(le_file)
    num_classes = len(le.classes_)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Setup Model
    input_dim = getattr(scaler, 'n_features_in_', 1024)
    
    model = MLP_PyTorch(input_dim=input_dim, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(best_model_file, map_location=device))
    model.eval()

    # 3. Compute Training Centers (if required)
    train_centers = None
    if DISTANCE_METRIC in ['l2', 'cosine']:
        print("\nLoading Training Data to calculate prototype centers...")
        X_train, y_train, _ = [], [],[]
        load_features_and_filenames(TRAIN_BASE_DIR, X_train, y_train, _)
        load_features_and_filenames(TRAIN_NOVEL_DIR, X_train, y_train, _)
        
        X_train = np.array(X_train)
        y_train_encoded = le.transform(np.array(y_train))
        X_train_scaled = scaler.transform(X_train)
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train_encoded))
        train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=False)
        
        train_centers = compute_train_centers(model, train_loader, num_classes, device)
        print("Train Centers computed successfully.")

    # 4. Load Validation Data + Filenames
    print("\nLoading Validation Data...")
    X_val, y_val, val_filenames = [], [],[]
    load_features_and_filenames(VAL_BASE_DIR, X_val, y_val, val_filenames)
    load_features_and_filenames(VAL_NOVEL_DIR, X_val, y_val, val_filenames)

    X_val = np.array(X_val)
    y_val_encoded = le.transform(np.array(y_val))
    X_val_scaled = scaler.transform(X_val)
    
    print(f"Loaded {X_val.shape[0]} validation samples.")

    # 5. Evaluate and Get Top-3 Scores
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    
    with torch.no_grad():
        logits, hidden_feats = model(X_val_tensor)
        
        if DISTANCE_METRIC in['l2', 'cosine']:
            scores = compute_prototype_scores(hidden_feats, train_centers, metric=DISTANCE_METRIC)
        elif DISTANCE_METRIC == 'logits':
            scores = logits

        # Extract top 3 predicted class indices for each sample
        # shape: (N, 3), where each row has the class indices sorted by highest score
        _, top3_preds = torch.topk(scores, 3, dim=1)
        top3_preds = top3_preds.cpu().numpy()

    # 6. Categorize into Mutually Exclusive Lists
    top1_list =[]
    top2_list = []
    top3_list = []
    not_in_top3_list =[]

    for i in range(len(y_val_encoded)):
        true_cls_idx = y_val_encoded[i]
        filename = val_filenames[i]
        true_class_name = le.inverse_transform([true_cls_idx])[0]

        # Filter by Target Class if specified (e.g., 'Trailer')
        if TARGET_EVAL_CLASS is not None and true_class_name != TARGET_EVAL_CLASS:
            continue

        # Check where the true class sits in the predicted rankings
        # Logic naturally makes them mutually exclusive (elif)
        if true_cls_idx == top3_preds[i, 0]:
            top1_list.append(filename)
        elif true_cls_idx == top3_preds[i, 1]:
            top2_list.append(filename)
        elif true_cls_idx == top3_preds[i, 2]:
            top3_list.append(filename)
        else:
            not_in_top3_list.append(filename)

    # 7. Write to Output TXT Files
    def write_list_to_file(file_name, data_list):
        path = os.path.join(OUTPUT_DIR, file_name)
        with open(path, 'w') as f:
            for item in data_list:
                f.write(f"{item}\n")

    prefix = f"{TARGET_EVAL_CLASS}_" if TARGET_EVAL_CLASS else "ALL_CLASSES_"
    
    write_list_to_file(f"{prefix}TOP1_List.txt", top1_list)
    write_list_to_file(f"{prefix}TOP2_List.txt", top2_list)
    write_list_to_file(f"{prefix}TOP3_List.txt", top3_list)

    # 8. Generate Statistics File
    stats_path = os.path.join(OUTPUT_DIR, f"{prefix}Statistics.txt")
    total_tracked = len(top1_list) + len(top2_list) + len(top3_list) + len(not_in_top3_list)
    
    with open(stats_path, 'w') as f:
        f.write("="*40 + "\n")
        f.write(f"TOP-K EVALUATION STATISTICS\n")
        f.write(f"Target Class: {TARGET_EVAL_CLASS if TARGET_EVAL_CLASS else 'ALL CLASSES'}\n")
        f.write(f"Prediction Mode: {DISTANCE_METRIC.upper()}\n")
        f.write("="*40 + "\n\n")
        f.write(f"Total Ground Truth Samples Evaluated: {total_tracked}\n\n")
        f.write(f"Samples ranked as TOP-1: {len(top1_list)}\n")
        f.write(f"Samples ranked as TOP-2: {len(top2_list)}\n")
        f.write(f"Samples ranked as TOP-3: {len(top3_list)}\n")
        f.write(f"Samples NOT in Top-3 : {len(not_in_top3_list)}\n\n")
        
        f.write("-" * 40 + "\n")
        if total_tracked > 0:
            f.write(f"Top-1 Accuracy for {TARGET_EVAL_CLASS}: {(len(top1_list) / total_tracked) * 100:.2f}%\n")
            f.write(f"Top-2 Accuracy for {TARGET_EVAL_CLASS}: {((len(top1_list) + len(top2_list)) / total_tracked) * 100:.2f}%\n")
            f.write(f"Top-3 Accuracy for {TARGET_EVAL_CLASS}: {((len(top1_list) + len(top2_list) + len(top3_list)) / total_tracked) * 100:.2f}%\n")

    print(f"\nEvaluation Complete! Results saved to '{OUTPUT_DIR}'")
    print(f"TOP-1 Hits: {len(top1_list)}")
    print(f"TOP-2 Hits: {len(top2_list)}")
    print(f"TOP-3 Hits: {len(top3_list)}")
    print(f"Not in TOP-3: {len(not_in_top3_list)}")

if __name__ == "__main__":
    main()