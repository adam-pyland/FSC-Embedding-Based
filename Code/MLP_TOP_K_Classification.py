import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# Global Configuration
# ==========================================
USE_TOP_K_METRICS = True
TOP_K_VALUE = 3
CUSTOM_METRIC_TYPE = 'combined'

# Select your evaluation metric: 'l2', 'cosine', or 'logits'
DISTANCE_METRIC = 'cosine'

# Set the target class to track (e.g., 'Trailer'). 
TARGET_EVAL_CLASS = 'ExtremelyLongHeavyDutyTraileronly'

Dataset_Name = 'Lavyanut'

SHOTS = 20

# Path to the directory where the MLP model, scaler, and label encoder are saved
LOSS_COMBINATION = 'focal_center'
SAVED_MODEL_DIR = f"models_Generalize/{Dataset_Name}/{SHOTS}_shots/{TARGET_EVAL_CLASS}/MLP-Pytorch-Few-Shots-{LOSS_COMBINATION}-Loss-TOP{TOP_K_VALUE if USE_TOP_K_METRICS else 1}-{DISTANCE_METRIC.upper()}-{'Distance' if DISTANCE_METRIC != 'logits' else 'Logits'}-F-SCORE-{CUSTOM_METRIC_TYPE}-Based"

# Output directory for the text files
OUTPUT_DIR = f"Outputs_Generalize/{Dataset_Name}/{SHOTS}_shots/{TARGET_EVAL_CLASS}/TopK_Evaluation_Lists-{LOSS_COMBINATION}-Loss-{DISTANCE_METRIC}-{'Distance' if DISTANCE_METRIC == 'logits' else 'Logits'}-Based"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALL_CLASSES =[
'Bulldozers',
'CementMixerTrucks',
'ExtremelyLongHeavyDutyTraileronly',
'Forklifts',
'HeavyDuty',
'LongHeavyDuty',
'MediumSmall',
'MediumStandard',
'Other',
'Small',
'TruckTractor'
]

WORK_PLACE = 'matrix' 
data_path = r'C:\Adams\FSOD\Data\Lavyanut\Lavyanut_old' if WORK_PLACE == 'yehud' else '/home/adamm/Documents/FSOD/Data/Lavyanut_partial/'
if WORK_PLACE == 'WSL':
    data_path = '/mnt/c/Adams/FSOD/Data/Lavyanut/Lavyanut'

SAFE_CLASS_NAMES =[cls.replace(" ", "_") for cls in ALL_CLASSES]

# Directories
TRAIN_BASE_DIR  = f'{data_path}/Obj_Embs/train/base_class/'
VAL_BASE_DIR  = f'{data_path}/Obj_Embs/test/base_class/'

if TARGET_EVAL_CLASS == 'ExtremelyLongHeavyDutyTraileronly':
    ALL_CLASSES.remove('Forklifts')
    TRAIN_NOVEL_DIR = f'{data_path}/Obj_Embs/train/trailer_{SHOTS}_shots/'
    VAL_NOVEL_DIR   =f'{data_path}/Obj_Embs/test/novel_class_trailer_{SHOTS}_shots/'
elif TARGET_EVAL_CLASS == 'Forklifts':
    ALL_CLASSES.remove('ExtremelyLongHeavyDutyTraileronly')
    TRAIN_NOVEL_DIR = f'{data_path}/Obj_Embs/train/forklifts_{SHOTS}_shots/'
    VAL_NOVEL_DIR   = f'{data_path}/Obj_Embs/test/novel_class_forklifts_{SHOTS}_shots/'
else:
    raise ValueError("Unknown target class for directories!")

# ==========================================
# 1. PyTorch Model Definition
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
# 2. Evaluation Helpers
# ==========================================

def compute_train_centers(model, dataloader, num_classes, device):
    model.eval()
    centers = torch.zeros(num_classes, 128).to(device)
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
    if metric.lower() == 'cosine':
        h2_norm = F.normalize(h2, p=2, dim=1)
        centers_norm = F.normalize(centers, p=2, dim=1)
        scores = torch.mm(h2_norm, centers_norm.t()) 
    elif metric.lower() == 'l2':
        dist = torch.cdist(h2, centers, p=2.0)
        scores = -dist 
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

def get_image_name(filename, safe_classes):
    for cls in safe_classes:
        if f"_{cls}_" in filename:
            return filename.split(f"_{cls}_")[0]
    return filename.rsplit('_', 2)[0]

# --- PLOTTING FUNCTION 1: CONTINUOUS CURVES ---
def plot_ranking_evaluation(df_sorted, target_class, output_dir):
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Global Ranking Curves for: {target_class}\nSorted by Model Confidence", fontsize=16, fontweight='bold')

    K_vals = df_sorted['K']

    axs[0, 0].plot(K_vals, df_sorted['Cum_TP_obj'], label='True Positive Objects', color='green', linewidth=2)
    axs[0, 0].plot(K_vals, df_sorted['Cum_FP_obj'], label='False Positive Objects', color='red', linewidth=2)
    axs[0, 0].set_title("1. Cumulative Objects (Absolute Count)")
    axs[0, 0].set_xlabel("Rank (Top K Items)")
    axs[0, 0].set_ylabel("Total Objects")
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)

    axs[0, 1].plot(K_vals, df_sorted['Precision_obj_at_K'] * 100, label='Precision @ K', color='blue', linewidth=2)
    axs[0, 1].set_title("2. Object Precision at Top K")
    axs[0, 1].set_xlabel("Rank (Top K Items)")
    axs[0, 1].set_ylabel("Precision (%)")
    axs[0, 1].set_ylim(0, 105)
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)

    axs[1, 0].plot(K_vals, df_sorted['Cum_TP_img'], label='True Positive Images', color='green', linewidth=2)
    axs[1, 0].plot(K_vals, df_sorted['Cum_FP_img'], label='False Positive Images (No TPs)', color='red', linewidth=2)
    axs[1, 0].set_title("3. Cumulative Unique Images (Absolute Count)")
    axs[1, 0].set_xlabel("Rank (Top K Items)")
    axs[1, 0].set_ylabel("Total Unique Images")
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)

    axs[1, 1].plot(K_vals, df_sorted['Precision_img_at_K'] * 100, label='Precision @ K', color='purple', linewidth=2)
    axs[1, 1].set_title("4. Image Precision at Top K")
    axs[1, 1].set_xlabel("Rank (Top K Items)")
    axs[1, 1].set_ylabel("Precision (%)")
    axs[1, 1].set_ylim(0, 105)
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plot_path = os.path.join(output_dir, f"{target_class}_Ranking_Curves.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved Continuous Ranking curves to: {plot_path}")

# --- NEW PLOTTING FUNCTION 2: GROUPED BAR CHARTS FOR BOSS ---
def plot_sampled_bar_charts(df_sorted, target_class, output_dir):
    total_k = len(df_sorted)
    
    # Define sensible business milestones 
    potential_milestones =[10, 25, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500]
    
    # Filter milestones strictly <= total items we actually evaluated
    selected_ks = [k for k in potential_milestones if k <= total_k]
    if total_k not in selected_ks:
        selected_ks.append(total_k)
        
    # Ensure max 10 bars so it doesn't get cluttered (take the most relevant evenly)
    if len(selected_ks) > 10:
        # Keep 1st, 2nd, and 8 evenly spaced from the rest up to max
        idx = np.round(np.linspace(0, len(selected_ks)-1, 10)).astype(int)
        selected_ks = [selected_ks[i] for i in sorted(list(set(idx)))]
    
    # Extract data just for these K values
    df_sampled = df_sorted[df_sorted['K'].isin(selected_ks)].copy()
    
    x = np.arange(len(selected_ks))  # the label locations
    width = 0.35                     # the width of the bars
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Sampled Top-K Performance: {target_class}", fontsize=16, fontweight='bold')
    
    labels =[f"Top {k}" for k in df_sampled['K']]
    
    # --- 1. Objects TP/FP Grouped Bar ---
    axs[0, 0].bar(x - width/2, df_sampled['Cum_TP_obj'], width, label='TP Objects', color='forestgreen')
    axs[0, 0].bar(x + width/2, df_sampled['Cum_FP_obj'], width, label='FP Objects', color='lightcoral')
    axs[0, 0].set_title("1. Cumulative Objects (TP vs FP)")
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(labels, rotation=45)
    axs[0, 0].set_ylabel("Total Object Count")
    axs[0, 0].legend()
    # Add actual numbers on top of bars
    for i, (tp, fp) in enumerate(zip(df_sampled['Cum_TP_obj'], df_sampled['Cum_FP_obj'])):
        axs[0, 0].text(i - width/2, tp + (tp*0.02), str(int(tp)), ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkgreen')
        axs[0, 0].text(i + width/2, fp + (fp*0.02), str(int(fp)), ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkred')

    # --- 2. Object Precision Bar ---
    prec_obj = df_sampled['Precision_obj_at_K'] * 100
    axs[0, 1].bar(x, prec_obj, width=0.5, label='Precision (%)', color='royalblue')
    axs[0, 1].set_title("2. Object Precision at Rank K")
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(labels, rotation=45)
    axs[0, 1].set_ylabel("Precision (%)")
    axs[0, 1].set_ylim(0, 115) 
    for i, p in enumerate(prec_obj):
        axs[0, 1].text(i, p + 2, f"{p:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

    # --- 3. Images TP/FP Grouped Bar ---
    axs[1, 0].bar(x - width/2, df_sampled['Cum_TP_img'], width, label='TP Images', color='forestgreen')
    axs[1, 0].bar(x + width/2, df_sampled['Cum_FP_img'], width, label='FP Images (No TP)', color='lightcoral')
    axs[1, 0].set_title("3. Cumulative Unique Images (TP vs FP)")
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(labels, rotation=45)
    axs[1, 0].set_ylabel("Total Unique Images")
    axs[1, 0].legend()
    for i, (tp, fp) in enumerate(zip(df_sampled['Cum_TP_img'], df_sampled['Cum_FP_img'])):
        axs[1, 0].text(i - width/2, tp + (tp*0.02), str(int(tp)), ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkgreen')
        axs[1, 0].text(i + width/2, fp + (fp*0.02), str(int(fp)), ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkred')

    # --- 4. Images Precision Bar ---
    prec_img = df_sampled['Precision_img_at_K'] * 100
    axs[1, 1].bar(x, prec_img, width=0.5, label='Precision (%)', color='purple')
    axs[1, 1].set_title("4. Image Precision at Rank K")
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(labels, rotation=45)
    axs[1, 1].set_ylabel("Precision (%)")
    axs[1, 1].set_ylim(0, 115)
    for i, p in enumerate(prec_img):
        axs[1, 1].text(i, p + 2, f"{p:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plot_path = os.path.join(output_dir, f"{target_class}_Sampled_Bar_Charts.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved Sampled Bar Charts to: {plot_path}")

# ==========================================
# 3. Main Evaluation Pipeline
# ==========================================

def main():
    start_time = time.time()
    
    if TARGET_EVAL_CLASS is None:
        raise ValueError("TARGET_EVAL_CLASS must be specified.")

    print("="*50)
    print("Starting Prediction-Based Top-K Evaluation")
    print(f"Prediction Mode: {DISTANCE_METRIC.upper()}")
    print(f"Target Class to Track: {TARGET_EVAL_CLASS}")
    print("="*50)

    # 1. Load Scaler and Label Encoder
    scaler_file = os.path.join(SAVED_MODEL_DIR, 'saved_scaler.joblib')
    le_file = os.path.join(SAVED_MODEL_DIR, 'saved_le.joblib')
    best_model_file = os.path.join(SAVED_MODEL_DIR, 'best_mlp.pth')

    if not (os.path.exists(scaler_file) and os.path.exists(le_file) and os.path.exists(best_model_file)):
        print(f"Error: Could not find model, scaler, or LabelEncoder in {SAVED_MODEL_DIR}")
        return

    scaler = load(scaler_file)
    le = load(le_file)
    num_classes = len(le.classes_)

    if TARGET_EVAL_CLASS not in le.classes_:
        print(f"Error: '{TARGET_EVAL_CLASS}' not found in loaded Label Encoder classes.")
        return

    target_class_idx = le.transform([TARGET_EVAL_CLASS])[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Setup Model
    input_dim = getattr(scaler, 'n_features_in_', 1024)
    model = MLP_PyTorch(input_dim=input_dim, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(best_model_file, map_location=device))
    model.eval()

    # 3. Compute Training Centers (if required)
    train_centers = None
    if DISTANCE_METRIC in['l2', 'cosine']:
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

    # 4. Load Validation Data + Filenames
    print("\nLoading Validation Data...")
    X_val, y_val, val_filenames = [], [],[]
    load_features_and_filenames(VAL_BASE_DIR, X_val, y_val, val_filenames)
    load_features_and_filenames(VAL_NOVEL_DIR, X_val, y_val, val_filenames)

    X_val = np.array(X_val)
    y_val_encoded = le.transform(np.array(y_val))
    X_val_scaled = scaler.transform(X_val)

    # 5. Evaluate Scores
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    
    with torch.no_grad():
        logits, hidden_feats = model(X_val_tensor)
        
        if DISTANCE_METRIC in ['l2', 'cosine']:
            scores = compute_prototype_scores(hidden_feats, train_centers, metric=DISTANCE_METRIC)
        elif DISTANCE_METRIC == 'logits':
            scores = logits

        _, top3_preds = torch.topk(scores, 3, dim=1)
        top3_preds = top3_preds.cpu().numpy()
        
        # Extract absolute scores specifically for the TARGET CLASS
        target_class_scores = scores[:, target_class_idx].cpu().numpy()

    # ==========================================
    # --- PANDAS RANKING LOGIC FOR PLOTS ---
    # ==========================================
    print("\nCalculating Global Rankings (Boss's request)...")
    
    df = pd.DataFrame({
        'filename': val_filenames,
        'is_target': (y_val_encoded == target_class_idx),
        'score': target_class_scores
    })
    
    df['image_name'] = df['filename'].apply(lambda x: get_image_name(x, SAFE_CLASS_NAMES))
    df_sorted = df.sort_values(by='score', ascending=False).reset_index(drop=True)
    df_sorted['K'] = df_sorted.index + 1
    
    # 1. Object-Level Calculations
    df_sorted['TP_obj'] = df_sorted['is_target'].astype(int)
    df_sorted['FP_obj'] = (~df_sorted['is_target']).astype(int)
    
    df_sorted['Cum_TP_obj'] = df_sorted['TP_obj'].cumsum()
    df_sorted['Cum_FP_obj'] = df_sorted['FP_obj'].cumsum()
    df_sorted['Precision_obj_at_K'] = df_sorted['Cum_TP_obj'] / df_sorted['K']
    
    # 2. Image-Level Calculations
    cum_tp_imgs =[]
    cum_fp_imgs =[]
    seen_tp_imgs = set()
    seen_fp_imgs = set()
    
    for idx, row in df_sorted.iterrows():
        img = row['image_name']
        if row['is_target']:
            seen_tp_imgs.add(img)
        else:
            seen_fp_imgs.add(img)
            
        pure_fp_imgs = seen_fp_imgs - seen_tp_imgs
        cum_tp_imgs.append(len(seen_tp_imgs))
        cum_fp_imgs.append(len(pure_fp_imgs))
        
    df_sorted['Cum_TP_img'] = cum_tp_imgs
    df_sorted['Cum_FP_img'] = cum_fp_imgs
    total_imgs_at_K = df_sorted['Cum_TP_img'] + df_sorted['Cum_FP_img']
    df_sorted['Precision_img_at_K'] = df_sorted['Cum_TP_img'] / total_imgs_at_K.replace(0, 1)

    # 6. Categorize into Mutually Exclusive Lists (Original Code)
    top1_list =[]
    top2_list = []
    top3_list =[]
    not_in_top3_list =[]

    top1_objs = {'TP':[], 'FP':[]}
    top2_objs = {'TP':[], 'FP':[]}
    top3_objs = {'TP':[], 'FP':[]}
    not3_objs = {'TP':[], 'FP':[]}

    for i in range(len(y_val_encoded)):
        filename = val_filenames[i]
        is_real = (y_val_encoded[i] == target_class_idx)

        if target_class_idx == top3_preds[i, 0]:
            top1_list.append(filename)
            if is_real: top1_objs['TP'].append(filename)
            else: top1_objs['FP'].append(filename)
            
        elif target_class_idx == top3_preds[i, 1]:
            top2_list.append(filename)
            if is_real: top2_objs['TP'].append(filename)
            else: top2_objs['FP'].append(filename)
            
        elif target_class_idx == top3_preds[i, 2]:
            top3_list.append(filename)
            if is_real: top3_objs['TP'].append(filename)
            else: top3_objs['FP'].append(filename)
            
        else:
            not_in_top3_list.append(filename)
            if is_real: not3_objs['TP'].append(filename)
            else: not3_objs['FP'].append(filename)

    # 7. Write to Output TXT Files
    def write_list_to_file(file_name, data_list):
        path = os.path.join(OUTPUT_DIR, file_name)
        with open(path, 'w') as f:
            for item in data_list:
                f.write(f"{item}\n")

    prefix = f"{TARGET_EVAL_CLASS}_"
    write_list_to_file(f"{prefix}TOP1_List.txt", top1_list)
    write_list_to_file(f"{prefix}TOP2_List.txt", top2_list)
    write_list_to_file(f"{prefix}TOP3_List.txt", top3_list)
    
    end_time = time.time()
    measured_time = end_time - start_time
    
    # 8. Generate Statistics File (Original logic untouched)
    stats_path = os.path.join(OUTPUT_DIR, f"{prefix}Statistics.txt")
    total_samples = len(y_val_encoded)
    
    with open(stats_path, 'w') as f:
        f.write("="*40 + "\n")
        f.write(f"PREDICTION-BASED TOP-K STATISTICS\n")
        f.write(f"Target Class Evaluated: {TARGET_EVAL_CLASS}\n")
        f.write("="*40 + "\n\n")

        def write_category_stats(cat_name, total_count, real_count):
            acc = (real_count / total_count * 100) if total_count > 0 else 0.0
            f.write(f"--- {cat_name} ---\n")
            f.write(f"Total objects scored into this category: {total_count}\n")
            f.write(f"Objects actually belonging to ground truth '{TARGET_EVAL_CLASS}': {real_count}\n")
            f.write(f"Accuracy (Real / Total) within category: {acc:.2f}%\n\n")

        write_category_stats("TOP-1 Category", len(top1_list), len(top1_objs['TP']))
        write_category_stats("TOP-2 Category", len(top2_list), len(top2_objs['TP']))
        write_category_stats("TOP-3 Category", len(top3_list), len(top3_objs['TP']))
        write_category_stats("Not in TOP-3 Category", len(not_in_top3_list), len(not3_objs['TP']))

    print(f"\nEvaluation Complete! Results saved to '{OUTPUT_DIR}'")

    # --- Generate Visualizations requested by the boss ---
    plot_ranking_evaluation(df_sorted, TARGET_EVAL_CLASS, OUTPUT_DIR)
    plot_sampled_bar_charts(df_sorted, TARGET_EVAL_CLASS, OUTPUT_DIR)
    
    # Save a CSV of the data we plotted 
    csv_path = os.path.join(OUTPUT_DIR, f"{prefix}Ranking_Data.csv")
    df_sorted.to_csv(csv_path, index=False)
    
    print("-" * 50)
    print(f"Measured block execution time: {measured_time:.4f} seconds")
    print("-" * 50)
    
if __name__ == "__main__":
    main()