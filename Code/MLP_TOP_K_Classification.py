import os
import glob
import time
import numpy as np
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
# Must be set to evaluate prediction-based rankings.
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

WORK_PLACE = 'matrix' # The place where I am working in: 'yehud' or 'matrix'. Or WSL if decided to work on WSL on windows in Yehud.
data_path = r'C:\Adams\FSOD\Data\Lavyanut\Lavyanut_old' if WORK_PLACE == 'yehud' else '/home/adamm/Documents/FSOD/Data/Lavyanut_partial/'
if WORK_PLACE == 'WSL':
    data_path = '/mnt/c/Adams/FSOD/Data/Lavyanut/Lavyanut'


SAFE_CLASS_NAMES =[cls.replace(" ", "_") for cls in ALL_CLASSES]

# Directories (Change these if they differ from your environment)
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
    """
    Replicates the scikit-learn MLP architecture:
    Input -> Linear(512) -> ReLU -> Linear(256) -> ReLU -> Linear(num_classes)
    """
    def __init__(self, input_dim, num_classes):
        super(MLP_PyTorch, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256) # Reduced size
        self.dropout1 = nn.Dropout(p=0.4)    # Added dropout
        self.fc2 = nn.Linear(256, 128)       # Reduced size
        self.dropout2 = nn.Dropout(p=0.4)    # Added dropout
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = self.dropout1(h1)
        h2 = F.relu(self.fc2(h1))
        h2_drop = self.dropout2(h2)
        logits = self.fc3(h2_drop)
        return logits, h2 # Still return pure h2 for metric distances
    
# ==========================================
# 2. Evaluation Helpers
# ==========================================

def compute_train_centers(model, dataloader, num_classes, device):
    """Calculates prototype centers required for 'cosine' or 'l2' metrics."""
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

def get_image_name(filename, safe_classes):
    """Extracts the base image name from the full object npy filename."""
    for cls in safe_classes:
        if f"_{cls}_" in filename:
            return filename.split(f"_{cls}_")[0]
    # Fallback method just in case
    return filename.rsplit('_', 2)[0]

def plot_statistics(obj_tp, obj_fp, img_tp, img_fp,
                    g_tp_obj, g_fp_obj, g_tp_img, g_fp_img, 
                    target_class, output_dir):
    categories =['TOP-1', 'TOP-2', 'TOP-3', 'Not in TOP-3']
    
    obj_tp = np.array(obj_tp)
    obj_fp = np.array(obj_fp)
    img_tp = np.array(img_tp)
    img_fp = np.array(img_fp)
    
    obj_totals = obj_tp + obj_fp
    img_totals = img_tp + img_fp
    
    # Calculate Percentages for 100% Stacked Bars (Ratio)
    obj_tp_pct = np.divide(obj_tp, obj_totals, out=np.zeros_like(obj_tp, dtype=float), where=obj_totals!=0) * 100
    obj_fp_pct = np.divide(obj_fp, obj_totals, out=np.zeros_like(obj_fp, dtype=float), where=obj_totals!=0) * 100
    
    img_tp_pct = np.divide(img_tp, img_totals, out=np.zeros_like(img_tp, dtype=float), where=img_totals!=0) * 100
    img_fp_pct = np.divide(img_fp, img_totals, out=np.zeros_like(img_fp, dtype=float), where=img_totals!=0) * 100

    fig, axs = plt.subplots(1, 3, figsize=(18, 6.5))
    fig.suptitle(f'Prediction-Based Top-K Evaluation: {target_class}', fontsize=16, fontweight='bold')

    # --- Plot 1: Stacked Bar Chart (Object Ratios) ---
    axs[0].bar(categories, obj_tp_pct, label='TP Objects', color='forestgreen')
    axs[0].bar(categories, obj_fp_pct, bottom=obj_tp_pct, label='FP Objects', color='lightcoral')
    axs[0].set_title('Object Ratio (%)\nTP vs FP')
    axs[0].set_ylabel('Percentage')
    axs[0].set_ylim(0, 100)
    axs[0].legend(loc='lower left')
    for i in range(len(categories)):
        if obj_totals[i] > 0:
            if obj_tp_pct[i] >= 5:
                axs[0].text(i, obj_tp_pct[i] / 2, f"{obj_tp_pct[i]:.1f}%", ha='center', color='white', fontweight='bold')
            if obj_fp_pct[i] >= 5:
                axs[0].text(i, obj_tp_pct[i] + (obj_fp_pct[i] / 2), f"{obj_fp_pct[i]:.1f}%", ha='center', color='black', fontweight='bold')

    # --- Plot 2: Stacked Bar Chart (Image Ratios) ---
    axs[1].bar(categories, img_tp_pct, label='TP Images', color='forestgreen')
    axs[1].bar(categories, img_fp_pct, bottom=img_tp_pct, label='FP Images', color='lightcoral')
    axs[1].set_title('Image Ratio (%)\nTP vs FP')
    axs[1].set_ylabel('Percentage')
    axs[1].set_ylim(0, 100)
    axs[1].legend(loc='lower left')
    for i in range(len(categories)):
        if img_totals[i] > 0:
            if img_tp_pct[i] >= 5:
                axs[1].text(i, img_tp_pct[i] / 2, f"{img_tp_pct[i]:.1f}%", ha='center', color='white', fontweight='bold')
            if img_fp_pct[i] >= 5:
                axs[1].text(i, img_tp_pct[i] + (img_fp_pct[i] / 2), f"{img_fp_pct[i]:.1f}%", ha='center', color='black', fontweight='bold')

    # --- Plot 3: Pie Chart (Test Novel Class Distribution) ---
    labels =[cat for cat, r in zip(categories, obj_tp) if r > 0]
    sizes = [r for r in obj_tp if r > 0]
    colors =['#ff9999','#66b3ff','#99ff99','#ffcc99'][:len(sizes)]
    
    if sum(sizes) > 0:
        axs[2].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, explode=[0.05]*len(sizes))
        axs[2].set_title('Test Novel Class Distribution to categories')
    else:
        axs[2].text(0.5, 0.5, "No Real Targets Found", ha='center', va='center')
        axs[2].set_title('Test Novel Class Distribution to categories')
        axs[2].axis('off')

    # --- Calculate and Display Global Percentages at the Bottom ---
    g_obj_total = g_tp_obj + g_fp_obj
    g_img_total = g_tp_img + g_fp_img
    
    g_tp_obj_pct = (g_tp_obj / g_obj_total * 100) if g_obj_total > 0 else 0.0
    g_fp_obj_pct = (g_fp_obj / g_obj_total * 100) if g_obj_total > 0 else 0.0
    
    g_tp_img_pct = (g_tp_img / g_img_total * 100) if g_img_total > 0 else 0.0
    g_fp_img_pct = (g_fp_img / g_img_total * 100) if g_img_total > 0 else 0.0

    # UPDATE: Changed "Overall" to "TOP-3" to accurately reflect the data
    stats_text = (
        f"TOP-3 Objects - TP (Target Class): {g_tp_obj_pct:.1f}% | FP (Other Classes): {g_fp_obj_pct:.1f}%\n"
        f"TOP-3 Images - TP (Contains Target): {g_tp_img_pct:.1f}% | FP (No Target Present): {g_fp_img_pct:.1f}%"
    )
    
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=13, 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))

    plt.tight_layout(rect=[0, 0.08, 1, 0.95]) # Leaves space for the bottom text
    
    # Save the plot
    plot_path = os.path.join(output_dir, f"{target_class}_Evaluation_Plots.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved evaluation plots to: {plot_path}")

# ==========================================
# 3. Main Evaluation Pipeline
# ==========================================

def main():
    
    # --- BEGINNING OF TIMED BLOCK ---
    start_time = time.time()
    
    if TARGET_EVAL_CLASS is None:
        raise ValueError("TARGET_EVAL_CLASS must be specified to perform prediction-based category ranking.")

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

    # Ensure TARGET_EVAL_CLASS is in the label encoder
    if TARGET_EVAL_CLASS not in le.classes_:
        print(f"Error: '{TARGET_EVAL_CLASS}' not found in loaded Label Encoder classes.")
        return

    target_class_idx = le.transform([TARGET_EVAL_CLASS])[0]

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

    # 6. Categorize into Mutually Exclusive Lists based on PREDICTED rank of the target class
    top1_list = []
    top2_list = []
    top3_list =[]
    not_in_top3_list =[]

    # Store objects categorized into True Positives (TP) and False Positives (FP)
    top1_objs = {'TP': [], 'FP':[]}
    top2_objs = {'TP': [], 'FP': []}
    top3_objs = {'TP':[], 'FP': []}
    not3_objs = {'TP': [], 'FP':[]}

    for i in range(len(y_val_encoded)):
        filename = val_filenames[i]
        is_real = (y_val_encoded[i] == target_class_idx)

        # Check where the TARGET_EVAL_CLASS sits in the predicted rankings for this sample
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
    
    # --- END OF TIMED BLOCK ---
    end_time = time.time()
    measured_time = end_time - start_time
    
    
    # --- Calculate Image and Object TP/FP Metrics for plotting ---
    obj_tp = []
    obj_fp =[]
    img_tp = []
    img_fp =[]
    
    global_tp_objs = 0
    global_fp_objs = 0
    global_tp_imgs_set = set()
    global_fp_imgs_set = set()

    # UPDATE: Wrap with enumerate to easily track which category we are processing
    for idx, cat_dict in enumerate([top1_objs, top2_objs, top3_objs, not3_objs]):
        tps = cat_dict['TP']
        fps = cat_dict['FP']

        o_tp = len(tps)
        o_fp = len(fps)
        obj_tp.append(o_tp)
        obj_fp.append(o_fp)

        tp_img_names = set(get_image_name(f, SAFE_CLASS_NAMES) for f in tps)
        fp_img_names = set(get_image_name(f, SAFE_CLASS_NAMES) for f in fps)

        # "if an image has the TP objects it will be considered as TP image" -> remove overlap from FP
        pure_fp_img_names = fp_img_names - tp_img_names

        img_tp.append(len(tp_img_names))
        img_fp.append(len(pure_fp_img_names))

        # UPDATE: Only add to the global pool if it's TOP-1, TOP-2, or TOP-3 (Indices 0, 1, 2)
        if idx < 3:
            global_tp_objs += o_tp
            global_fp_objs += o_fp
            global_tp_imgs_set.update(tp_img_names)
            global_fp_imgs_set.update(fp_img_names)
    # Resolve global images the exact same way
    global_pure_fp_imgs_set = global_fp_imgs_set - global_tp_imgs_set
    global_tp_img_count = len(global_tp_imgs_set)
    global_fp_img_count = len(global_pure_fp_imgs_set)

    # 8. Generate Statistics File
    stats_path = os.path.join(OUTPUT_DIR, f"{prefix}Statistics.txt")
    total_samples = len(y_val_encoded)
    
    with open(stats_path, 'w') as f:
        f.write("="*40 + "\n")
        f.write(f"PREDICTION-BASED TOP-K STATISTICS\n")
        f.write(f"Target Class Evaluated: {TARGET_EVAL_CLASS}\n")
        f.write(f"Prediction Mode: {DISTANCE_METRIC.upper()}\n")
        f.write("="*40 + "\n\n")
        f.write(f"Total Validation Samples Evaluated: {total_samples}\n\n")

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
    print(f"Categorized into TOP-1: {len(top1_list)} (Actually real: {len(top1_objs['TP'])})")
    print(f"Categorized into TOP-2: {len(top2_list)} (Actually real: {len(top2_objs['TP'])})")
    print(f"Categorized into TOP-3: {len(top3_list)} (Actually real: {len(top3_objs['TP'])})")
    print(f"Not in TOP-3: {len(not_in_top3_list)} (Actually real: {len(not3_objs['TP'])})")

    # Generate Visualizations
    plot_statistics(
        obj_tp, obj_fp, 
        img_tp, img_fp, 
        global_tp_objs, global_fp_objs, 
        global_tp_img_count, global_fp_img_count,
        TARGET_EVAL_CLASS, OUTPUT_DIR
    )
    
    # Print measured execution time as requested
    print("-" * 50)
    print(f"Measured block execution time: {measured_time:.4f} seconds")
    print("-" * 50)
    
if __name__ == "__main__":
    main()