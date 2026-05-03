import os
import sys
import shutil
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import optuna

from sklearn.preprocessing import StandardScaler, LabelEncoder, Normalizer
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE, MDS
from sklearn.metrics import classification_report, f1_score, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import json

# === NEW: PyTorch Metric Learning for Triplet Loss ===
try:
    from pytorch_metric_learning import losses as pml_losses
    from pytorch_metric_learning import miners as pml_miners
    from pytorch_metric_learning import distances as pml_distances
except ImportError:
    print("Error: Please install pytorch-metric-learning to use Triplet Loss functionalities.")
    print("Run: pip install pytorch-metric-learning")
    sys.exit(1)

def seed_everything(seed=42):
    """
    Locks all random seeds so that model initializations, data shuffling, 
    and loss functions behave exactly the same way every time the script runs.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # If you use multi-GPU
    
    # Force deterministic algorithms in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# Global Configuration & Top-K / Distance Flags
# ==========================================

WORK_PLACE = 'yehud' # The place where I am working in: 'yehud' or 'matrix'. Or WSL if decided to work on WSL on windows in Yehud.

data_path = r'C:\Adams\FSOD\Data\Lavyanut\Lavyanut' if WORK_PLACE is 'yehud' else '/home/adamm/Documents/FSOD/Data/Lavyanut'
if WORK_PLACE == 'WSL':
    data_path = '/mnt/c/Adams/FSOD/Data/Lavyanut/Lavyanut'
# Top-K Metrics
USE_TOP_K_METRICS = False
TOP_K_VALUE = 1

# Prediction & Distance Metrics
# Options: 'l2' (Euclidean), 'cosine' (Cosine similarity), or 'logits' (MLP Output Scores)
DISTANCE_METRIC = 'cosine'             

# Loss Function Combinations
# Options: 'focal_center', 'triplet_center', 'focal_triplet_center'
LOSS_COMBINATION = 'focal_center'

CUSTOM_METRIC_TYPE = 'combined' # use 'f1_novel', 'f2_novel' or 'combined' 
SEED = 9

SHOTS = 10

BEST_HYPERPARAMETERS=None

# BEST_HYPERPARAMETERS = {
#     "batch_size": 1024,
#     "gamma": 2.580627947590925,
#     "center_weight": 0.028365290137279026,
#     "lr": 0.004351134823645674,
#     "weight_decay": 0.003054107429563069,
#     "weight_smoothing": 0.5503699400128665,
#     "novel_multiplier": 9.498785088342668
# }


MAX_EPOCHS = 500

# Combined Metric ratios for F1 and F2 scores
F2_NOVEL_RATIO = 0.70; F1_ALL_RATIO = 0.30

ALL_CLASSES = [
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

TARGET_NOVEL_CLASS = 'ExtremelyLongHeavyDutyTraileronly'
VISUALIZATION_SOURCE = 'train'

Dataset_Name = 'Lavyanut'





SAVE_DIR = f"models_Generalized_Windows/{Dataset_Name}/{SHOTS}_shots/{TARGET_NOVEL_CLASS}/MLP-Pytorch-Few-Shots-{LOSS_COMBINATION}-Loss-TOP{TOP_K_VALUE if USE_TOP_K_METRICS else 1}-{DISTANCE_METRIC.upper()}-{'Distance' if DISTANCE_METRIC != 'logits' else 'Logits'}-F-SCORE-{CUSTOM_METRIC_TYPE}-Based"

PLOT_DIR = f"Outputs_Generalized_Windows/{Dataset_Name}/{SHOTS}_shots/{TARGET_NOVEL_CLASS}/MLP-Pytorch-Few-Shots-{LOSS_COMBINATION}-Loss-TOP{TOP_K_VALUE if USE_TOP_K_METRICS else 1}-{DISTANCE_METRIC.upper()}-{'Distance' if DISTANCE_METRIC != 'logits' else 'Logits'}-F-SCORE-{CUSTOM_METRIC_TYPE}-Based"
os.makedirs(PLOT_DIR, exist_ok=True)

TRAIN_BASE_DIR  = f'{data_path}/Obj_Embs/train/base_class/'
VAL_BASE_DIR  = f'{data_path}/Obj_Embs/test/base_class/'

if TARGET_NOVEL_CLASS == 'ExtremelyLongHeavyDutyTraileronly':
    ALL_CLASSES.remove('Forklifts')
    TRAIN_NOVEL_DIR = f'{data_path}/Obj_Embs/train/trailer_{SHOTS}_shots/'
    VAL_NOVEL_DIR   =f'{data_path}/Obj_Embs/test/novel_class_trailer_{SHOTS}_shots/'
elif TARGET_NOVEL_CLASS == 'Forklifts':
    ALL_CLASSES.remove('ExtremelyLongHeavyDutyTraileronly')
    TRAIN_NOVEL_DIR = f'{data_path}/Obj_Embs/train/forklifts_{SHOTS}_shots/'
    VAL_NOVEL_DIR   = f'{data_path}/Obj_Embs/test/novel_class_forklifts_{SHOTS}_shots/'
else:
    raise ValueError("Unknown target class for directories!")


# ==========================================
# 1. PyTorch Model and Center Loss Definitions
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
    

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight # This acts as the Alpha (class weights)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 1. Get raw, unweighted Cross Entropy to extract true probabilities (pt)
        ce_loss_unweighted = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss_unweighted) 
        
        # 2. Calculate the Focal Term
        focal_term = (1 - pt) ** self.gamma
        
        # 3. Apply Alpha (Class Weights) manually
        if self.weight is not None:
            alpha_t = self.weight[targets]
            focal_loss = alpha_t * focal_term * ce_loss_unweighted
        else:
            focal_loss = focal_term * ce_loss_unweighted

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CenterLoss(nn.Module):
    """
    Center Loss helps pull features of the same class towards a learned class center.
    This creates much tighter clusters in the embedding space.
    """
    def __init__(self, num_classes, feat_dim, device=torch.device('cpu')):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss

# ==========================================
# 2. Evaluation and Distance Helpers
# ==========================================

def compute_train_centers(model, dataloader, num_classes, device):
    """
    Passes over the training data to calculate the exact mean vector (prototype) 
    for each class in the 256-D space.
    """
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
    """
    Calculates distance/similarity to prototypes and returns a 'score'
    where a HIGHER score means a closer match. This natively aligns with top-k logic.
    """
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

def compute_train_logit_centers(model, dataloader, num_classes, device):
    model.eval()
    logit_centers = torch.zeros(num_classes, num_classes).to(device)
    counts = torch.zeros(num_classes).to(device)
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            logits, _ = model(features)
            logit_centers.index_add_(0, labels, logits)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float))
    return (logit_centers / counts.clamp(min=1e-8).unsqueeze(1)).cpu().numpy()

def get_predictions(scores, labels, top_k=1, use_top_k=False):
    """
    Returns predictions based on top-1 or top-k logic using provided scores.
    """
    if not use_top_k or top_k <= 1:
        _, preds = torch.max(scores, 1)
        return preds
    
    actual_top_k = min(top_k, scores.size(1))
    _, top_k_preds = torch.topk(scores, actual_top_k, dim=1)
    
    labels_expanded = labels.view(-1, 1)
    correct_in_top_k = (top_k_preds == labels_expanded).any(dim=1)
    
    top_1_preds = top_k_preds[:, 0]
    final_preds = torch.where(correct_in_top_k, labels, top_1_preds)
    
    return final_preds

def evaluate_and_visualize_superclasses(y_test, y_pred, X_viz, y_test_viz, centers_viz=None, le=None, metric_name=''):
    base_subclasses = [
        'Bulldozers', 'CementMixerTrucks', 'HeavyDuty', 
        'LongHeavyDuty', 'MediumSmall', 'MediumStandard', 
        'Other', 'Small', 'TruckTractor'
    ]
    
    def map_to_superclass(labels):
        return np.array(['Novel' if lbl == TARGET_NOVEL_CLASS else 'Base' for lbl in labels])
    
    y_test_super = map_to_superclass(y_test)
    y_pred_super = map_to_superclass(y_pred)
    
    print("\n" + "="*50)
    print("--- BASE vs NOVEL Classification Report ---")
    file_path = os.path.join(PLOT_DIR, 'Base_Novel_Classification_Report.txt')
    report = classification_report(y_test_super, y_pred_super, digits=4, zero_division=0)
    with open(file_path, "w") as f:
        f.write(report)
        print(report)
    print("="*50 + "\n")
    
    y_test_viz_super = map_to_superclass(y_test_viz)
    
    print(f"Plotting Base vs Novel separation ({metric_name})...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 10))
    
    super_palette = {'Base': 'royalblue', 'Novel': 'crimson'}
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1, alpha=0.85)

    # Plot samples
    sns.scatterplot(
        x=X_viz[:, 0], y=X_viz[:, 1],
        hue=y_test_viz_super, palette=super_palette, s=70, alpha=0.5, edgecolor=None
    )

    # Plot Superclass Centers
    unique_superclasses = ['Base', 'Novel']
    for cls in unique_superclasses:
        if cls in y_test_viz_super:
            # Calculate the center of the superclass stars
            if centers_viz is not None and le is not None:
                # Find which indices in the label encoder belong to this superclass
                superclass_classes = [c for c in le.classes_ if map_to_superclass([c])[0] == cls]
                superclass_indices = le.transform(superclass_classes).astype(int)
                center_x = np.mean(centers_viz[superclass_indices, 0])
                center_y = np.mean(centers_viz[superclass_indices, 1])
            else:
                cls_points = X_viz[y_test_viz_super == cls]
                center_x, center_y = np.mean(cls_points[:, 0]), np.mean(cls_points[:, 1])
                
            plt.scatter(center_x, center_y, marker='*', s=1200, color=super_palette[cls], 
                        edgecolor='black', linewidth=2, zorder=10, label=f"{cls} Prototype")
            plt.annotate(f"{cls} Train Center", (center_x, center_y), xytext=(15, 15), 
                         textcoords='offset points', fontsize=12, fontweight='bold', 
                         bbox=bbox_props, zorder=11)

    plt.title(f'Base vs Novel Separation Map ({metric_name})\nDistance represents Model Logic', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(title='Superclass', loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'Base_vs_Novel_Separation_{metric_name}.png'), dpi=300)
    print(f"Done! Saved superclass plot as 'Base_vs_Novel_Separation_{metric_name}.png'")

# ==========================================
# 3. Main Script Pipeline
# ==========================================

def main():
    seed_everything(SEED)
    # 0. Save a copy of the executing script to ensure reproducibility
    try:
        current_script = os.path.abspath(__file__)
        script_backup_path = os.path.join(PLOT_DIR, 'code_that_made_it.py')
        shutil.copy(current_script, script_backup_path)
        print(f"-> Successfully saved copy of executing script to {script_backup_path}\n")
    except NameError:
        print("-> Could not save script (likely running in a Jupyter/interactive environment).")

    print("="*50)
    print(f"Metrics Mode: Top-{TOP_K_VALUE} Accuracy Enabled: {USE_TOP_K_METRICS}")
    print(f"Prediction Mode: {DISTANCE_METRIC.upper()}")
    print(f"Loss Combination : {LOSS_COMBINATION.upper()}")
    print("="*50)

    train_base_dir = TRAIN_BASE_DIR
    train_novel_dir = TRAIN_NOVEL_DIR

    val_base_dir = VAL_BASE_DIR
    val_novel_dir = VAL_NOVEL_DIR

    all_classes = ALL_CLASSES

    safe_class_names =[cls.replace(" ", "_") for cls in all_classes]

    X_train, y_train = [],[]
    X_test, y_test = [],[]

    def load_features_from_dir(directory, X_list, y_list):
        if not os.path.exists(directory):
            print(f"Warning: Directory does not exist -> {directory}")
            return
            
        files = sorted(glob.glob(os.path.join(directory, '*.npy'))) 
        for f in files:
            filename = os.path.basename(f)
            for cls in all_classes:
                if f"_{cls}_" in filename:
                    embedding = np.load(f).flatten()
                    X_list.append(embedding)
                    y_list.append(cls) 
                    break

    print("Loading 100% of Training features... (This might take a minute)")
    load_features_from_dir(train_base_dir, X_train, y_train)
    load_features_from_dir(train_novel_dir, X_train, y_train)

    print("Loading Validation/Testing features...")
    load_features_from_dir(val_base_dir, X_test, y_test)
    load_features_from_dir(val_novel_dir, X_test, y_test)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"Loaded {X_train.shape[0]} training embeddings.")
    print(f"Loaded {X_test.shape[0]} validation embeddings.")

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("Error: Training or testing data is empty. Please check your file paths.")
        return

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    os.makedirs(SAVE_DIR, exist_ok=True) 
    
    best_model_file = os.path.join(SAVE_DIR, 'best_mlp.pth')
    last_model_file = os.path.join(SAVE_DIR, 'last_mlp.pth')
    scaler_file = os.path.join(SAVE_DIR, 'saved_scaler.joblib')
    le_file = os.path.join(SAVE_DIR, 'saved_le.joblib') 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_dim = X_train.shape[1]
    num_classes = len(le.classes_)
    
    model = MLP_PyTorch(input_dim=input_dim, num_classes=num_classes).to(device)

    # Base Triplet Configurations
    if DISTANCE_METRIC == 'cosine':
        pml_dist = pml_distances.CosineSimilarity()
        triplet_margin = 0.2
    else: 
        pml_dist = pml_distances.LpDistance(p=2)
        triplet_margin = 1.0

    if os.path.exists(best_model_file) and os.path.exists(scaler_file) and os.path.exists(le_file):
        print(f"\nFound saved PyTorch model at {SAVE_DIR}! Loading weights instantly...")
        model.load_state_dict(torch.load(best_model_file, map_location=device))
        model.eval()
        scaler = load(scaler_file)
        le = load(le_file)
        X_test_scaled = scaler.transform(X_test)
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        print("\nNo saved model found. Preparing for Training...")
        # scaler = StandardScaler()
        scaler = Normalizer(norm='l2')
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) 
        
        dump(scaler, scaler_file)
        dump(le, le_file)
        
        X_tr, X_val, y_tr, y_val = train_test_split(X_train_scaled, y_train_encoded, test_size=0.1, random_state=42, stratify=y_train_encoded)
        
        # Define DataLoaders once
        batch_size = 2048
        train_dataset = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        
        # Base Raw Class weights for final training
        print("\nCalculating Base Class Weights...")
        raw_class_weights = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)

        # ==============================================================
        # OPTUNA HYPERPARAMETER SEARCH (META "LEAVE-ONE-OUT" STRATEGY)
        # ==============================================================
        novel_class_idx = le.transform([TARGET_NOVEL_CLASS])[0]
        
        # 1. Identify pure base classes (exclude the real novel class entirely)
        base_class_indices = [idx for idx in range(num_classes) if idx != novel_class_idx]
        
        # 2. Randomly select 3 Base classes to act as our "Simulated 5-shot Novel Classes"
        simulated_novel_indices = np.random.choice(base_class_indices, size=3, replace=False)
        print(f"\n--- Setting up Meta-Optuna Few-Shot Simulation ---")
        print(f"Hiding real novel class: {TARGET_NOVEL_CLASS}")
        print(f"Simulating K-shot learning on base classes: {le.inverse_transform(simulated_novel_indices)}")

        # 3. Create the Meta-Training Dataset
        X_meta_tr_list, y_meta_tr_list = [], []
        for cls_idx in range(num_classes):
            if cls_idx == novel_class_idx:
                continue # Skip the real novel class
                
            cls_mask = (y_tr == cls_idx)
            X_cls = X_tr[cls_mask]
            y_cls = y_tr[cls_mask]
            
            if cls_idx in simulated_novel_indices:
                # DOWNSAMPLE THESE 3 CLASSES TO EXACTLY 'SHOTS' (e.g., 5)
                selected_indices = np.random.choice(len(X_cls), min(SHOTS, len(X_cls)), replace=False)
                X_meta_tr_list.append(X_cls[selected_indices])
                y_meta_tr_list.append(y_cls[selected_indices])
            else:
                # Keep full data for the rest of the base classes
                X_meta_tr_list.append(X_cls)
                y_meta_tr_list.append(y_cls)
                
        X_meta_tr = np.vstack(X_meta_tr_list)
        y_meta_tr = np.concatenate(y_meta_tr_list)
        meta_train_dataset = TensorDataset(torch.FloatTensor(X_meta_tr), torch.LongTensor(y_meta_tr))
        
        # 4. Create Meta-Validation Dataset (Exclude real novel class, keep full val set for simulated ones)
        val_base_mask = (y_val != novel_class_idx)
        meta_val_dataset = TensorDataset(torch.FloatTensor(X_val[val_base_mask]), torch.LongTensor(y_val[val_base_mask]))

        # 5. Calculate class weights for the simulated setup
        meta_raw_class_weights = compute_class_weight('balanced', classes=np.unique(y_meta_tr), y=y_meta_tr)
        full_meta_raw_weights = np.ones(num_classes) # Fill a full array to match model output dimension
        for i, cls_idx in enumerate(np.unique(y_meta_tr)):
            full_meta_raw_weights[cls_idx] = meta_raw_class_weights[i]


        def objective(trial):
            batch_size = trial.suggest_categorical('batch_size',[256, 512, 1024, 2048, 4096])
            gamma = trial.suggest_float('gamma', 0.5, 3.0)
            center_weight = trial.suggest_float('center_weight', 0.005, 0.1, log=True)
            lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
            weight_smoothing = trial.suggest_float('weight_smoothing', 0.2, 0.8)
            novel_multiplier = trial.suggest_float('novel_multiplier', 1.0, 15.0)

            # Apply multiplier to the 3 SIMULATED novel classes
            smoothed_weights = np.power(full_meta_raw_weights, weight_smoothing)
            for sim_idx in simulated_novel_indices:
                smoothed_weights[sim_idx] *= novel_multiplier
            class_weights_tensor = torch.FloatTensor(smoothed_weights).to(device)

            trial_model = MLP_PyTorch(input_dim=input_dim, num_classes=num_classes).to(device)
            criterion_focal = FocalLoss(weight=class_weights_tensor, gamma=gamma).to(device)
            criterion_center = CenterLoss(num_classes=num_classes, feat_dim=128, device=device)
            
            miner = pml_miners.BatchHardMiner(distance=pml_dist)
            criterion_triplet = pml_losses.TripletMarginLoss(margin=triplet_margin, distance=pml_dist)

            optimizer = optim.Adam(trial_model.parameters(), lr=lr, weight_decay=weight_decay)
            optimizer_center = optim.SGD(criterion_center.parameters(), lr=0.5)

            # USE META DATASETS FOR SEARCH
            train_loader = DataLoader(meta_train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(meta_val_dataset, batch_size=batch_size, shuffle=False)

            best_val_metric = -1.0 
            epochs_no_improve = 0
            search_patience = 50
            
            for epoch in range(MAX_EPOCHS):
                trial_model.train()
                for features, labels in train_loader:
                    features, labels = features.to(device), labels.to(device)
                    optimizer.zero_grad()
                    optimizer_center.zero_grad()
                    
                    logits, hidden_feats = trial_model(features)
                    
                    # Core Center Loss is always applied
                    loss_center = criterion_center(hidden_feats, labels)
                    loss = center_weight * loss_center

                    # Process Configurable loss logic
                    if LOSS_COMBINATION in['focal_center', 'focal_triplet_center']:
                        loss += criterion_focal(logits, labels)

                    if LOSS_COMBINATION in ['triplet_center', 'focal_triplet_center']:
                        hard_pairs = miner(hidden_feats, labels)
                        loss += criterion_triplet(hidden_feats, labels, hard_pairs)

                    loss.backward()
                    
                    optimizer.step()
                    for param in criterion_center.parameters():
                        param.grad.data *= (1. / center_weight)
                    optimizer_center.step()
                    
                # Calculate training centers for validation if using distance metrics natively
                if DISTANCE_METRIC in ['l2', 'cosine']:
                    train_centers = compute_train_centers(trial_model, train_loader, num_classes, device)

                # Validation loop
                trial_model.eval()
                all_preds =[]
                all_labels =[]
                
                with torch.no_grad():
                    for features, labels in val_loader:
                        features, labels = features.to(device), labels.to(device)
                        logits, hidden_feats = trial_model(features)
                        
                        if DISTANCE_METRIC in ['l2', 'cosine']:
                            scores = compute_prototype_scores(hidden_feats, train_centers, metric=DISTANCE_METRIC)
                        elif DISTANCE_METRIC == 'logits':
                            scores = logits

                        preds = get_predictions(scores, labels, top_k=TOP_K_VALUE, use_top_k=USE_TOP_K_METRICS)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                
                f2_scores = fbeta_score(all_labels, all_preds, beta=2, average=None, labels=np.arange(num_classes), zero_division=0)
                f1_scores = f1_score(all_labels, all_preds, average=None, labels=np.arange(num_classes), zero_division=0)
                
                # Average scores over the 3 SIMULATED novel classes
                simulated_novel_f2 = np.mean([f2_scores[i] for i in simulated_novel_indices])
                simulated_novel_f1 = np.mean([f1_scores[i] for i in simulated_novel_indices])

                # Calculate base F1 over the PURE base classes (excluding the simulated ones)
                pure_base_indices = [i for i in base_class_indices if i not in simulated_novel_indices]
                macro_base_f1 = np.mean([f1_scores[i] for i in pure_base_indices])
                
                if CUSTOM_METRIC_TYPE == 'f1_novel':
                    custom_metric = simulated_novel_f1
                elif CUSTOM_METRIC_TYPE == 'f2_novel':
                    custom_metric = simulated_novel_f2
                else:
                    custom_metric = (F2_NOVEL_RATIO * simulated_novel_f2) + (F1_ALL_RATIO * macro_base_f1)

                
                if custom_metric > best_val_metric:
                    best_val_metric = custom_metric
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                trial.report(custom_metric, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                
                if epochs_no_improve >= search_patience:
                    break
                    
            return best_val_metric

        file_path = os.path.join(PLOT_DIR, 'Best_Hyparameters.json')
        if not os.path.exists(file_path) and not BEST_HYPERPARAMETERS:
            print("\n" + "="*50)
            print("Starting Optuna Hyperparameter Search (25 Trials)...")
            print("="*50)
            sampler = optuna.samplers.TPESampler(seed=SEED) 
            pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            study = optuna.create_study(direction='maximize', pruner=pruner, sampler=sampler)
            study.optimize(objective, n_trials=25)

            print("\n" + "="*50)
            print("Optuna Search Complete!")
            print("Best Hyperparameters:", study.best_params)
            print("="*50 + "\n")

            best_params = study.best_params
            file_path = os.path.join(PLOT_DIR, 'Best_Hyparameters.json')
            with open(file_path, 'w') as f:
                json.dump(best_params, f, indent=4)
        elif not os.path.exists(file_path) and BEST_HYPERPARAMETERS:
            with open(file_path, 'w') as f:
                json.dump(BEST_HYPERPARAMETERS, f, indent=4)

        # ==============================================================
        # FINAL FULL TRAINING WITH BEST HYPERPARAMETERS
        # ==============================================================
        print("Training Final Model with Best Hyperparameters...")
        seed_everything(SEED)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                best_params = json.load(f)
                print("+"*30)
                print(f"Found parameter: {best_params}")
        else:
            best_params = BEST_HYPERPARAMETERS
            print("+"*30)
            print(f"Found parameter: {best_params}")
        
        final_batch_size = best_params.get('batch_size', 2048)
        train_loader = DataLoader(train_dataset, batch_size=final_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=final_batch_size, shuffle=False)

        smoothed_weights = np.power(raw_class_weights, best_params['weight_smoothing'])
        if 'novel_multiplier' in best_params:
            smoothed_weights[novel_class_idx] *= best_params['novel_multiplier']
        class_weights_tensor = torch.FloatTensor(smoothed_weights).to(device)

        criterion_focal = FocalLoss(weight=class_weights_tensor, gamma=best_params['gamma']).to(device)
        criterion_center = CenterLoss(num_classes=num_classes, feat_dim=128, device=device)
        final_center_weight = best_params['center_weight']

        miner = pml_miners.BatchHardMiner(distance=pml_dist)
        criterion_triplet = pml_losses.TripletMarginLoss(margin=triplet_margin, distance=pml_dist)
        
        optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay']) 
        optimizer_center = optim.SGD(criterion_center.parameters(), lr=0.5)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        
        patience = 50
        best_val_metric = -1.0
        epochs_no_improve = 0
        
        for epoch in range(MAX_EPOCHS):
            model.train()
            train_loss = 0.0
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                
                optimizer.zero_grad()
                optimizer_center.zero_grad()
                
                logits, hidden_feats = model(features)
                
                loss_center = criterion_center(hidden_feats, labels)
                loss = final_center_weight * loss_center
                
                if LOSS_COMBINATION in['focal_center', 'focal_triplet_center']:
                    loss += criterion_focal(logits, labels)

                if LOSS_COMBINATION in ['triplet_center', 'focal_triplet_center']:
                    hard_pairs = miner(hidden_feats, labels)
                    loss += criterion_triplet(hidden_feats, labels, hard_pairs)

                loss.backward()
                
                optimizer.step()
                for param in criterion_center.parameters():
                    param.grad.data *= (1. / final_center_weight)
                optimizer_center.step()
                
                train_loss += loss.item()
                
            # Calculation of Validation prototypes if active
            if DISTANCE_METRIC in ['l2', 'cosine']:
                train_centers = compute_train_centers(model, train_loader, num_classes, device)

            model.eval()
            all_preds =[]
            all_labels =[]
            val_loss = 0.0
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    logits, hidden_feats = model(features)
                    
                    loss_center = criterion_center(hidden_feats, labels)
                    loss = final_center_weight * loss_center
                    
                    if LOSS_COMBINATION in ['focal_center', 'focal_triplet_center']:
                        loss += criterion_focal(logits, labels)
                    if LOSS_COMBINATION in['triplet_center', 'focal_triplet_center']:
                        hard_pairs = miner(hidden_feats, labels)
                        loss += criterion_triplet(hidden_feats, labels, hard_pairs)

                    val_loss += loss.item()

                    if DISTANCE_METRIC in ['l2', 'cosine']:
                        scores = compute_prototype_scores(hidden_feats, train_centers, metric=DISTANCE_METRIC)
                    elif DISTANCE_METRIC == 'logits':
                        scores = logits

                    preds = get_predictions(scores, labels, top_k=TOP_K_VALUE, use_top_k=USE_TOP_K_METRICS)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            val_loss /= len(val_loader)
            
            f2_scores = fbeta_score(all_labels, all_preds, beta=2, average=None, labels=np.arange(num_classes), zero_division=0) 
            f1_scores = f1_score(all_labels, all_preds, average=None, labels=np.arange(num_classes), zero_division=0)
            novel_f1 = f1_scores[novel_class_idx] 
            novel_f2 = f2_scores[novel_class_idx] 
            macro_f1 = np.mean(f1_scores)
            
            if CUSTOM_METRIC_TYPE == 'f1_novel':
                custom_metric = novel_f1
            elif CUSTOM_METRIC_TYPE == 'f2_novel':
                custom_metric = novel_f2
            else:
                custom_metric = (F2_NOVEL_RATIO * novel_f2) + (F1_ALL_RATIO * macro_f1) ### ADAM CHANGED
            scheduler.step(custom_metric)

            if (epoch+1) % 10 == 0:
                print(f"Epoch[{epoch+1}/{MAX_EPOCHS}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

            if custom_metric > best_val_metric + 1e-4: 
                best_val_metric = custom_metric
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_file) 
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1} (No improvement in Custom Metric)")
                break
                
        torch.save(model.state_dict(), last_model_file)
        print("Training Complete! Saved best and last models.")
        
        model.load_state_dict(torch.load(best_model_file))
        model.eval()

    # --- 4. DETAILED METRICS CALCULATION (Final Dataset) ---
    final_train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train_encoded))
    final_train_loader = DataLoader(final_train_dataset, batch_size=2048, shuffle=False)

    # Compute training centers unconditionally here as they will be needed for the plot
    train_centers = compute_train_centers(model, final_train_loader, num_classes, device)
    train_centers_np = train_centers.cpu().numpy()

    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_test_tensor = torch.LongTensor(y_test_encoded).to(device)

    with torch.no_grad():
        test_logits, X_separated_features_all = model(X_test_tensor)

        if DISTANCE_METRIC in['l2', 'cosine']:
            scores = compute_prototype_scores(X_separated_features_all, train_centers, metric=DISTANCE_METRIC)
        elif DISTANCE_METRIC == 'logits':
            scores = test_logits

        y_pred_encoded = get_predictions(scores, y_test_tensor, top_k=TOP_K_VALUE, use_top_k=USE_TOP_K_METRICS)
        y_pred_encoded = y_pred_encoded.cpu().numpy()
        X_separated_features_all = X_separated_features_all.cpu().numpy()

    accuracy = np.mean(y_pred_encoded == y_test_encoded)
    metric_type = f"Top-{TOP_K_VALUE}" if USE_TOP_K_METRICS else "Top-1"
    
    print(f"\nOverall Validation Set Accuracy ({metric_type}): {accuracy * 100:.2f}%")
    
    y_pred = le.inverse_transform(y_pred_encoded) 
    
    print(f"\n--- Detailed Classification Report (Classes, {metric_type}) ---")
    os.makedirs(PLOT_DIR, exist_ok=True)
    file_path = os.path.join(PLOT_DIR, 'Classification_Report_CHECK_ADAM.txt')
    report = classification_report(y_test, y_pred, digits=4)
    with open(file_path, "w") as f:
        mode_str = "MLP Logits" if DISTANCE_METRIC == 'logits' else f"Prototypes ({DISTANCE_METRIC.upper()})"
        f.write(f"Prediction Mode: {mode_str}\n")
        f.write(f"Loss Combination : {LOSS_COMBINATION.upper()}\n")
        f.write(f"Overall Validation Set Accuracy ({metric_type}): {accuracy * 100:.2f}%\n")
        f.write(report)
        print(report)

    # --- 5. Sampling Data for Visualization (Train or Test) ---
    print(f"\nSampling {VISUALIZATION_SOURCE.upper()} data for visualization...")
    
    if VISUALIZATION_SOURCE == 'train':
        X_source = X_train_scaled
        y_source = le.inverse_transform(y_train_encoded)
    else:
        X_source = X_test_scaled
        y_source = y_test # Original string labels
        
    X_viz_raw = []
    y_test_viz = []
    samples_per_class = 200 
    
    unique_classes_in_source = np.unique(y_source)
    for cls in unique_classes_in_source:
        idx = np.where(y_source == cls)[0]
        selected_idx = np.random.choice(idx, min(samples_per_class, len(idx)), replace=False)
        X_viz_raw.append(X_source[selected_idx])
        y_test_viz.extend([cls] * len(selected_idx))
        
    X_viz_raw = np.concatenate(X_viz_raw, axis=0)
    y_test_viz = np.array(y_test_viz)

    # --- 6. Extract Features/Logits for the sampled data ---
    model.eval()
    X_viz_tensor = torch.FloatTensor(X_viz_raw).to(device)
    with torch.no_grad():
        # Get both logits and embeddings (h2) for the sampled points
        viz_logits, viz_embeddings = model(X_viz_tensor)
        
        # This will be used by the MDS logic in Section 7
        X_separated_features = viz_embeddings.cpu().numpy()
        X_logits_features = viz_logits.cpu().numpy()

    # --- 7. Dynamic Dimensionality Reduction based on DISTANCE_METRIC ---
    print(f"Applying Dimensionality Reduction based on {DISTANCE_METRIC.upper()} logic...")

    if DISTANCE_METRIC == 'logits':
        # Points = Sample Logits + Center Logits
        train_logit_centers = compute_train_logit_centers(model, final_train_loader, num_classes, device)
        combined_points = np.vstack([X_logits_features, train_logit_centers])
        print("Computing Logit-based Euclidean distance matrix...")
        dist_matrix = euclidean_distances(combined_points)

    elif DISTANCE_METRIC == 'cosine':
        # Points = Sample Embeddings + Center Embeddings
        combined_points = np.vstack([X_separated_features, train_centers_np])
        print("Computing Cosine distance matrix (1 - Similarity)...")
        sim_matrix = cosine_similarity(combined_points)
        dist_matrix = 1 - sim_matrix
        dist_matrix = np.maximum(dist_matrix, 0)

    else: # DISTANCE_METRIC == 'l2'
        combined_points = np.vstack([X_separated_features, train_centers_np])
        print("Computing Euclidean distance matrix (L2)...")
        dist_matrix = euclidean_distances(combined_points)

    # Apply MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    coords = mds.fit_transform(dist_matrix)
    
    X_viz_2d = coords[:-num_classes]           
    centers_2d = coords[-num_classes:]  

    # --- 8. Dynamic Plotting ---
    print("Generating Plot...")
    plt.figure(figsize=(16, 12))
    sns.set_theme(style="whitegrid")
    
    unique_classes_viz = np.unique(y_test_viz)
    class_palette = dict(zip(le.classes_, sns.color_palette("tab20", len(le.classes_))))
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1, alpha=0.8)

    # 1. Plot the cloud of points (Validation Samples)
    sns.scatterplot(
        x=X_viz_2d[:, 0], y=X_viz_2d[:, 1],
        hue=y_test_viz, palette=class_palette, s=50, alpha=0.5, edgecolor=None
    )

    # 2. Plot the Stars (Train Prototype Centers)
    for i, cls in enumerate(le.classes_):
        c_x, c_y = centers_2d[i]
        plt.scatter(c_x, c_y, marker='*', s=1000, color=class_palette[cls], 
                    edgecolor='black', linewidth=1.5, zorder=10)
        
        plt.annotate(cls, (c_x, c_y), xytext=(10, 10), textcoords='offset points',
                     fontsize=10, fontweight='bold', bbox=bbox_props, zorder=11)

    title_str = {
        'cosine': "MDS Cosine Map: Physical Distance $\\approx$ 1 - Cosine Similarity",
        'l2': "MDS Euclidean Map: Physical Distance $\\approx$ L2 Embedding Distance",
        'logits': "MDS Logit Map: Physical Distance $\\approx$ Distance between Class Scores"
    }

    plt.title(f"{title_str.get(DISTANCE_METRIC, 'MDS Distance Map')}\n"
              f"({VISUALIZATION_SOURCE.capitalize()} Samples vs. Training Centers)", 
              fontsize=16, fontweight='bold')
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(title='Vehicle Classes', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plot_filename = f'Distance_Map_{DISTANCE_METRIC.upper()}_{VISUALIZATION_SOURCE}_Source.png'
    plt.savefig(os.path.join(PLOT_DIR, plot_filename), dpi=300)
    print(f"Done! Saved distance-based plot as '{plot_filename}'")

    # === 9. Base vs Novel Evaluation and Visualization ===
    evaluate_and_visualize_superclasses(
        y_test, 
        y_pred, 
        X_viz_2d,            # The MDS coordinates for samples
        y_test_viz, 
        centers_viz=centers_2d, # The MDS coordinates for stars
        le=le,
        metric_name=DISTANCE_METRIC.upper()
    )

if __name__ == "__main__":
    main()