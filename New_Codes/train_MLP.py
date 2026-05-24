import os
import sys
import glob
import random
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import optuna
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.manifold import MDS

# PyTorch Metric Learning for Triplet Loss
try:
    from pytorch_metric_learning import losses as pml_losses
    from pytorch_metric_learning import miners as pml_miners
    from pytorch_metric_learning import distances as pml_distances
except ImportError:
    print("Error: Please install pytorch-metric-learning to use Triplet Loss.")
    print("Run: pip install pytorch-metric-learning")
    sys.exit(1)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# Global Constants
# ==========================================

ALL_CLASSES = [
    'Small', 'MediumStandard', 'MediumSmall', 'TruckTractor', 
    'LongHeavyDuty', 'HeavyDuty', 'Other', 'Bulldozers', 
    'CementMixerTrucks', 'ExtremelyLongHeavyDutyTraileronly', 
    'MobileCranes', 'Forklifts', 'ExtremelyLongHeavyDuty', 'HeavyDutyTractorTruck'
]

# Exclude classes with < 100 instances
EXCLUDED_CLASSES = {'HeavyDutyTractorTruck', 'ExtremelyLongHeavyDuty', 'Forklifts'}

# ==========================================
# 1. Neural Network & Losses
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
        return logits, h2 # h2 is the 128-D embedding used for centers

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.weight = weight 
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss) 
        focal_term = (1 - pt) ** self.gamma
        
        if self.weight is not None:
            focal_loss = self.weight[targets] * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss
        return focal_loss.mean()

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.centers = nn.Parameter(torch.randn(self.num_classes, feat_dim).to(device))
        self.device = device

    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)
        
        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        
        dist = distmat * mask.float()
        return dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

# ==========================================
# 2. Distance Helpers
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
    return centers / counts.clamp(min=1e-8).unsqueeze(1)

def compute_train_logit_centers(model, dataloader, num_classes, device):
    model.eval()
    centers = torch.zeros(num_classes, num_classes).to(device)
    counts = torch.zeros(num_classes).to(device)
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            logits, _ = model(features)
            centers.index_add_(0, labels, logits)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float))
    return centers / counts.clamp(min=1e-8).unsqueeze(1)

def compute_prototype_scores(h2, centers, metric='cosine'):
    if metric == 'cosine':
        h2_norm = F.normalize(h2, p=2, dim=1)
        centers_norm = F.normalize(centers, p=2, dim=1)
        return torch.mm(h2_norm, centers_norm.t())
    elif metric == 'l2':
        return -torch.cdist(h2, centers, p=2.0)
    else:
        raise ValueError("Invalid metric")

# ==========================================
# 3. Main Script
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP Model with Optuna")
    
    # Paths
    parser.add_argument('--input_emb_dir', type=str, default='/home/adamm/Documents/FSOD/Data/Lavyanut/Images/train/Obj_Embs/', help="Directory containing input embeddings")
    parser.add_argument('--output_dir', type=str, default='./Output_MLP_Training', help="Output directory for prototypes and logs")
    parser.add_argument('--models_dir', type=str, default='models/MLP', help="Output directory for saved models")
    
    # Model Options
    parser.add_argument('--distance_metric', type=str, default='cosine', choices=['l2', 'cosine', 'logits'], help="Metric to compute distance")
    parser.add_argument('--optimization_metric', type=str, default='f1_macro', choices=['f1_macro', 'accuracy', 'f1_weighted'], help="Optuna objective metric")
    parser.add_argument('--loss_combination', type=str, default='focal_center', choices=['focal_center', 'triplet_center', 'focal_triplet_center'], help="Losses to combine")
    
    # Hyperparameters
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--max_epochs', type=int, default=500, help="Maximum epochs per training")
    parser.add_argument('--optuna_trials', type=int, default=20, help="Number of Optuna trials")
    
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Derived directories
    PROTOTYPES_DIR = os.path.join(args.output_dir, 'prototypes')
    MODELS_DIR = args.models_dir
    
    os.makedirs(PROTOTYPES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- 1. Load Data ---
    X, y = [], []
    print("Loading Embeddings...")
    
    for f in glob.glob(os.path.join(args.input_emb_dir, '*.npy')):
        filename = os.path.basename(f)
        found_cls = None
        for cls in ALL_CLASSES:
            if f"_{cls}_" in filename:
                found_cls = cls
                break
                
        # Filter excluded classes
        if found_cls and found_cls not in EXCLUDED_CLASSES:
            emb = np.load(f).flatten()
            X.append(emb)
            y.append(found_cls)
            
    X = np.array(X)
    y = np.array(y)
    print(f"Loaded {len(X)} samples across {len(np.unique(y))} classes.\n")

    # --- 2. Prep Data (Scale, Encode, Split) ---
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    scaler = Normalizer(norm='l2')
    X_scaled = scaler.fit_transform(X)

    dump(le, os.path.join(MODELS_DIR, 'label_encoder.joblib'))
    dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_encoded, test_size=0.15, random_state=args.seed, stratify=y_encoded
    )

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

    raw_class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

    pml_dist = pml_distances.CosineSimilarity() if args.distance_metric == 'cosine' else pml_distances.LpDistance(p=2)
    triplet_margin = 0.2 if args.distance_metric == 'cosine' else 1.0

    # --- 3. Optuna Hyperparameter Search ---
    def objective(trial):
        batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
        gamma = trial.suggest_float('gamma', 1.0, 3.0)
        center_weight = trial.suggest_float('center_weight', 0.01, 0.1, log=True)
        lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        weight_smoothing = trial.suggest_float('weight_smoothing', 0.2, 0.9)

        smoothed_weights = np.power(raw_class_weights, weight_smoothing)
        class_weights_tensor = torch.FloatTensor(smoothed_weights).to(device)

        trial_model = MLP_PyTorch(X_train.shape[1], num_classes).to(device)
        criterion_focal = FocalLoss(weight=class_weights_tensor, gamma=gamma).to(device)
        criterion_center = CenterLoss(num_classes, 128, device)
        criterion_triplet = pml_losses.TripletMarginLoss(margin=triplet_margin, distance=pml_dist)
        miner = pml_miners.BatchHardMiner(distance=pml_dist)

        optimizer = optim.Adam(trial_model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer_center = optim.SGD(criterion_center.parameters(), lr=0.5)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_metric = 0.0
        patience, epochs_no_improve = 10, 0

        for epoch in range(args.max_epochs):
            trial_model.train()
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                optimizer_center.zero_grad()

                logits, hidden = trial_model(features)
                loss = center_weight * criterion_center(hidden, labels)
                
                if 'focal' in args.loss_combination:
                    loss += criterion_focal(logits, labels)
                if 'triplet' in args.loss_combination:
                    loss += criterion_triplet(hidden, labels, miner(hidden, labels))

                loss.backward()
                optimizer.step()
                for param in criterion_center.parameters():
                    param.grad.data *= (1. / center_weight)
                optimizer_center.step()

            trial_model.eval()
            all_preds, all_labels = [], []
            if args.distance_metric in ['l2', 'cosine']:
                centers = compute_train_centers(trial_model, train_loader, num_classes, device)

            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    logits, hidden = trial_model(features)

                    if args.distance_metric == 'logits':
                        scores = logits
                    else:
                        scores = compute_prototype_scores(hidden, centers, metric=args.distance_metric)

                    _, preds = torch.max(scores, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            if args.optimization_metric == 'f1_macro':
                val_metric = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            elif args.optimization_metric == 'accuracy':
                val_metric = accuracy_score(all_labels, all_preds)

            if val_metric > best_metric:
                best_metric = val_metric
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience: break
            
            trial.report(val_metric, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_metric

    print("\n--- Starting Optuna Hyperparameter Search ---")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.optuna_trials)
    
    best_params = study.best_params
    with open(os.path.join(MODELS_DIR, 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"\nBest Params Found: {best_params}")

    # --- 4. Final Full Training ---
    print("\n--- Training Final Model ---")
    model = MLP_PyTorch(X_train.shape[1], num_classes).to(device)
    
    smoothed_weights = np.power(raw_class_weights, best_params['weight_smoothing'])
    class_weights_tensor = torch.FloatTensor(smoothed_weights).to(device)

    criterion_focal = FocalLoss(weight=class_weights_tensor, gamma=best_params['gamma']).to(device)
    criterion_center = CenterLoss(num_classes, 128, device)
    criterion_triplet = pml_losses.TripletMarginLoss(margin=triplet_margin, distance=pml_dist)
    miner = pml_miners.BatchHardMiner(distance=pml_dist)

    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    optimizer_center = optim.SGD(criterion_center.parameters(), lr=0.5)

    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)

    best_val_metric = 0.0
    patience, epochs_no_improve = 15, 0
    best_model_path = os.path.join(MODELS_DIR, 'best_mlp.pth')

    for epoch in range(args.max_epochs):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            logits, hidden = model(features)
            loss = best_params['center_weight'] * criterion_center(hidden, labels)
            if 'focal' in args.loss_combination: loss += criterion_focal(logits, labels)
            if 'triplet' in args.loss_combination: loss += criterion_triplet(hidden, labels, miner(hidden, labels))

            loss.backward()
            optimizer.step()
            for param in criterion_center.parameters():
                param.grad.data *= (1. / best_params['center_weight'])
            optimizer_center.step()

        model.eval()
        if args.distance_metric in ['l2', 'cosine']:
            train_centers = compute_train_centers(model, train_loader, num_classes, device)

        all_preds, all_labels = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                logits, hidden = model(features)

                if args.distance_metric == 'logits':
                    scores = logits
                else:
                    scores = compute_prototype_scores(hidden, train_centers, metric=args.distance_metric)

                _, preds = torch.max(scores, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        if args.optimization_metric == 'f1_macro':
            current_metric = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        else:
            current_metric = accuracy_score(all_labels, all_preds)

        if current_metric > best_val_metric:
            best_val_metric = current_metric
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
            print(f"Epoch {epoch+1:03d} | New Best Val {args.optimization_metric.upper()}: {best_val_metric:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    # --- 5. Extract and Save Prototypes ---
    print("\n--- Extracting Class Prototypes (Centers) ---")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    full_train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=512)
    
    if args.distance_metric == 'logits':
        final_centers = compute_train_logit_centers(model, full_train_loader, num_classes, device)
    else:
        final_centers = compute_train_centers(model, full_train_loader, num_classes, device)
        
    final_centers_np = final_centers.cpu().numpy()

    # Save to .npy files
    for i, cls_name in enumerate(le.classes_):
        save_path = os.path.join(PROTOTYPES_DIR, f"{cls_name}_center.npy")
        np.save(save_path, final_centers_np[i])
    print(f"Saved {num_classes} center .npy files to {PROTOTYPES_DIR}")

    # --- 6. Visualization using MDS ---
    print("\n--- Generating 2D MDS Visualization ---")
    
    # Get embeddings for validation set to plot
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    with torch.no_grad():
        val_logits, val_hidden = model(X_val_tensor)
        
    if args.distance_metric == 'logits':
        val_feats_to_plot = val_logits.cpu().numpy()
    else:
        val_feats_to_plot = val_hidden.cpu().numpy()

    # Combine data points and centers for MDS relative scaling
    combined_points = np.vstack([val_feats_to_plot, final_centers_np])
    
    if args.distance_metric == 'cosine':
        sim_matrix = cosine_similarity(combined_points)
        dist_matrix = np.maximum(1 - sim_matrix, 0)
    else:
        dist_matrix = euclidean_distances(combined_points)

    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=args.seed)
    coords = mds.fit_transform(dist_matrix)
    
    val_coords = coords[:-num_classes]           
    center_coords = coords[-num_classes:]  

    plt.figure(figsize=(14, 10))
    sns.set_theme(style="whitegrid")
    
    class_palette = dict(zip(le.classes_, sns.color_palette("tab10", num_classes)))
    y_val_names = le.inverse_transform(y_val)

    # Plot Scatter Points
    sns.scatterplot(
        x=val_coords[:, 0], y=val_coords[:, 1],
        hue=y_val_names, palette=class_palette, s=60, alpha=0.5, edgecolor=None
    )

    # Plot Class Centers (Stars)
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1, alpha=0.8)
    for i, cls_name in enumerate(le.classes_):
        c_x, c_y = center_coords[i]
        plt.scatter(c_x, c_y, marker='*', s=1000, color=class_palette[cls_name], 
                    edgecolor='black', linewidth=1.5, zorder=10)
        plt.annotate(cls_name, (c_x, c_y), xytext=(10, 10), textcoords='offset points',
                     fontsize=10, fontweight='bold', bbox=bbox_props, zorder=11)

    plt.title(f"MDS Embedding Space Map ({args.distance_metric.upper()})\nValidation Set vs Learned Centers", 
              fontsize=16, fontweight='bold')
    plt.legend(title='Vehicle Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plot_path = os.path.join(args.output_dir, f'MDS_Map_{args.distance_metric}.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Visualization saved to {plot_path}")
    print("\n✅ Training Complete!")

if __name__ == "__main__":
    main()