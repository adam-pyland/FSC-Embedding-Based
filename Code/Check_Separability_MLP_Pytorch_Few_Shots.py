import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import optuna # NEW: Imported Optuna

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, f1_score, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import json

# ==========================================
# 1. PyTorch Model and Center Loss Definitions
# ==========================================

SAVE_DIR = 'models/MLP-Pytorch-Few-Shots-Focal-Loss'
PLOT_DIR = 'Outputs/MLP-Pytorch-t-SNE-Graphs_Few-Shots-Focal_Loss'

MAX_EPOCHS = 500

class MLP_PyTorch(nn.Module):
    """
    Replicates the scikit-learn MLP architecture:
    Input -> Linear(512) -> ReLU -> Linear(256) -> ReLU -> Linear(num_classes)
    """
    def __init__(self, input_dim, num_classes):
        super(MLP_PyTorch, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        logits = self.fc3(h2)
        return logits, h2  # Returning h2 to extract the 256-D features easily
    

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
# 2. Evaluation and Visualization
# ==========================================

def evaluate_and_visualize_superclasses(y_test, y_pred, X_kpca, X_tsne, y_test_viz):
    base_subclasses =[
        'Bus', 'Dump Truck', 'Tractor', 
        'Truck Tractor', 'Excavator', 
        'Cargo Truck', 'other-vehicle'
    ]
    
    def map_to_superclass(labels):
        return np.array(['Base' if lbl in base_subclasses else 'Novel' for lbl in labels])
    
    y_test_super = map_to_superclass(y_test)
    y_pred_super = map_to_superclass(y_pred)
    
    print("\n" + "="*50)
    print("--- BASE vs NOVEL Classification Report ---")
    file_path = os.path.join(PLOT_DIR, 'Base_Novel_Classification_CHECK_ADAM.txt')
    report = classification_report(y_test_super, y_pred_super, digits=4)
    with open(file_path, "w") as f:
        f.write(report)
        print(report)
    print("="*50 + "\n")
    
    y_test_viz_super = map_to_superclass(y_test_viz)
    
    print("Plotting Base vs Novel separation graphs...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    
    unique_superclasses =['Base', 'Novel']
    super_palette = {'Base': 'royalblue', 'Novel': 'crimson'}
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1, alpha=0.85)

    sns.scatterplot(
        ax=axes[0], x=X_kpca[:, 0], y=X_kpca[:, 1],
        hue=y_test_viz_super, palette=super_palette, s=60, alpha=0.6, edgecolor=None
    )
    for cls in unique_superclasses:
        if cls in y_test_viz_super:
            cls_points = X_kpca[y_test_viz_super == cls]
            center_x, center_y = np.mean(cls_points[:, 0]), np.mean(cls_points[:, 1])
            axes[0].scatter(center_x, center_y, marker='*', s=800, color=super_palette[cls], edgecolor='black', zorder=10)
            axes[0].annotate(f"{cls} Center", (center_x, center_y), xytext=(10, 10), textcoords='offset points',
                             fontsize=12, fontweight='bold', color='black', bbox=bbox_props, zorder=11)

    axes[0].set_title('Kernel PCA: Base vs Novel Classes', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[0].legend(title='Superclass', bbox_to_anchor=(1.05, 1), loc='upper left')

    sns.scatterplot(
        ax=axes[1], x=X_tsne[:, 0], y=X_tsne[:, 1],
        hue=y_test_viz_super, palette=super_palette, s=60, alpha=0.6, edgecolor=None, legend=False
    )
    for cls in unique_superclasses:
        if cls in y_test_viz_super:
            cls_points = X_tsne[y_test_viz_super == cls]
            center_x, center_y = np.median(cls_points[:, 0]), np.median(cls_points[:, 1])
            axes[1].scatter(center_x, center_y, marker='*', s=800, color=super_palette[cls], edgecolor='black', zorder=10)
            axes[1].annotate(f"{cls} Center", (center_x, center_y), xytext=(10, 10), textcoords='offset points',
                             fontsize=12, fontweight='bold', color='black', bbox=bbox_props, zorder=11)

    axes[1].set_title('t-SNE Map: Base vs Novel Classes', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('t-SNE Dimension 1')
    axes[1].set_ylabel('t-SNE Dimension 2')

    plt.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOT_DIR, 'Base_vs_Novel_Separation_NO_CARS_VANS.png'), dpi=300)
    print("Done! Saved plot as 'Base_vs_Novel_Separation_NO_CARS_VANS.png'")

# ==========================================
# 3. Main Script Pipeline
# ==========================================

def main():
    train_base_dir = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Vehicle_Dataset/Image_Obj_Embs/train/base_class'
    train_novel_dir = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Vehicle_Dataset/Image_Obj_Embs/train/novel_class_few_shot_trailer/'

    val_base_dir = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Vehicle_Dataset/Image_Obj_Embs/val/base_class'
    val_novel_dir = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Vehicle_Dataset/Image_Obj_Embs/val/novel_class'

    all_classes =[
        'Bus', 'Dump Truck', 'Tractor', 
        'Truck Tractor', 'Excavator', 
        'Cargo Truck', 'Trailer'
    ]
    safe_class_names =[cls.replace(" ", "_") for cls in all_classes]

    X_train, y_train = [],[]
    X_test, y_test = [],[]

    def load_features_from_dir(directory, X_list, y_list):
        if not os.path.exists(directory):
            print(f"Warning: Directory does not exist -> {directory}")
            return
            
        files = glob.glob(os.path.join(directory, '*.npy'))
        for f in files:
            filename = os.path.basename(f)
            for safe_cls in safe_class_names:
                if f"_{safe_cls}_" in filename:
                    embedding = np.load(f).flatten()
                    X_list.append(embedding)
                    y_list.append(safe_cls.replace("_", " ")) 
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
    
    best_model_file = os.path.join(SAVE_DIR, 'best_mlp_NO_CARS_VANS.pth')
    last_model_file = os.path.join(SAVE_DIR, 'last_mlp_NO_CARS_VANS.pth')
    scaler_file = os.path.join(SAVE_DIR, 'saved_scaler_NO_CARS_VANS.joblib')
    le_file = os.path.join(SAVE_DIR, 'saved_le_NO_CARS_VANS.joblib') 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_dim = X_train.shape[1]
    num_classes = len(le.classes_)
    
    model = MLP_PyTorch(input_dim=input_dim, num_classes=num_classes).to(device)

    if os.path.exists(best_model_file) and os.path.exists(scaler_file) and os.path.exists(le_file):
        print(f"\nFound saved PyTorch model at {SAVE_DIR}! Loading weights instantly...")
        model.load_state_dict(torch.load(best_model_file, map_location=device))
        model.eval()
        scaler = load(scaler_file)
        le = load(le_file)
        X_test_scaled = scaler.transform(X_test)
    else:
        print("\nNo saved model found. Preparing for Training...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) 
        
        dump(scaler, scaler_file)
        dump(le, le_file)
        
        X_tr, X_val, y_tr, y_val = train_test_split(X_train_scaled, y_train_encoded, test_size=0.1, random_state=42)
        
        # Define DataLoaders once
        batch_size = 2048
        train_dataset = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        
        

        # Base Raw Class weights
        print("\nCalculating Base Class Weights...")
        raw_class_weights = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)

        # ==============================================================
        # OPTUNA HYPERPARAMETER SEARCH
        # ==============================================================
        # Get the integer index for the 'Trailer' class
        novel_class_idx = le.transform(['Trailer'])[0]
        def objective(trial):
            # 1. Suggest hyperparameters
            batch_size = trial.suggest_categorical('batch_size',[256, 512, 1024, 2048, 4096])
            gamma = trial.suggest_float('gamma', 0.5, 3.0)
            center_weight = trial.suggest_float('center_weight', 0.005, 0.1, log=True)
            lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
            weight_smoothing = trial.suggest_float('weight_smoothing', 0.2, 0.8)

            # Let Optuna find the perfect Alpha booster for the Novel class
            novel_multiplier = trial.suggest_float('novel_multiplier', 1.0, 15.0)

            smoothed_weights = np.power(raw_class_weights, weight_smoothing)

            # Explicitly boost the Trailer (Novel) class weight
            smoothed_weights[novel_class_idx] *= novel_multiplier

            class_weights_tensor = torch.FloatTensor(smoothed_weights).to(device)

            trial_model = MLP_PyTorch(input_dim=input_dim, num_classes=num_classes).to(device)
            criterion_focal = FocalLoss(weight=class_weights_tensor, gamma=gamma).to(device)
            criterion_center = CenterLoss(num_classes=num_classes, feat_dim=256, device=device)
            
            optimizer = optim.Adam(trial_model.parameters(), lr=lr, weight_decay=weight_decay)
            optimizer_center = optim.SGD(criterion_center.parameters(), lr=0.5)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # NEW: Track Best F1 Score (Maximize) instead of Loss (Minimize)
            best_val_metric = -1.0 
            epochs_no_improve = 0
            search_patience = 50
            max_search_epochs = 500
            
            for epoch in range(max_search_epochs):
                trial_model.train()
                for features, labels in train_loader:
                    features, labels = features.to(device), labels.to(device)
                    optimizer.zero_grad()
                    optimizer_center.zero_grad()
                    
                    logits, hidden_feats = trial_model(features)
                    
                    loss_focal = criterion_focal(logits, labels)
                    loss_center = criterion_center(hidden_feats, labels)
                    
                    loss = loss_focal + center_weight * loss_center
                    loss.backward()
                    
                    optimizer.step()
                    for param in criterion_center.parameters():
                        param.grad.data *= (1. / center_weight)
                    optimizer_center.step()
                    
                # Validation loop to calculate F1 Score
                trial_model.eval()
                all_preds = []
                all_labels =[]
                
                with torch.no_grad():
                    for features, labels in val_loader:
                        features, labels = features.to(device), labels.to(device)
                        logits, _ = trial_model(features)
                        
                        _, preds = torch.max(logits, 1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                
                # NEW: Calculate F1 Scores
                # Calculate F1 for all classes individually
                f2_scores = fbeta_score(all_labels, all_preds, beta=2, average=None, zero_division=0)
                novel_f2 = f2_scores[novel_class_idx]

                f1_scores = f1_score(all_labels, all_preds, average=None, zero_division=0)
                
                # Extract the F1 for the Novel class
                novel_f1 = f1_scores[novel_class_idx]
                
                # Extract Global Macro F1
                macro_f1 = np.mean(f1_scores)
                
                # THE SECRET SAUCE: Create a custom reward metric
                # 75% focus on improving the Novel Class, 25% focus on keeping Base Classes healthy
                custom_metric = (0.80 * novel_f2) + (0.20 * macro_f1)
                
                # Maximize the custom metric
                if custom_metric > best_val_metric:
                    best_val_metric = custom_metric
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                # Optuna Early Pruning
                trial.report(custom_metric, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                
                # Early Stopping
                if epochs_no_improve >= search_patience:
                    break
                    
            return best_val_metric

        # Run Study
        file_path = os.path.join(PLOT_DIR, 'Best_Hyparameters.json')
        if not os.path.exists(file_path):
            print("\n" + "="*50)
            print("Starting Optuna Hyperparameter Search (25 Trials)...")
            print("="*50)
            pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            study = optuna.create_study(direction='maximize', pruner=pruner)
            study.optimize(objective, n_trials=25)

            print("\n" + "="*50)
            print("Optuna Search Complete!")
            print("Best Hyperparameters:", study.best_params)
            print("="*50 + "\n")

            best_params = study.best_params
            file_path = os.path.join(PLOT_DIR, 'Best_Hyparameters.json')
            # Dump the dictionary into the file
            with open(file_path, 'w') as f:
                json.dump(best_params, f, indent=4)

        # ==============================================================
        # FINAL FULL TRAINING WITH BEST HYPERPARAMETERS
        # ==============================================================
        print("Training Final Model with Best Hyperparameters...")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                best_params = json.load(f)
                print("+"*30)
                print(f"Found parameter: {best_params}")
        
        final_batch_size = best_params.get('batch_size', 2048) # Default to 2048 if not found
        train_loader = DataLoader(train_dataset, batch_size=final_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=final_batch_size, shuffle=False)

        # Setup final dynamically smoothed weights
        smoothed_weights = np.power(raw_class_weights, best_params['weight_smoothing'])
        if 'novel_multiplier' in best_params:
            smoothed_weights[novel_class_idx] *= best_params['novel_multiplier']
        class_weights_tensor = torch.FloatTensor(smoothed_weights).to(device)

        # Final Setup
        criterion_focal = FocalLoss(weight=class_weights_tensor, gamma=best_params['gamma']).to(device)
        criterion_center = CenterLoss(num_classes=num_classes, feat_dim=256, device=device)
        final_center_weight = best_params['center_weight']
        
        optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay']) 
        optimizer_center = optim.SGD(criterion_center.parameters(), lr=0.5)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        
        patience = 50
        best_val_metric = -1.0
        epochs_no_improve = 0
        novel_class_idx = le.transform(['Trailer'])[0]

        for epoch in range(MAX_EPOCHS):
            model.train()
            train_loss = 0.0
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                
                optimizer.zero_grad()
                optimizer_center.zero_grad()
                
                logits, hidden_feats = model(features)
                
                loss_focal = criterion_focal(logits, labels)
                loss_center = criterion_center(hidden_feats, labels)
                
                loss = loss_focal + final_center_weight * loss_center
                loss.backward()
                
                optimizer.step()
                for param in criterion_center.parameters():
                    param.grad.data *= (1. / final_center_weight)
                optimizer_center.step()
                
                train_loss += loss.item()
                
            model.eval()
            all_preds = []
            all_labels =[]
            val_loss = 0.0
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    logits, hidden_feats = model(features)
                    
                    loss_focal = criterion_focal(logits, labels)
                    loss_center = criterion_center(hidden_feats, labels)
                    loss = loss_focal + final_center_weight * loss_center
                    val_loss += loss.item()

                    # Track predictions for Custom Metric
                    _, preds = torch.max(logits, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            val_loss /= len(val_loader)
            

            # --- CALCULATE CUSTOM METRIC FOR EARLY STOPPING ---
            f2_scores = fbeta_score(all_labels, all_preds, beta=2, average=None, zero_division=0) 
            f1_scores = f1_score(all_labels, all_preds, average=None, zero_division=0)
            
            novel_f2 = f2_scores[novel_class_idx] 
            macro_f1 = np.mean(f1_scores)
            
            # This is what we actually care about now
            custom_metric = (0.80 * novel_f2) + (0.20 * macro_f1) 
            
            scheduler.step(custom_metric)


            if (epoch+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{MAX_EPOCHS}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

            # NEW: Save model based on MAXIMIZING the custom metric
            if custom_metric > best_val_metric + 1e-4: 
                best_val_metric = custom_metric
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_file) 
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1} (No improvement in Novel F2/Metric)")
                break
                
        torch.save(model.state_dict(), last_model_file)
        print("Training Complete! Saved best and last models.")
        
        # Load best weights for evaluation
        model.load_state_dict(torch.load(best_model_file))
        model.eval()

    # --- 4. DETAILED METRICS CALCULATION ---
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_test_tensor = torch.LongTensor(y_test_encoded).to(device)

    with torch.no_grad():
        test_logits, X_separated_features_all = model(X_test_tensor)
        _, y_pred_encoded = torch.max(test_logits, 1)
        y_pred_encoded = y_pred_encoded.cpu().numpy()
        X_separated_features_all = X_separated_features_all.cpu().numpy()

    accuracy = np.mean(y_pred_encoded == y_test_encoded)
    print(f"\nOverall Validation Set Accuracy: {accuracy * 100:.2f}%")
    
    y_pred = le.inverse_transform(y_pred_encoded) 
    
    print("\n--- Detailed Classification Report (Classes) ---")
    os.makedirs(PLOT_DIR, exist_ok=True)
    file_path = os.path.join(PLOT_DIR, 'Classification_Report_CHECK_ADAM.txt')
    report = classification_report(y_test, y_pred, digits=4)
    with open(file_path, "w") as f:
        f.write(f"\nOverall Validation Set Accuracy: {accuracy * 100:.2f}%\n")
        f.write(report)
        print(report)

    # --- 5. Sample Validation Data for Visualization ---
    print("\nSampling Validation data for clear visualization...")
    X_test_viz =[]
    y_test_viz = []
    X_features_viz =[]
    samples_per_class = 200 
    
    for cls in np.unique(y_test):
        idx = np.where(y_test == cls)[0]
        selected_idx = np.random.choice(idx, min(samples_per_class, len(idx)), replace=False)
        X_test_viz.extend(X_test_scaled[selected_idx])
        y_test_viz.extend(y_test[selected_idx])
        X_features_viz.extend(X_separated_features_all[selected_idx])
        
    X_test_viz = np.array(X_test_viz)
    y_test_viz = np.array(y_test_viz)
    X_separated_features = np.array(X_features_viz)

    # --- 7. Dimensionality Reduction ---
    print("Applying Kernel PCA (RBF)...")
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=None) 
    X_kpca = kpca.fit_transform(X_separated_features)

    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_separated_features)

    # --- 8. Plotting ---
    print("Plotting graphs and calculating Prototype centers...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    unique_classes = np.unique(y_test_viz)

    class_palette = dict(zip(unique_classes, sns.color_palette("tab10", len(unique_classes))))
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1, alpha=0.85)

    sns.scatterplot(
        ax=axes[0], x=X_kpca[:, 0], y=X_kpca[:, 1],
        hue=y_test_viz, palette=class_palette, s=60, alpha=0.6, edgecolor=None
    )
    for cls in unique_classes:
        cls_points = X_kpca[y_test_viz == cls]
        center_x, center_y = np.mean(cls_points[:, 0]), np.mean(cls_points[:, 1])
        axes[0].scatter(center_x, center_y, marker='*', s=800, color=class_palette[cls], edgecolor='black', zorder=10)
        axes[0].annotate(cls, (center_x, center_y), xytext=(8, 8), textcoords='offset points',
                         fontsize=10, fontweight='bold', color='black', bbox=bbox_props, zorder=11)

    axes[0].set_title('Kernel PCA (RBF) with Cluster Labels', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[0].legend(title='Vehicle Classes', bbox_to_anchor=(1.05, 1), loc='upper left')

    sns.scatterplot(
        ax=axes[1], x=X_tsne[:, 0], y=X_tsne[:, 1],
        hue=y_test_viz, palette=class_palette, s=60, alpha=0.6, edgecolor=None, legend=False
    )
    for cls in unique_classes:
        cls_points = X_tsne[y_test_viz == cls]
        center_x, center_y = np.median(cls_points[:, 0]), np.median(cls_points[:, 1])
        axes[1].scatter(center_x, center_y, marker='*', s=800, color=class_palette[cls], edgecolor='black', zorder=10)
        axes[1].annotate(cls, (center_x, center_y), xytext=(8, 8), textcoords='offset points',
                         fontsize=10, fontweight='bold', color='black', bbox=bbox_props, zorder=11)

    axes[1].set_title('t-SNE Map with Cluster Labels (Prototypes)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('t-SNE Dimension 1')
    axes[1].set_ylabel('t-SNE Dimension 2')

    plt.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOT_DIR, 'Test_Set_Labeled_Centers_NO_CARS_VANS.png'), dpi=300)
    print("Done! Saved plot as 'Test_Set_Labeled_Centers_NO_CARS_VANS.png'")

    # === 9. Base vs Novel Evaluation and Visualization ===
    evaluate_and_visualize_superclasses(y_test, y_pred, X_kpca, X_tsne, y_test_viz)

if __name__ == "__main__":
    main()