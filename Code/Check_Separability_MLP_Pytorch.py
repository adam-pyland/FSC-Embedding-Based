import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import optuna 

# NEW: Import Metric Learning components
from pytorch_metric_learning import losses as pml_losses
from pytorch_metric_learning import miners as pml_miners
from pytorch_metric_learning import distances

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# ==========================================
# 1. PyTorch Model and Loss Definitions
# ==========================================

# Updated directories for Triplet Loss
SAVE_DIR = 'models/MLP-Pytorch-Triplet-Loss-CosSim'
PLOT_DIR = 'Outputs/MLP-Pytorch-t-SNE-Graphs_Triplet_Loss-CosSim'

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
    """
    Focal Loss pushes the model to focus on hard-to-classify examples 
    and scales down the loss for easy (majority) examples.
    """
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight 
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss) 
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# CenterLoss class has been REMOVED. 
# We now use pytorch_metric_learning directly in the training loops.

# ==========================================
# 2. Evaluation and Visualization
# ==========================================

def evaluate_and_visualize_superclasses(y_test, y_pred, X_kpca, X_tsne, y_test_viz):
    base_subclasses =[
        'Bus', 'Dump Truck', 'Tractor', 
        'Truck Tractor', 'Excavator', 'other-vehicle'
    ]
    
    def map_to_superclass(labels):
        return np.array(['Base' if lbl in base_subclasses else 'Novel' for lbl in labels])
    
    y_test_super = map_to_superclass(y_test)
    y_pred_super = map_to_superclass(y_pred)
    
    print("\n" + "="*50)
    print("--- BASE vs NOVEL Classification Report ---")
    os.makedirs(PLOT_DIR, exist_ok=True)
    file_path = os.path.join(PLOT_DIR, 'Base_Novel_Classification.txt')
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
    plt.savefig(os.path.join(PLOT_DIR, 'Base_vs_Novel_Separation_NO_CARS_VANS.png'), dpi=300)
    print("Done! Saved plot as 'Base_vs_Novel_Separation_NO_CARS_VANS.png'")

# ==========================================
# 3. Main Script Pipeline
# ==========================================

def main():
    hyperparam_search = True  # Set to True if you want to run Optuna again
    train_base_dir = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Vehicle_Dataset/Image_Obj_Embs/train/base_class'
    train_novel_dir = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Vehicle_Dataset/Image_Obj_Embs/train/novel_class'

    val_base_dir = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Vehicle_Dataset/Image_Obj_Embs/val/base_class'
    val_novel_dir = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Vehicle_Dataset/Image_Obj_Embs/val/novel_class'

    all_classes =[
        'Bus', 'Dump Truck', 'Tractor', 
        'Truck Tractor', 'Excavator', 
        'Cargo Truck', 'Trailer'
    ]
    safe_class_names =[cls.replace(" ", "_") for cls in all_classes]

    X_train, y_train =[],[]
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
        batch_size = 512
        train_dataset = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Base Raw Class weights
        print("\nCalculating Base Class Weights...")
        raw_class_weights = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)

        # ==============================================================
        # OPTUNA HYPERPARAMETER SEARCH
        # ==============================================================
        def objective(trial):
            gamma = trial.suggest_float('gamma', 0.5, 3.0)
            triplet_weight = trial.suggest_float('triplet_weight', 0.05, 1.0, log=True) # Triplet weight replacing center
            lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
            weight_smoothing = trial.suggest_float('weight_smoothing', 0.2, 0.8)

            smoothed_weights = np.power(raw_class_weights, weight_smoothing)
            class_weights_tensor = torch.FloatTensor(smoothed_weights).to(device)

            trial_model = MLP_PyTorch(input_dim=input_dim, num_classes=num_classes).to(device)
            criterion_focal = FocalLoss(weight=class_weights_tensor, gamma=gamma).to(device)
            
            # --- NEW: Triplet Miner and Loss ---
            cosine_dist = distances.CosineSimilarity()
            miner = pml_miners.BatchHardMiner(distance=cosine_dist)
            criterion_triplet = pml_losses.TripletMarginLoss(margin=0.2, distance=cosine_dist)
            
            optimizer = optim.Adam(trial_model.parameters(), lr=lr, weight_decay=weight_decay)

            best_val_loss = float('inf')
            
            for epoch in range(200):
                trial_model.train()
                for features, labels in train_loader:
                    features, labels = features.to(device), labels.to(device)
                    optimizer.zero_grad()
                    
                    logits, hidden_feats = trial_model(features)
                    
                    # Optional but recommended: L2 normalize features for Triplet distance
                    norm_feats = F.normalize(hidden_feats, p=2, dim=1)
                    
                    # Mine triplets and calculate losses
                    indices_tuple = miner(norm_feats, labels)
                    loss_triplet = criterion_triplet(norm_feats, labels, indices_tuple)
                    
                    loss_focal = criterion_focal(logits, labels)
                    
                    loss = loss_focal + triplet_weight * loss_triplet
                    loss.backward()
                    optimizer.step()
                    
                # Validation
                trial_model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for features, labels in val_loader:
                        features, labels = features.to(device), labels.to(device)
                        logits, hidden_feats = trial_model(features)
                        
                        norm_feats = F.normalize(hidden_feats, p=2, dim=1)
                        indices_tuple = miner(norm_feats, labels)
                        
                        loss_triplet = criterion_triplet(norm_feats, labels, indices_tuple)
                        loss_focal = criterion_focal(logits, labels)
                        
                        val_loss += (loss_focal + triplet_weight * loss_triplet).item()
                
                val_loss /= len(val_loader)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                    
            return best_val_loss

        # Run Study
        os.makedirs(PLOT_DIR, exist_ok=True)
        file_path = os.path.join(PLOT_DIR, 'Best_Hyparameters.json')

        if not os.path.exists(file_path):
            print("\n" + "="*50)
            print("Starting Optuna Hyperparameter Search (25 Trials)...")
            print("="*50)
            pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            study = optuna.create_study(direction='minimize', pruner=pruner)
            study.optimize(objective, n_trials=25)

            print("\n" + "="*50)
            print("Optuna Search Complete!")
            print("Best Hyperparameters:", study.best_params)
            print("="*50 + "\n")

            best_params = study.best_params
            with open(file_path, 'w') as f:
                json.dump(best_params, f, indent=4)
        else:
            best_params = None


        # ==============================================================
        # FINAL FULL TRAINING WITH BEST HYPERPARAMETERS
        # ==============================================================
        print("Training Final Model with Best Hyperparameters...")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                best_params = json.load(f)
                print("+"*30)
                print(f"Loaded parameters from JSON: {best_params}")
                
                # Graceful fallback: map old 'center_weight' to 'triplet_weight' if reusing old JSON
                if 'triplet_weight' not in best_params and 'center_weight' in best_params:
                    print("Note: Found old 'center_weight'. Automatically mapping to 'triplet_weight'.")
                    best_params['triplet_weight'] = best_params['center_weight']
        
        if best_params is None:
            raise ValueError("No hyperparameters found! Either run hyperparam_search=True or provide a JSON file.")

        # Setup final dynamically smoothed weights
        smoothed_weights = np.power(raw_class_weights, best_params['weight_smoothing'])
        class_weights_tensor = torch.FloatTensor(smoothed_weights).to(device)

        # Final Setup
        criterion_focal = FocalLoss(weight=class_weights_tensor, gamma=best_params['gamma']).to(device)
        
        # --- NEW: Final Triplet setup ---
        cosine_dist = distances.CosineSimilarity()
        miner = pml_miners.BatchHardMiner(distance=cosine_dist)
        criterion_triplet = pml_losses.TripletMarginLoss(margin=0.2, distance=cosine_dist)
        final_triplet_weight = best_params['triplet_weight']
        
        # Removed optimizer_center entirely
        optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay']) 

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        patience = 50
        best_val_loss = float('inf')
        epochs_no_improve = 0
        max_epochs = 500

        for epoch in range(max_epochs):
            model.train()
            train_loss = 0.0
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                logits, hidden_feats = model(features)
                norm_feats = F.normalize(hidden_feats, p=2, dim=1)
                
                indices_tuple = miner(norm_feats, labels)
                loss_triplet = criterion_triplet(norm_feats, labels, indices_tuple)
                loss_focal = criterion_focal(logits, labels)
                
                loss = loss_focal + final_triplet_weight * loss_triplet
                loss.backward()
                
                optimizer.step()
                train_loss += loss.item()
                
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    logits, hidden_feats = model(features)
                    
                    norm_feats = F.normalize(hidden_feats, p=2, dim=1)
                    indices_tuple = miner(norm_feats, labels)
                    
                    loss_triplet = criterion_triplet(norm_feats, labels, indices_tuple)
                    loss_focal = criterion_focal(logits, labels)
                    
                    loss = loss_focal + final_triplet_weight * loss_triplet
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            if (epoch+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{max_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss - 1e-6: 
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_file) 
            else:
                epochs_no_improve += 1
                
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
    file_path = os.path.join(PLOT_DIR, 'Classification_Report.txt')
    report = classification_report(y_test, y_pred, digits=4)
    with open(file_path, "w") as f:
        f.write(f"\nOverall Validation Set Accuracy: {accuracy * 100:.2f}%\n")
        f.write(report)
        print(report)

    # --- 5. Sample Validation Data for Visualization ---
    print("\nSampling Validation data for clear visualization...")
    X_test_viz =[]
    y_test_viz =[]
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