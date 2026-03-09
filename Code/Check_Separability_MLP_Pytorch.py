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

# Metric Learning components
from pytorch_metric_learning import losses as pml_losses
from pytorch_metric_learning import distances

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

SAVE_DIR = 'models/MLP-Pytorch-TwoStage-SupCon'
PLOT_DIR = 'Outputs/MLP-Pytorch-t-SNE-Graphs_TwoStage-SupCon'

# ==========================================
# 1. Splitting the Model for Two-Stage Training
# ==========================================

class MLP_Encoder(nn.Module):
    """Stage 1: Learns the 256-D clusters using SupCon"""
    def __init__(self, input_dim, dropout_rate=0.4):
        super(MLP_Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)  # BatchNorm stabilizes SupCon
        self.drop1 = nn.Dropout(dropout_rate)    # Prevents memorizing DINO features
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        h1 = self.drop1(F.relu(self.bn1(self.fc1(x))))
        h2 = F.relu(self.bn2(self.fc2(h1)))
        return h2

class MLP_Classifier(nn.Module):
    """Stage 2: Classifies the frozen clusters"""
    def __init__(self, num_classes):
        super(MLP_Classifier, self).__init__()
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, h2):
        return self.fc3(h2)

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight 
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss) 
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

# ==========================================
# 2. Evaluation and Visualization
# ==========================================
def evaluate_and_visualize_superclasses(y_test, y_pred, X_kpca, X_tsne, y_test_viz):
    base_subclasses =['Bus', 'Dump Truck', 'Tractor', 'Truck Tractor', 'Excavator', 'other-vehicle']
    def map_to_superclass(labels):
        return np.array(['Base' if lbl in base_subclasses else 'Novel' for lbl in labels])
    
    y_test_super = map_to_superclass(y_test)
    y_pred_super = map_to_superclass(y_pred)
    
    print("\n" + "="*50)
    print("--- BASE vs NOVEL Classification Report ---")
    os.makedirs(PLOT_DIR, exist_ok=True)
    report = classification_report(y_test_super, y_pred_super, digits=4)
    with open(os.path.join(PLOT_DIR, 'Base_Novel_Classification.txt'), "w") as f:
        f.write(report)
        print(report)
    print("="*50 + "\n")
    
    y_test_viz_super = map_to_superclass(y_test_viz)
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    unique_superclasses = ['Base', 'Novel']
    super_palette = {'Base': 'royalblue', 'Novel': 'crimson'}
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1, alpha=0.85)

    sns.scatterplot(ax=axes[0], x=X_kpca[:, 0], y=X_kpca[:, 1], hue=y_test_viz_super, palette=super_palette, s=60, alpha=0.6, edgecolor=None)
    sns.scatterplot(ax=axes[1], x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_test_viz_super, palette=super_palette, s=60, alpha=0.6, edgecolor=None, legend=False)
    
    axes[0].set_title('Kernel PCA: Base vs Novel', fontsize=14, fontweight='bold')
    axes[1].set_title('t-SNE Map: Base vs Novel', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'Base_vs_Novel_Separation_NO_CARS_VANS.png'), dpi=300)
    print("Done! Saved Base vs Novel plot.")

# ==========================================
# 3. Main Script Pipeline
# ==========================================
def main():
    hyperparam_search = True  # Set to True to run Optuna
    train_base_dir = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Vehicle_Dataset/Image_Obj_Embs/train/base_class'
    train_novel_dir = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Vehicle_Dataset/Image_Obj_Embs/train/novel_class'
    val_base_dir = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Vehicle_Dataset/Image_Obj_Embs/val/base_class'
    val_novel_dir = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Vehicle_Dataset/Image_Obj_Embs/val/novel_class'

    all_classes =['Bus', 'Dump Truck', 'Tractor', 'Truck Tractor', 'Excavator', 'Cargo Truck', 'Trailer']
    safe_class_names =[cls.replace(" ", "_") for cls in all_classes]

    X_train, y_train, X_test, y_test = [], [], [],[]

    def load_features_from_dir(directory, X_list, y_list):
        if not os.path.exists(directory): return
        for f in glob.glob(os.path.join(directory, '*.npy')):
            filename = os.path.basename(f)
            for safe_cls in safe_class_names:
                if f"_{safe_cls}_" in filename:
                    X_list.append(np.load(f).flatten())
                    y_list.append(safe_cls.replace("_", " ")) 
                    break

    print("Loading features...")
    load_features_from_dir(train_base_dir, X_train, y_train)
    load_features_from_dir(train_novel_dir, X_train, y_train)
    load_features_from_dir(val_base_dir, X_test, y_test)
    load_features_from_dir(val_novel_dir, X_test, y_test)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    os.makedirs(SAVE_DIR, exist_ok=True) 
    best_encoder_file = os.path.join(SAVE_DIR, 'best_encoder.pth')
    best_classifier_file = os.path.join(SAVE_DIR, 'best_classifier.pth')
    scaler_file = os.path.join(SAVE_DIR, 'saved_scaler.joblib')
    le_file = os.path.join(SAVE_DIR, 'saved_le.joblib') 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]
    num_classes = len(le.classes_)

    if os.path.exists(best_encoder_file) and os.path.exists(best_classifier_file) and not hyperparam_search:
        print("\nFound saved Two-Stage PyTorch model! Loading weights...")
        encoder = MLP_Encoder(input_dim=input_dim).to(device)
        classifier = MLP_Classifier(num_classes=num_classes).to(device)
        encoder.load_state_dict(torch.load(best_encoder_file, map_location=device))
        classifier.load_state_dict(torch.load(best_classifier_file, map_location=device))
        encoder.eval()
        classifier.eval()
        scaler = load(scaler_file)
        le = load(le_file)
        X_test_scaled = scaler.transform(X_test)
    else:
        print("\nPreparing Data for Two-Stage Training...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) 
        dump(scaler, scaler_file)
        dump(le, le_file)
        
        X_tr, X_val, y_tr, y_val = train_test_split(X_train_scaled, y_train_encoded, test_size=0.1, random_state=42)
        
        batch_size = 512
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)), batch_size=batch_size, shuffle=False)
        raw_class_weights = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)

        # ==============================================================
        # OPTUNA HYPERPARAMETER SEARCH (TWO-STAGE)
        # ==============================================================
        def objective(trial):
            # Stage 1 Params (Encoder/SupCon)
            supcon_temp = trial.suggest_float('supcon_temp', 0.01, 0.2, log=True)
            enc_lr = trial.suggest_float('enc_lr', 1e-4, 5e-3, log=True)
            enc_wd = trial.suggest_float('enc_wd', 1e-5, 1e-2, log=True)
            dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.6)
            
            # Stage 2 Params (Classifier/Focal)
            gamma = trial.suggest_float('gamma', 0.5, 3.0)
            clf_lr = trial.suggest_float('clf_lr', 1e-4, 5e-3, log=True)
            clf_wd = trial.suggest_float('clf_wd', 1e-5, 1e-2, log=True)
            weight_smoothing = trial.suggest_float('weight_smoothing', 0.2, 0.8)

            # --- Initialize Trial Models ---
            trial_encoder = MLP_Encoder(input_dim=input_dim, dropout_rate=dropout_rate).to(device)
            trial_classifier = MLP_Classifier(num_classes=num_classes).to(device)
            
            # --- TRIAL STAGE 1: Train Encoder ---
            cosine_dist = distances.CosineSimilarity()
            criterion_supcon = pml_losses.SupConLoss(temperature=supcon_temp, distance=cosine_dist).to(device)
            optimizer_enc = optim.Adam(trial_encoder.parameters(), lr=enc_lr, weight_decay=enc_wd)
            
            for epoch in range(250): # Short Stage 1 run for search
                trial_encoder.train()
                for features, labels in train_loader:
                    features, labels = features.to(device), labels.to(device)
                    optimizer_enc.zero_grad()
                    hidden_feats = trial_encoder(features)
                    norm_feats = F.normalize(hidden_feats, p=2, dim=1)
                    loss = criterion_supcon(norm_feats, labels)
                    loss.backward()
                    optimizer_enc.step()
                    
            # Freeze trial encoder
            trial_encoder.eval()
            for param in trial_encoder.parameters():
                param.requires_grad = False
                
            # --- TRIAL STAGE 2: Train Classifier ---
            smoothed_weights = np.power(raw_class_weights, weight_smoothing)
            class_weights_tensor = torch.FloatTensor(smoothed_weights).to(device)
            criterion_focal = FocalLoss(weight=class_weights_tensor, gamma=gamma).to(device)
            optimizer_clf = optim.Adam(trial_classifier.parameters(), lr=clf_lr, weight_decay=clf_wd)
            
            best_val_loss = float('inf')
            
            for epoch in range(500): # Short Stage 2 run for search
                trial_classifier.train()
                for features, labels in train_loader:
                    features, labels = features.to(device), labels.to(device)
                    optimizer_clf.zero_grad()
                    with torch.no_grad():
                        hidden_feats = trial_encoder(features)
                        norm_feats = F.normalize(hidden_feats, p=2, dim=1)
                    logits = trial_classifier(norm_feats)
                    loss = criterion_focal(logits, labels)
                    loss.backward()
                    optimizer_clf.step()
                    
                # Validation (This is what Optuna cares about)
                trial_classifier.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for features, labels in val_loader:
                        features, labels = features.to(device), labels.to(device)
                        hidden_feats = trial_encoder(features)
                        norm_feats = F.normalize(hidden_feats, p=2, dim=1)
                        logits = trial_classifier(norm_feats)
                        val_loss += criterion_focal(logits, labels).item()
                
                val_loss /= len(val_loader)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                    
            return best_val_loss

        # --- RUN OPTUNA ---
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
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    best_params = json.load(f)
            else:
                raise ValueError("No hyperparameters found! Run hyperparam_search=True.")

        print("+"*30)
        print(f"Using parameters: {best_params}")

        # ==============================================================
        # FINAL FULL TRAINING: STAGE 1 (SUPCON ONLY)
        # ==============================================================
        print("\n" + "="*50)
        print("STAGE 1: Training Final Encoder with SupCon Loss")
        print("="*50)
        
        encoder = MLP_Encoder(input_dim=input_dim, dropout_rate=best_params['dropout_rate']).to(device)
        cosine_dist = distances.CosineSimilarity()
        criterion_supcon = pml_losses.SupConLoss(temperature=best_params['supcon_temp'], distance=cosine_dist).to(device)
        optimizer_enc = optim.Adam(encoder.parameters(), lr=best_params['enc_lr'], weight_decay=best_params['enc_wd'])
        
        best_val_supcon = float('inf')
        epochs_no_improve_enc = 0
        
        for epoch in range(250):
            encoder.train()
            train_loss = 0.0
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer_enc.zero_grad()
                
                hidden_feats = encoder(features)
                norm_feats = F.normalize(hidden_feats, p=2, dim=1)
                
                loss = criterion_supcon(norm_feats, labels)
                loss.backward()
                optimizer_enc.step()
                train_loss += loss.item()
                
            encoder.eval()
            val_loss = 0.0
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    hidden_feats = encoder(features)
                    norm_feats = F.normalize(hidden_feats, p=2, dim=1)
                    val_loss += criterion_supcon(norm_feats, labels).item()
            
            val_loss /= len(val_loader)
            if (epoch+1) % 10 == 0:
                print(f"Epoch[{epoch+1}/150], Train SupCon: {train_loss/len(train_loader):.4f}, Val SupCon: {val_loss:.4f}")

            if val_loss < best_val_supcon: 
                best_val_supcon = val_loss
                torch.save(encoder.state_dict(), best_encoder_file)
                epochs_no_improve_enc = 0
            else:
                epochs_no_improve_enc += 1
                
            if epochs_no_improve_enc >= 70: # Early stopping for Stage 1
                print(f"Stage 1 Early stopping at epoch {epoch+1}")
                break

        print("Stage 1 Complete! Encoder frozen.")
        encoder.load_state_dict(torch.load(best_encoder_file))
        
        # FREEZE ENCODER
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()

        # ==============================================================
        # FINAL FULL TRAINING: STAGE 2 (FOCAL LOSS ONLY)
        # ==============================================================
        print("\n" + "="*50)
        print("STAGE 2: Training Classifier on Frozen Embeddings")
        print("="*50)
        
        classifier = MLP_Classifier(num_classes=num_classes).to(device)
        smoothed_weights = np.power(raw_class_weights, best_params['weight_smoothing']) 
        class_weights_tensor = torch.FloatTensor(smoothed_weights).to(device)
        
        criterion_focal = FocalLoss(weight=class_weights_tensor, gamma=best_params['gamma']).to(device)
        optimizer_clf = optim.Adam(classifier.parameters(), lr=best_params['clf_lr'], weight_decay=best_params['clf_wd'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_clf, mode='min', factor=0.5, patience=10)
        
        best_val_focal = float('inf')
        epochs_no_improve_clf = 0
        
        for epoch in range(500):
            classifier.train()
            train_loss = 0.0
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer_clf.zero_grad()
                
                with torch.no_grad(): # Embeddings are frozen!
                    hidden_feats = encoder(features)
                    norm_feats = F.normalize(hidden_feats, p=2, dim=1) 
                
                logits = classifier(norm_feats)
                loss = criterion_focal(logits, labels)
                loss.backward()
                optimizer_clf.step()
                train_loss += loss.item()
                
            classifier.eval()
            val_loss = 0.0
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    hidden_feats = encoder(features)
                    norm_feats = F.normalize(hidden_feats, p=2, dim=1)
                    logits = classifier(norm_feats)
                    val_loss += criterion_focal(logits, labels).item()
            
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            if (epoch+1) % 10 == 0:
                print(f"Epoch[{epoch+1}/250], Train Focal: {train_loss/len(train_loader):.4f}, Val Focal: {val_loss:.4f}")

            if val_loss < best_val_focal: 
                best_val_focal = val_loss
                torch.save(classifier.state_dict(), best_classifier_file)
                epochs_no_improve_clf = 0
            else:
                epochs_no_improve_clf += 1
                
            # if epochs_no_improve_clf >= 30: # Early stopping for Stage 2
            #     print(f"Stage 2 Early stopping at epoch {epoch+1}")
            #     break
                
        print("Stage 2 Complete! Model Fully Trained.")
        classifier.load_state_dict(torch.load(best_classifier_file))

    # --- 4. DETAILED METRICS CALCULATION ---
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_test_tensor = torch.LongTensor(y_test_encoded).to(device)

    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        X_separated_features_all = encoder(X_test_tensor)
        norm_test_feats = F.normalize(X_separated_features_all, p=2, dim=1)
        test_logits = classifier(norm_test_feats)
        
        _, y_pred_encoded = torch.max(test_logits, 1)
        y_pred_encoded = y_pred_encoded.cpu().numpy()
        X_separated_features_all = X_separated_features_all.cpu().numpy()

    accuracy = np.mean(y_pred_encoded == y_test_encoded)
    print(f"\nOverall Validation Set Accuracy: {accuracy * 100:.2f}%")
    
    y_pred = le.inverse_transform(y_pred_encoded) 
    
    print("\n--- Detailed Classification Report (Classes) ---")
    os.makedirs(PLOT_DIR, exist_ok=True)
    with open(os.path.join(PLOT_DIR, 'Classification_Report.txt'), "w") as f:
        report = classification_report(y_test, y_pred, digits=4)
        f.write(f"\nOverall Validation Set Accuracy: {accuracy * 100:.2f}%\n")
        f.write(report)
        print(report)

    # --- 5. Sample Validation Data for Visualization ---
    print("\nSampling Validation data for clear visualization...")
    X_test_viz, y_test_viz, X_features_viz = [], [],[]
    samples_per_class = 200 
    
    for cls in np.unique(y_test):
        idx = np.where(y_test == cls)[0]
        selected_idx = np.random.choice(idx, min(samples_per_class, len(idx)), replace=False)
        X_test_viz.extend(X_test_scaled[selected_idx])
        y_test_viz.extend(y_test[selected_idx])
        X_features_viz.extend(X_separated_features_all[selected_idx])
        
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

    sns.scatterplot(ax=axes[0], x=X_kpca[:, 0], y=X_kpca[:, 1], hue=y_test_viz, palette=class_palette, s=60, alpha=0.6, edgecolor=None)
    sns.scatterplot(ax=axes[1], x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_test_viz, palette=class_palette, s=60, alpha=0.6, edgecolor=None, legend=False)

    axes[0].set_title('Kernel PCA (RBF) with Cluster Labels', fontsize=14, fontweight='bold')
    axes[1].set_title('t-SNE Map with Cluster Labels', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'Test_Set_Labeled_Centers.png'), dpi=300)
    print("Done! Saved plot.")

    evaluate_and_visualize_superclasses(y_test, y_pred, X_kpca, X_tsne, y_test_viz)

if __name__ == "__main__":
    main()