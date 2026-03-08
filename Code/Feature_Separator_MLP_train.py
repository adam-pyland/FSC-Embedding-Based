import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, average_precision_score
from sklearn.model_selection import ParameterSampler
import warnings

warnings.filterwarnings("ignore") # Suppress sklearn undefined metric warnings for clean output

# ==========================================
# 1. Configuration & Paths
# ==========================================
# INPUT DIRECTORIES (Update these to your actual paths)
INPUT_BASE_DIR = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Vehicle_Dataset/Image_Obj_Embs/train/base_class'
NOVEL_FEW_SHOT_DIR = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Vehicle_Dataset/Image_Obj_Embs/val/splits/few_shots/30/'
NOVEL_TEST_DIR = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Vehicle_Dataset/Image_Obj_Embs/val/splits/inference_objs/'
# If you have a separate Base Test dir, define it. Otherwise, the script uses the 20% validation set as Base Test.

# OUTPUT DIRECTORIES
MODEL_SAVE_DIR = '/home/adamm/Documents/FSOD/codes/FSC-Embedding-Based-Satellites/FSC-Embedding-Based-Satellites/models/MLPs_Feature_Adapters'
BASE_PROTO_DIR = '/home/adamm/Documents/FSOD/codes/FSC-Embedding-Based-Satellites/FSC-Embedding-Based-Satellites/Outputs/MLP_Feautures/base_features'
NOVEL_PROTO_DIR = '/home/adamm/Documents/FSOD/codes/FSC-Embedding-Based-Satellites/FSC-Embedding-Based-Satellites/Outputs/MLP_Feautures/novel_features'

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(BASE_PROTO_DIR, exist_ok=True)
os.makedirs(NOVEL_PROTO_DIR, exist_ok=True)

BASE_CLASSES =['Small Car', 'Bus', 'Dump Truck', 'Van', 'Tractor', 'Truck Tractor', 'Excavator', 'other-vehicle']
NOVEL_CLASSES = ['Cargo Truck', 'Trailer']
ALL_CLASSES = BASE_CLASSES + NOVEL_CLASSES

# Dynamically determine input dimension (1024 for ViT-L, 768 for ViT-B, etc.)
try:
    sample_file = glob.glob(os.path.join(INPUT_BASE_DIR, '**', '*.npy'), recursive=True)[0]
    DINO_DIM = np.load(sample_file).flatten().shape[0]
except IndexError:
    print(f"Warning: No .npy files found in {INPUT_BASE_DIR}. Defaulting DINO_DIM to 1024.")
    DINO_DIM = 1024

# ==========================================
# 2. Dataset Definition
# ==========================================
def get_label_from_filename(filepath):
    filename = os.path.basename(filepath)
    for cls in ALL_CLASSES:
        safe_cls = cls.replace(' ', '_')
        if safe_cls in filename or cls in filename:
            return cls
    return None

class EmbeddingDataset(Dataset):
    def __init__(self, directory, allowed_classes):
        self.filepaths = glob.glob(os.path.join(directory, '**', '*.npy'), recursive=True)
        self.data =[]
        self.labels = []
        self.class_names =[]
        self.class_to_id = {cls: i for i, cls in enumerate(allowed_classes)}
        
        for fp in self.filepaths:
            cls = get_label_from_filename(fp)
            if cls in allowed_classes:
                try:
                    feat = np.load(fp).flatten()
                    self.data.append(feat)
                    self.labels.append(self.class_to_id[cls])
                    self.class_names.append(cls)
                except Exception:
                    pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx], dtype=torch.float32), 
                torch.tensor(self.labels[idx], dtype=torch.long),
                self.class_names[idx])

# ==========================================
# 3. Model Architecture
# ==========================================
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=1)  # L2 Normalize for Cosine Similarity

class CosineClassifier(nn.Module):
    def __init__(self, num_classes, feat_dim, scale=20.0):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim)) # Learnable class centers
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale 

    def forward(self, features):
        norm_weight = F.normalize(self.weight, p=2, dim=1)
        cosine_sim = F.linear(features, norm_weight)
        return cosine_sim * self.scale

# ==========================================
# 4. Evaluation Metrics Function
# ==========================================
def calculate_metrics(outputs, labels, num_classes):
    """Calculates Acc, Precision, Recall, and mAP (Macro Average Precision)"""
    probs = F.softmax(outputs, dim=1).detach().cpu().numpy()
    preds = np.argmax(probs, axis=1)
    labels_np = labels.detach().cpu().numpy()
    
    acc = accuracy_score(labels_np, preds)
    precision, recall, _, _ = precision_recall_fscore_support(labels_np, preds, average='macro', zero_division=0)
    
    # Calculate classification mAP (Macro Average Precision over P-R curve)
    labels_one_hot = F.one_hot(labels, num_classes=num_classes).cpu().numpy()
    try:
        mAP = average_precision_score(labels_one_hot, probs, average='macro')
    except ValueError:
        mAP = 0.0 # Fallback if batch is completely missing classes

    return acc, precision, recall, mAP

# ==========================================
# 5. Training & Hyperparameter Search
# ==========================================
def train_and_search(device):
    # Load dataset and split 80% Train / 20% Validation
    full_dataset = EmbeddingDataset(INPUT_BASE_DIR, BASE_CLASSES)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Define Hyperparameter Search Space
    param_grid = {
        'hidden_dim':[256, 512, 1024],
        'lr':[1e-3, 5e-4, 1e-4],
        'weight_decay':[1e-4, 1e-3, 1e-2],
        'scale':[10.0, 20.0, 30.0]
    }
    
    # Randomly sample 5 combinations for search
    search_space = list(ParameterSampler(param_grid, n_iter=5, random_state=42))
    
    best_acc = 0.0
    EPOCHS = 20
    
    best_model_path = os.path.join(MODEL_SAVE_DIR, 'best_MLP.pth')
    last_model_path = os.path.join(MODEL_SAVE_DIR, 'last_MLP.pth')

    print(f"Starting Hyperparameter Search ({len(search_space)} configurations)...")
    
    for trial, params in enumerate(search_space):
        print(f"\n--- Trial {trial+1}/{len(search_space)} | Params: {params} ---")
        
        projector = ProjectionHead(DINO_DIM, params['hidden_dim']).to(device)
        classifier = CosineClassifier(len(BASE_CLASSES), params['hidden_dim'], scale=params['scale']).to(device)
        
        optimizer = optim.Adam(list(projector.parameters()) + list(classifier.parameters()), 
                               lr=params['lr'], weight_decay=params['weight_decay'])
        criterion = nn.CrossEntropyLoss()

        trial_best_mAP = 0.0

        for epoch in range(EPOCHS):
            # --- TRAIN ---
            projector.train()
            classifier.train()
            for features, labels, _ in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = classifier(projector(features))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # --- VALIDATION ---
            projector.eval()
            classifier.eval()
            all_outputs, all_labels = [],[]
            with torch.no_grad():
                for features, labels, _ in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = classifier(projector(features))
                    all_outputs.append(outputs)
                    all_labels.append(labels)
            
            all_outputs = torch.cat(all_outputs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            acc, prec, rec, mAP = calculate_metrics(all_outputs, all_labels, len(BASE_CLASSES))
            
            print(f"  Epoch {epoch+1:02d} | Val Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | mAP: {mAP:.4f}")

            # Save if it's the absolute best model overall
            if acc > best_acc:
                best_acc = acc
                print(f"  --> New Global Best Accuracy! Saving to {best_model_path}")
                torch.save({
                    'projector_state_dict': projector.state_dict(),
                    'hidden_dim': params['hidden_dim'], # Save param so we can load it later
                    'Acc': acc
                }, best_model_path)
        
        # Save the absolute last model of the last trial (or you can modify to save last of best trial)
        if trial == len(search_space) - 1:
            torch.save({
                'projector_state_dict': projector.state_dict(),
                'hidden_dim': params['hidden_dim'],
            }, last_model_path)

    print(f"\nSearch Complete! Best overall Val mAP: {best_acc:.4f}")
    
    # Load and return the absolute best projector
    checkpoint = torch.load(best_model_path)
    best_projector = ProjectionHead(DINO_DIM, checkpoint['hidden_dim']).to(device)
    best_projector.load_state_dict(checkpoint['projector_state_dict'])
    
    return best_projector, val_dataset # Return val_dataset to act as the "Base Test" set

# ==========================================
# 6. Few-Shot Prototypical Processing
# ==========================================
def compute_and_save_prototypes(projector, dataloader, device, expected_classes, save_dir, prefix):
    projector.eval()
    hidden_dim = projector.mlp[0].out_features
    class_sums = {cls: torch.zeros(hidden_dim).to(device) for cls in expected_classes}
    class_counts = {cls: 0 for cls in expected_classes}

    with torch.no_grad():
        for features, _, class_names in dataloader:
            features = features.to(device)
            embeddings = projector(features) 
            
            for i, cls in enumerate(class_names):
                if cls in class_sums:
                    class_sums[cls] += embeddings[i]
                    class_counts[cls] += 1

    prototypes = {}
    for cls in expected_classes:
        if class_counts[cls] > 0:
            avg_vec = class_sums[cls] / class_counts[cls]
            proto = F.normalize(avg_vec.unsqueeze(0), p=2, dim=1).squeeze(0)
            prototypes[cls] = proto
            
            # Save individual prototype to specified directory
            save_path = os.path.join(save_dir, f"{prefix}_{cls.replace(' ', '_')}.pt")
            torch.save(proto.cpu(), save_path)
        else:
            print(f"Warning: No samples found to build prototype for {cls}")

    print(f"Saved {len(prototypes)} prototypes to {save_dir}")
    return prototypes

def evaluate_few_shot(projector, prototypes, base_test_dataset, device):
    print("\n--- Evaluating GFSL on Mixed Test Set (Base + Novel) ---")
    projector.eval()
    
    # Load Novel Test data
    novel_test_dataset = EmbeddingDataset(NOVEL_TEST_DIR, NOVEL_CLASSES)
    
    # Combine Base and Novel (Generalized Few Shot setting)
    all_test_data = torch.utils.data.ConcatDataset([base_test_dataset, novel_test_dataset])
    test_loader = DataLoader(all_test_data, batch_size=128, shuffle=False)

    proto_classes = list(prototypes.keys())
    proto_matrix = torch.stack([prototypes[cls] for cls in proto_classes]) # Shape: (Num_Classes, Hidden_Dim)
    
    all_preds = []
    all_trues = []
    all_similarities =[] # Used for mAP calculation

    with torch.no_grad():
        for features, _, class_names in test_loader:
            features = features.to(device)
            embeddings = projector(features)
            
            # Cosine similarity between embeddings and all 10 prototypes
            similarities = torch.matmul(embeddings, proto_matrix.T) 
            
            # Predict
            max_sim_indices = torch.argmax(similarities, dim=1)
            
            for i in range(len(max_sim_indices)):
                all_preds.append(max_sim_indices[i].item())
                # Map true string class to the matching index in proto_classes
                true_idx = proto_classes.index(class_names[i])
                all_trues.append(true_idx)
            
            all_similarities.append(similarities.cpu())

    # Calculate final metrics
    all_trues_np = np.array(all_trues)
    all_preds_np = np.array(all_preds)
    all_sims_tensor = torch.cat(all_similarities, dim=0)
    
    # Convert similarities to probabilities using Softmax for mAP
    probs = F.softmax(all_sims_tensor * 20.0, dim=1).numpy() 
    labels_one_hot = F.one_hot(torch.tensor(all_trues_np), num_classes=len(proto_classes)).numpy()
    
    acc = accuracy_score(all_trues_np, all_preds_np)
    prec, rec, _, _ = precision_recall_fscore_support(all_trues_np, all_preds_np, average='macro', zero_division=0)
    
    try:
        mAP = average_precision_score(labels_one_hot, probs, average='macro')
    except ValueError:
        mAP = 0.0

    print("\n[Final Generalized Few-Shot Test Results]")
    print(f"Overall Accuracy : {acc:.4f}")
    print(f"Macro Precision  : {prec:.4f}")
    print(f"Macro Recall     : {rec:.4f}")
    print(f"Macro mAP        : {mAP:.4f}")

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Hyperparameter Search & Train
    best_projector, base_test_dataset = train_and_search(device)

    # 2. Compute & Save Base Prototypes (Using the whole base training set)
    print("\nComputing Base Prototypes...")
    base_train_loader = DataLoader(EmbeddingDataset(INPUT_BASE_DIR, BASE_CLASSES), batch_size=128)
    base_prototypes = compute_and_save_prototypes(
        best_projector, base_train_loader, device, BASE_CLASSES, BASE_PROTO_DIR, "base"
    )

    # 3. Compute & Save Novel Prototypes (Using the 5-10 few shots)
    print("\nComputing Novel Prototypes from Few-Shots...")
    novel_few_shot_loader = DataLoader(EmbeddingDataset(NOVEL_FEW_SHOT_DIR, NOVEL_CLASSES), batch_size=16)
    novel_prototypes = compute_and_save_prototypes(
        best_projector, novel_few_shot_loader, device, NOVEL_CLASSES, NOVEL_PROTO_DIR, "novel"
    )

    # 4. Merge prototypes
    all_prototypes = {**base_prototypes, **novel_prototypes}

    # 5. Evaluate combined Test sets (Base + Novel)
    evaluate_few_shot(best_projector, all_prototypes, base_test_dataset, device)