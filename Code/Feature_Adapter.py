import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Configuration & Paths
# ==========================================
# INPUT DIRECTORIES (Your current DINO features)
INPUT_BASE_DIR = '/home/adamm/Documents/FSOD/codes/FSC-Embedding-Based-Satellites/FSC-Embedding-Based-Satellites/Input_Images/DINO_FAIR1M_features/base_classes'
INPUT_NOVEL_DIR = '/home/adamm/Documents/FSOD/codes/FSC-Embedding-Based-Satellites/FSC-Embedding-Based-Satellites/Input_Images/DINO_FAIR1M_features/novel_classes'

# OUTPUT DIRECTORIES (Where the new 512-D features will be saved)
OUTPUT_BASE_DIR = '/home/adamm/Documents/FSOD/codes/FSC-Embedding-Based-Satellites/FSC-Embedding-Based-Satellites/Input_Images/Adapted_Features/base_classes'
OUTPUT_NOVEL_DIR = '/home/adamm/Documents/FSOD/codes/FSC-Embedding-Based-Satellites/FSC-Embedding-Based-Satellites/Input_Images/Adapted_Features/novel_classes'

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
os.makedirs(OUTPUT_NOVEL_DIR, exist_ok=True)

BASE_CLASSES = ['Small Car', 'Bus', 'Dump Truck', 'Van', 'Tractor', 'Truck Tractor', 'Excavator', 'other-vehicle']
NOVEL_CLASSES = ['Cargo Truck', 'Trailer']

# Create safe names for string matching
SAFE_BASE_CLASSES = [c.replace(' ', '_') for c in BASE_CLASSES]
SAFE_NOVEL_CLASSES = [c.replace(' ', '_') for c in NOVEL_CLASSES]

# Map base classes to integer IDs for training (0 to 7)
CLASS_TO_ID = {cls: i for i, cls in enumerate(SAFE_BASE_CLASSES)}

# Determine input dimension by loading one file
sample_file = glob.glob(os.path.join(INPUT_BASE_DIR, '*.npy'))[0]
DINO_DIM = np.load(sample_file).flatten().shape[0] # Usually 768 or 1024
HIDDEN_DIM = 512

# ==========================================
# 2. Dataset Definition
# ==========================================
class FeatureDataset(Dataset):
    def __init__(self, file_paths, class_to_id, safe_classes):
        self.file_paths = file_paths
        self.class_to_id = class_to_id
        self.safe_classes = safe_classes
        
        # Pre-filter files to make sure we only load valid classes
        self.valid_files = []
        self.labels = []
        
        for f in file_paths:
            filename = os.path.basename(f)
            for cls in self.safe_classes:
                if f"_{cls}_" in filename:
                    self.valid_files.append(f)
                    self.labels.append(self.class_to_id[cls])
                    break

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        path = self.valid_files[idx]
        label = self.labels[idx]
        feature = np.load(path).flatten()
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ==========================================
# 3. Model Definition
# ==========================================
class FeatureAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(FeatureAdapter, self).__init__()
        # The embedding layers (we keep these for few-shot later)
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        # The classification layer (we throw this away later)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, return_embedding=False):
        embed = self.embedding(x)
        if return_embedding:
            return embed
        return self.classifier(embed)

# ==========================================
# 4. Training the Adapter
# ==========================================
def train_adapter():
    all_base_files = glob.glob(os.path.join(INPUT_BASE_DIR, '*.npy'))
    
    # Split into train/val to monitor overfitting
    train_files, val_files = train_test_split(all_base_files, test_size=0.15, random_state=42)
    
    train_dataset = FeatureDataset(train_files, CLASS_TO_ID, SAFE_BASE_CLASSES)
    val_dataset = FeatureDataset(val_files, CLASS_TO_ID, SAFE_BASE_CLASSES)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = FeatureAdapter(input_dim=DINO_DIM, hidden_dim=HIDDEN_DIM, num_classes=len(BASE_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    EPOCHS = 30
    best_val_loss = float('inf')

    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features, return_embedding=False)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] - "
              f"Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/best_feature_adapter.pth')

    print("Training Complete. Best model saved as 'best_feature_adapter.pth'.")
    return device

# ==========================================
# 5. Inference: Generate and Save New Features
# ==========================================
def extract_and_save_new_features(device):
    print("\n--- Starting Feature Extraction Phase ---")
    
    # Load the best trained model
    model = FeatureAdapter(input_dim=DINO_DIM, hidden_dim=HIDDEN_DIM, num_classes=len(BASE_CLASSES)).to(device)
    model.load_state_dict(torch.load('best_feature_adapter.pth'))
    model.eval() # Set to evaluation mode

    def process_directory(input_dir, output_dir, label_type):
        files = glob.glob(os.path.join(input_dir, '*.npy'))
        print(f"Processing {len(files)} {label_type} files...")
        
        with torch.no_grad():
            for f in files:
                filename = os.path.basename(f)
                
                # Load old feature
                old_feature = np.load(f).flatten()
                
                # Convert to tensor and pass through model (ONLY the embedding part)
                feature_tensor = torch.tensor(old_feature, dtype=torch.float32).unsqueeze(0).to(device)
                new_feature = model(feature_tensor, return_embedding=True)
                
                # Convert back to numpy and save
                new_feature_np = new_feature.cpu().numpy().flatten()
                
                save_path = os.path.join(output_dir, filename)
                np.save(save_path, new_feature_np)

    process_directory(INPUT_BASE_DIR, OUTPUT_BASE_DIR, "Base Class")
    process_directory(INPUT_NOVEL_DIR, OUTPUT_NOVEL_DIR, "Novel Class")
    
    print("All new adapted features have been successfully saved!")

# ==========================================
# RUN THE SCRIPT
# ==========================================
if __name__ == "__main__":
    device = train_adapter()
    extract_and_save_new_features(device)