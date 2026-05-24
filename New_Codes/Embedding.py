import os
import argparse
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import time

# -------------------------------------------------------------------------
# 1. Padding Class
# -------------------------------------------------------------------------
class LetterboxPad:
    def __init__(self, target_size=224, fill=0):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        if w > self.target_size or h > self.target_size:
            img.thumbnail((self.target_size, self.target_size), Image.Resampling.BICUBIC)
            w, h = img.size 
            
        delta_w = self.target_size - w
        delta_h = self.target_size - h
        
        padding = (
            delta_w // 2, 
            delta_h // 2, 
            delta_w - (delta_w // 2), 
            delta_h - (delta_h // 2)
        )
        return TF.pad(img, padding, fill=self.fill, padding_mode='constant')

# -------------------------------------------------------------------------
# 2. Custom Dataset Class
# -------------------------------------------------------------------------
class SatelliteCropDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # Get all valid image files
        self.image_files =[f for f in os.listdir(image_dir) 
                            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, img_name  
        
        except Exception as e:
            print(f"Warning: Could not load {img_name} - {e}")
            return torch.zeros((3, 224, 224)), f"ERROR_{img_name}"

# -------------------------------------------------------------------------
# 3. Main Processing Function
# -------------------------------------------------------------------------
def main():
    # Setup argparse
    parser = argparse.ArgumentParser(description="Extract embeddings from satellite object crops using DINO.")
    
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default='/home/adamm/Documents/FSOD/Data/Lavyanut/Images/test/Obj_Crops/',
        help='Directory containing the object image crops.'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='/home/adamm/Documents/FSOD/Data/Lavyanut/Images/test/',
        help='Directory where the resulting .npy embeddings will be saved.'
    )
    parser.add_argument(
        '--checkpoint_path', 
        type=str, 
        default='/home/adamm/Documents/FSOD/codes/Object_Sub_Classification/models/DINOv3/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
        help='Path to the pretrained DINO model checkpoint.'
    )
    
    args = parser.parse_args()

    # Assign args to variables
    input_dir = args.input_dir
    output_dir = args.output_dir + 'Obj_Embs/'
    checkpoint_path = args.checkpoint_path

    tic = time.time()

    # Hyperparameters for speed
    BATCH_SIZE = 128   
    NUM_WORKERS = 8   

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.hub.set_dir('/home/adamm/Documents/FSOD/codes/FSC-Embedding-Based-Satellites/FSC-Embedding-Based-Satellites/models')

    print(f"Using device: {device}")
    
    # --- Load Model ---
    repo_or_dir = 'facebookresearch/dinov3' 
    model = torch.hub.load(repo_or_dir, 'dinov3_vitl16', pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'teacher' in state_dict:
        state_dict = state_dict['teacher']
        
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()

    model = torch.compile(model)

    # --- Setup Transforms and DataLoader ---
    transform = T.Compose([
        LetterboxPad(target_size=224), 
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    dataset = SatelliteCropDataset(input_dir, transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)

    # --- Extract Features in Batches ---
    with torch.no_grad():
        for batch_images, batch_filenames in tqdm(dataloader, desc="Embedding Batches"):
            
            batch_images = batch_images.to(device)
            
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                batch_embeddings = model(batch_images)
            
            batch_embeddings = batch_embeddings.cpu().numpy()
            
            for i in range(len(batch_filenames)):
                filename = batch_filenames[i]
                
                if filename.startswith("ERROR_"):
                    continue
                
                save_name = os.path.splitext(filename)[0] + '.npy'
                save_path = os.path.join(output_dir, save_name)
                
                single_feature_vector = batch_embeddings[i].flatten()
                
                np.save(save_path, single_feature_vector)
                
    toc = time.time()
    tictoc = toc - tic
    print(f"Measured block execution time: {tictoc:.4f} seconds")

if __name__ == "__main__":
    main()