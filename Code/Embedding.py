import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

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
# 2. Custom Dataset Class (Handles keeping the filenames tied to the images)
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
            return img, img_name  # Return BOTH the image tensor and the original filename
        
        except Exception as e:
            # If an image is corrupt, return a dummy tensor and an error string
            print(f"Warning: Could not load {img_name} - {e}")
            return torch.zeros((3, 224, 224)), f"ERROR_{img_name}"

# -------------------------------------------------------------------------
# 3. Main Processing Function
# -------------------------------------------------------------------------
def main():
    input_dir = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Filtered_Dataset/Image_Obj_Crops/train/'
    output_dir = '/home/adamm/Documents/FSOD/Data/FAIR1M/new_attempt/archive/Filtered_Dataset/Image_Obj_Embs/train/'
    checkpoint_path = '/home/adamm/Documents/FSOD/codes/FSC-Embedding-Based-Satellites/FSC-Embedding-Based-Satellites/models/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'

    # Hyperparameters for speed
    BATCH_SIZE = 128   # Increase to 128 or 256 if your GPU has a lot of VRAM
    NUM_WORKERS = 8   # Helps load images faster using CPU threads

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
    
    # DataLoader handles the batching and parallel loading automatically
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)

    # --- Extract Features in Batches ---
    with torch.no_grad():
        # Wrap the dataloader in tqdm for a progress bar
        for batch_images, batch_filenames in tqdm(dataloader, desc="Embedding Batches"):
            
            # Move the whole batch of images to the GPU at once
            batch_images = batch_images.to(device)
            
            # Pass the whole batch through the model
            # This makes the forward pass use 16-bit math, which is much faster.
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                batch_embeddings = model(batch_images)
            
            # Move embeddings back to CPU as a numpy array
            batch_embeddings = batch_embeddings.cpu().numpy()
            
            # Save each embedding to its corresponding filename
            for i in range(len(batch_filenames)):
                filename = batch_filenames[i]
                
                # Skip saving if the image was corrupt
                if filename.startswith("ERROR_"):
                    continue
                
                # Replace the original extension (.jpg/.png) with .npy
                save_name = os.path.splitext(filename)[0] + '.npy'
                save_path = os.path.join(output_dir, save_name)
                
                # Get the specific 1D feature vector for this image
                single_feature_vector = batch_embeddings[i].flatten()
                
                # Save
                np.save(save_path, single_feature_vector)

if __name__ == "__main__":
    main()