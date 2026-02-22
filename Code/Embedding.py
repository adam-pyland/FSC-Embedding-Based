import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from tqdm import tqdm

def main():
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    input_dir = '/home/adamm/Documents/FSOD/codes/FSC-Embedding-Based-Satellites/FSC-Embedding-Based-Satellites/Input_Images/Original_Base_Class_Img_Crops'
    output_dir = 'home/adamm/Documents/FSOD/codes/FSC-Embedding-Based-Satellites/FSC-Embedding-Based-Satellites/Input_Images/DINO_FAIR1M_features/base_classes'
    
    # Update this path to where you saved the satellite model
    checkpoint_path = '/home/adamm/Documents/FSOD/codes/FSC-Embedding-Based-Satellites/FSC-Embedding-Based-Satellites/models/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.hub.set_dir('/home/adamm/Documents/FSOD/codes/FSC-Embedding-Based-Satellites/FSC-Embedding-Based-Satellites/models')

    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # Load The Satellite DINOv3 Model
    # -------------------------------------------------------------------------
    print(f"Loading Satellite DINOv3 from: {checkpoint_path}")
    
    # We load the architecture 'dinov3_vitl16'
    # source='local' assumes you have the repo folder in your cache as discussed before
    # If using online hub, remove source='local' and use 'facebookresearch/dinov3'
    repo_or_dir = 'facebookresearch/dinov3' 
    
    # NOTE: If you are strictly offline, point 'repo_or_dir' to your local unzipped code folder
    
    model = torch.hub.load(repo_or_dir, 'dinov3_vitl16', pretrained=False)
    
    # Load the custom Satellite weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Sometimes checkpoints are saved as {'model': ...} or {'teacher': ...}
    # This handles common DINO checkpoint formats:
    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'teacher' in state_dict:
        state_dict = state_dict['teacher']
        
    # Remove prefix if necessary (e.g. 'module.', 'backbone.')
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Weights loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        
    

    model.to(device)
    model.eval()

    # -------------------------------------------------------------------------
    # Transform & Process
    # -------------------------------------------------------------------------
    transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif'))]
    
    with torch.no_grad():
        for img_file in tqdm(image_files, desc="Embedding Satellite Crops"):
            img_path = os.path.join(input_dir, img_file)
            save_path = os.path.join(output_dir, os.path.splitext(img_file)[0] + '.npy')

            try:
                img = Image.open(img_path).convert('RGB')
                input_tensor = transform(img).unsqueeze(0).to(device)
                
                # Get embedding
                embedding = model(input_tensor)
                
                # Save
                np.save(save_path, embedding.cpu().numpy().flatten())
                
            except Exception as e:
                print(f"Error: {img_file} - {e}")

if __name__ == "__main__":
    main()