#!/usr/bin/env python3
import os
import argparse
import numpy as np
from pathlib import Path
from typing import List, Optional
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm


class ResNet50FeatureExtractor:
    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        resnet = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Initialized on {self.device} | Feature dim: 2048")
    
    def extract_batch(self, image_paths: List[str], batch_size: int = 32) -> List[np.ndarray]:
        all_features = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting"):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            for img_path in batch_paths:
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = self.preprocess(image)
                    batch_images.append(image_tensor)
                except Exception as e:
                    print(f"Error {img_path}: {e}")
                    continue
            
            if not batch_images:
                continue
            
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            with torch.no_grad():
                features = self.model(batch_tensor)
            
            features = features.squeeze().cpu().numpy()
            if len(batch_images) == 1:
                features = features.reshape(1, -1)
            
            all_features.extend(features)
        
        return all_features


def get_image_files(directory: str) -> List[str]:
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF')
    image_paths = []
    for ext in extensions:
        image_paths.extend(Path(directory).rglob(f'*{ext}'))
    return [str(p) for p in sorted(image_paths)]


def main():
    parser = argparse.ArgumentParser(description='Extract ResNet50 features (2048-dim)')
    
    parser.add_argument('--input_dir', type=str, default='/work/Sultan/data/for_resnet/1/', help='Directory with images')
    parser.add_argument('--input_image', type=str, default=None, help='Single image file')
    parser.add_argument('--output_dir', type=str, default='/work/Sultan/data/feature_vec_1/', help='Output directory')
    parser.add_argument('--save_average', action='store_true', default=True, help='Save averaged features')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu', None])
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = output_dir / 'individual_features'
    features_dir.mkdir(exist_ok=True)
    
    extractor = ResNet50FeatureExtractor(device=args.device)
    
    if args.input_image is not None:
        image_paths = [args.input_image]
    else:
        image_paths = get_image_files(args.input_dir)
        if not image_paths:
            print(f"No images found in {args.input_dir}")
            return
        print(f"Found {len(image_paths)} images")
    
    all_features = extractor.extract_batch(image_paths, batch_size=args.batch_size)
    
    for img_path, features in zip(image_paths, all_features):
        img_name = Path(img_path).stem
        np.save(features_dir / f"{img_name}_features.npy", features)
    
    print(f"Saved {len(all_features)} features to {features_dir}")
    
    if args.save_average and len(all_features) > 0:
        features_array = np.array(all_features)
        avg_features = np.mean(features_array, axis=0)
        avg_path = output_dir / 'averaged_features.npy'
        np.save(avg_path, avg_features)
        print(f"Saved averaged features to {avg_path} | Shape: {avg_features.shape}")


if __name__ == '__main__':
    main()
