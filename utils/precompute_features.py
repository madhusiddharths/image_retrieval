"""
Utility script to precompute and save feature vectors for the gallery images.
This is useful for large datasets where computing features on-the-fly would be slow.

The script selects a specified number of random images per category (default: 10)
and creates a FAISS index for fast similarity search. It also saves a mapping
between FAISS indices and the original image paths.

Usage:
    python precompute_features.py --model weights/model.pth --data train_val.json --faiss data/faiss_index.bin --num-per-class 10 --pickle data/features.pickle
"""

import argparse
import torch
import os
import json
import numpy as np
import pickle
from PIL import Image
from tqdm import tqdm
import sys
import random

# Add the parent directory to the path so we can import our modules
sys.path.append(".")
from src.model import ResNetTransferModel
from utils.image_utils import preprocess_image, extract_features
from utils.faiss_utils import build_faiss_index, save_faiss_index

def main():
    parser = argparse.ArgumentParser(description='Precompute features for gallery images')
    parser.add_argument('--model', type=str, required=True, help='Path to the model file')
    parser.add_argument('--data', type=str, required=True, help='Path to the gallery data JSON file')
    parser.add_argument('--faiss', type=str, required=True, help='Path to save the FAISS index')
    parser.add_argument('--pickle', type=str, required=True, help='Path to save features as a pickle dictionary')
    parser.add_argument('--num-per-class', type=int, default=10, help='Number of images per class to use (default: 10)')
    args = parser.parse_args()
    
    # Check if paths exist
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return
    
    if not os.path.exists(args.data):
        print(f"Error: Data file not found at {args.data}")
        return
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(args.faiss), exist_ok=True)
    os.makedirs(os.path.dirname(args.pickle), exist_ok=True)
    
    # Auto-detect and use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Load the model
    model = ResNetTransferModel(num_classes=101, embedding_size=128, pretrained=False).to(device)
    checkpoint = torch.load(args.model, map_location=device)
    # Handle both checkpoint formats (state_dict or dict with 'model_state_dict' key)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded successfully")
    
    # Load gallery data
    with open(args.data, 'r') as f:
        gallery_data = json.load(f)
    
    # Verify the structure
    if 'train' not in gallery_data or 'categories' not in gallery_data:
        print("Error: The data file must have 'train' and 'categories' keys")
        return
    
    categories = gallery_data['categories']
    print(f"Found {len(categories)} categories")
    
    # Group images by category
    images_by_category = {}
    for i, (label, img_path) in enumerate(gallery_data['train']):
        if label not in images_by_category:
            images_by_category[label] = []
        images_by_category[label].append((i, img_path))
    
    # Select random images from each category
    selected_images = []
    random.seed(42)  # For reproducibility
    
    for label, images in images_by_category.items():
        # Determine how many to select
        num_to_select = min(args.num_per_class, len(images))
        
        # Randomly select
        if len(images) > num_to_select:
            selected = random.sample(images, num_to_select)
        else:
            selected = images
        
        selected_images.extend(selected)
        
        # Get category name for display
        category_name = categories[label] if label < len(categories) else f"Category {label}"
        print(f"Category {category_name}: Selected {num_to_select} of {len(images)} images")
    
    print(f"\nProcessing {len(selected_images)} images for FAISS index...")
    
    # Extract features from selected images
    all_features = []
    all_paths = []
    features_dict = {}  # For pickle output format
    
    for i, (orig_idx, img_path) in enumerate(tqdm(selected_images)):
        try:
            # Fix path if needed (relative paths in caltech101 folder)
            full_path = img_path
            if not os.path.exists(full_path):
                # Try with caltech101 prefix
                caltech_path = os.path.join("caltech101", img_path)
                if os.path.exists(caltech_path):
                    full_path = caltech_path
                else:
                    print(f"Warning: Could not find image at {img_path} or {caltech_path}")
                    continue
            
            # Load and preprocess the image
            image = Image.open(full_path).convert('RGB')
            image_tensor = preprocess_image(image, device)
            
            # Extract features
            features = extract_features(model, image_tensor, device)
            
            # Store features and path
            all_features.append(features)
            all_paths.append(full_path)
            
            # Add to dictionary format for pickle
            features_dict[full_path] = features[0]  # Store the flattened feature vector
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Stack all features
    if all_features:
        all_features = np.vstack(all_features)
        print(f"Extracted features shape: {all_features.shape}")
        
        # Save paths and metadata array (essential for mapping FAISS indices back to images)
        # Use a consistent name for the paths file
        paths_file = os.path.join(os.path.dirname(args.faiss), "features_paths.json")
        
        # Create a list of dictionaries with path and category information
        paths_with_metadata = []
        categories = gallery_data['categories']
        
        for i, path in enumerate(all_paths):
            # Find the original image in the gallery data to get its label
            label = None
            for train_label, train_path in gallery_data['train']:
                if train_path == path or os.path.join("caltech101", train_path) == path:
                    label = train_label
                    break
            
            if label is not None and label < len(categories):
                category_name = categories[label]
            else:
                category_name = "Unknown"
            
            # Store path with metadata
            paths_with_metadata.append({
                "path": path,
                "label": label,
                "category": category_name
            })
        
        with open(paths_file, 'w') as f:
            json.dump(paths_with_metadata, f)
        print(f"Image paths with metadata saved to {paths_file} (contains {len(paths_with_metadata)} entries)")
        
        # Create and save FAISS index (required for inference)
        index = build_faiss_index(all_features)
        save_faiss_index(index, args.faiss)
        print(f"FAISS index saved to {args.faiss}")
        
        # Save features as pickle dictionary if requested
        if args.pickle:
            with open(args.pickle, 'wb') as f:
                pickle.dump(features_dict, f)
            print(f"Features dictionary saved to {args.pickle} (contains {len(features_dict)} entries)")
    else:
        print("No features extracted. Check that the image paths in the data file are correct.")

if __name__ == "__main__":
    main()
