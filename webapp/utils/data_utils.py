import json
import os
import numpy as np
import torch
from PIL import Image
import streamlit as st

@st.cache_data
def load_gallery_data(data_path):
    """Load gallery images and features"""
    try:
        # Load the dataset metadata
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                data = json.load(f)
            return data
        else:
            st.warning(f"Data file not found: {data_path}")
            return None
    except Exception as e:
        st.error(f"Error loading gallery data: {e}")
        return None

@st.cache_data
def load_features_dict(dict_path):
    """Load features dictionary where keys are image paths and values are features"""
    try:
        if os.path.exists(dict_path):
            with open(dict_path, 'r') as f:
                features_dict = json.load(f)
            return features_dict
        else:
            return None
    except Exception as e:
        st.error(f"Error loading features dictionary: {e}")
        return None

def generate_demo_data(num_samples=100, feature_dim=128, num_classes=10):
    """Generate dummy data for demo purposes"""
    np.random.seed(42)  # For reproducibility
    
    # Generate random feature vectors
    features = np.random.randn(num_samples, feature_dim).astype(np.float32)
    # Normalize features to unit length (like real embeddings)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    
    # Generate random labels
    labels = np.random.randint(0, num_classes, size=num_samples)
    
    # Create some demo categories
    categories = [
        "cat", "dog", "car", "flower", "building", 
        "mountain", "beach", "food", "person", "art"
    ]
    
    # Create dummy paths (we'll use placeholder images)
    image_paths = [f"demo_image_{i}.jpg" for i in range(num_samples)]
    
    # Create gallery data structure
    gallery_data = {
        "train": [(int(labels[i]), image_paths[i]) for i in range(num_samples)],
        "val": [],  # Not needed for demo
        "categories": categories
    }
    
    return gallery_data, features