import faiss
import numpy as np
import pickle
import os
import torch

def build_faiss_index(features, feature_dim=128):
    """
    Build FAISS index for fast similarity search
    
    Parameters:
    -----------
    features : numpy.ndarray
        Feature vectors to index
    feature_dim : int
        Feature dimension
    
    Returns:
    --------
    faiss.Index
        FAISS index containing the features
    """
    # Create index
    index = faiss.IndexFlatIP(feature_dim)  # Inner product (cosine similarity for normalized vectors)
    
    # Use GPU resources if available
    if torch.cuda.is_available():
        try:
            print("Using GPU for FAISS indexing")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception as e:
            print(f"Error using GPU for FAISS: {e}")
            print("Falling back to CPU")
    
    # Add features to index
    if len(features) > 0:
        index.add(features)
    
    # Convert back to CPU index for storage
    if torch.cuda.is_available() and hasattr(index, 'index'):
        index = faiss.index_gpu_to_cpu(index)
    
    return index

def save_faiss_index(index, file_path):
    """Save FAISS index to disk"""
    faiss.write_index(index, file_path)
    
def load_faiss_index(file_path):
    """
    Load FAISS index from disk
    
    Parameters:
    -----------
    file_path : str
        Path to the FAISS index file
    
    Returns:
    --------
    faiss.Index or None
        FAISS index loaded from disk, or None if file doesn't exist
    """
    if not os.path.exists(file_path):
        return None
    
    # Load index
    index = faiss.read_index(file_path)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        try:
            print("Using GPU for FAISS search")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception as e:
            print(f"Error using GPU for FAISS: {e}")
            print("Falling back to CPU")
    
    return index

def search_similar_images(query_feature, faiss_index, k=5):
    """
    Search for similar images using FAISS
    
    Parameters:
    -----------
    query_feature : numpy.ndarray
        Feature vector of the query image
    faiss_index : faiss.Index
        FAISS index containing gallery features
    k : int
        Number of results to return
    
    Returns:
    --------
    tuple of (numpy.ndarray, numpy.ndarray)
        Similarities and indices of the most similar images
    """
    # Search for k most similar images
    similarities, indices = faiss_index.search(query_feature, k)
    return similarities, indices