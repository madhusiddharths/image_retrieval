import streamlit as st
import os
import json
from PIL import Image
import requests
from io import BytesIO
import random

def display_results(similarities, indices, indexed_paths, placeholder_mode=False):
    """
    Display search results in a streamlit UI
    
    Parameters:
    -----------
    similarities : numpy array
        Similarity scores for each result
    indices : numpy array
        Indices of the retrieved images in the FAISS index
    indexed_paths : list of dicts
        List of dictionaries containing image paths and metadata
    placeholder_mode : bool, optional
        Whether to use placeholder images instead of real ones
    """
    st.header("Retrieved Similar Images")
    
    # Determine number of columns based on results
    num_columns = min(5, len(indices[0]))
    cols = st.columns(num_columns)
    
    for i, (idx, score) in enumerate(zip(indices[0], similarities[0])):
        col_idx = i % num_columns  # This ensures we wrap to next row after filling columns
        
        with cols[col_idx]:
            try:
                if not placeholder_mode and indexed_paths and idx < len(indexed_paths):
                    # Get the path and metadata for this index
                    item = indexed_paths[idx]
                    img_path = item["path"]
                    category = item.get("category", "Unknown")
                    
                    # Try to load and display the image
                    paths_to_try = [
                        img_path,  # Original path as stored
                        img_path.replace("caltech101/", ""),  # Without caltech101 prefix
                        os.path.join("caltech101", img_path)  # With caltech101 prefix
                    ]
                    
                    image_found = False
                    for path in paths_to_try:
                        if os.path.exists(path):
                            image = Image.open(path).convert('RGB')
                            
                            # Create a cleaner caption
                            caption = f"{category}"
                            if score >= 0:
                                caption += f" (Score: {score:.2f})"
                                
                            st.image(image, caption=caption)
                            image_found = True
                            break
                            
                    if not image_found:
                        # If file doesn't exist, use a placeholder
                        st.warning(f"Image not found")
                        display_placeholder_image(i, score, caption=f"{category} (Score: {score:.2f})")
                else:
                    # Fallback to placeholder
                    display_placeholder_image(i, score, caption=f"Result {i+1} (Score: {score:.2f})")
            except Exception as e:
                st.error(f"Error loading result {i+1}")

def display_placeholder_image(index, score=None, caption=None):
    """Display a placeholder image from Lorem Picsum"""
    try:
        img_id = (237 + index * 10) % 1000  # Generate different IDs
        response = requests.get(f"https://picsum.photos/id/{img_id}/300/200")
        image = Image.open(BytesIO(response.content))
        
        if caption is None:
            caption = f"Similar Image {index+1}"
            if score is not None:
                caption += f" (Score: {score:.2f})"
                
        st.image(image, caption=caption)
    except Exception as e:
        st.error(f"Error loading placeholder image")