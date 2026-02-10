import os
import subprocess
# Set OpenMP environment variable before importing any libraries that use it
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# --- Add this block ---
# Ensure Git LFS files (like model.pth) are pulled on deployment
# Only run if model file is missing or looks like a pointer file (< 1KB)
model_path_check = "weights/model.pth"
if not os.path.exists(model_path_check) or os.path.getsize(model_path_check) < 1024:
    print(f"Checking model at {model_path_check}...")
    if os.path.exists(model_path_check):
        print(f"File exists but is small: {os.path.getsize(model_path_check)} bytes")
    else:
        print("File does not exist.")

    print("Attempting to pull LFS files...")
    try:
        subprocess.run(["git", "lfs", "install"], check=True)
        result = subprocess.run(["git", "lfs", "pull"], check=True, capture_output=True, text=True)
        print("Git LFS pull command executed.")
        print("Output:", result.stdout)
    except Exception as e:
        print(f"Git LFS failed: {e}")

    # Verify again
    if os.path.exists(model_path_check):
        print(f"Post-pull size: {os.path.getsize(model_path_check)} bytes")
    else:
        print("Post-pull: File still missing.")
# -----------------------

import streamlit as st
import torch
import json
import numpy as np
from PIL import Image

# Import our modules
from src.model import ResNetTransferModel
from utils import (
    preprocess_image, extract_features,
    build_faiss_index, load_faiss_index, save_faiss_index, search_similar_images,
    display_results
)

# Set page configuration
st.set_page_config(
    page_title="Image Retrieval Demo", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔍"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        font-size: 1.2rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    .stButton>button {
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    /* Search button - Green (apply to primary buttons in second column) */
    div[data-testid="column"]:nth-child(2) button[kind="primaryFormSubmit"] {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%) !important;
    }
    div[data-testid="column"]:nth-child(2) button[kind="primaryFormSubmit"]:hover {
        background: linear-gradient(90deg, #059669 0%, #047857 100%) !important;
        box-shadow: 0 5px 15px rgba(16, 185, 129, 0.4) !important;
    }
    /* Clear button - Red (apply to buttons in third column) */
    div[data-testid="column"]:nth-child(3) button {
        background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%) !important;
    }
    div[data-testid="column"]:nth-child(3) button:hover {
        background: linear-gradient(90deg, #dc2626 0%, #b91c1c 100%) !important;
        box-shadow: 0 5px 15px rgba(239, 68, 68, 0.4) !important;
    }
    .info-box {
        background: #f0f4f8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
</style>
""", unsafe_allow_html=True)

# File paths (default)
MODEL_DIR = "weights"
DATA_DIR = "data"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pth")
DEFAULT_FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
DEFAULT_FEATURES_PATHS_FILE = os.path.join(DATA_DIR, "features_paths.json")

@st.cache_resource
def load_model(model_path, num_classes=101):
    """Load the trained model"""
    # Determine device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ResNetTransferModel(num_classes=num_classes, embedding_size=128, pretrained=False).to(device)
    
    try:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            st.error(f"Model not found at {model_path}. Please run precompute.sh first.")
            return None, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device
    
    model.eval()
    return model, device

def main():
    # Set up header with custom styling
    st.markdown('<h1 class="main-header">🔍 AI Powered Image Retrieval</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Find similar images using deep learning and vector databases</p>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.markdown("## ⚙️ Settings")
    
    # Custom path inputs
    st.sidebar.markdown("### 🔗 Custom Paths")
    faiss_index_path = st.sidebar.text_input("FAISS Index Path", value=DEFAULT_FAISS_INDEX_PATH, help="Path to the FAISS index file")
    features_paths_file = st.sidebar.text_input("Features Paths File", value=DEFAULT_FEATURES_PATHS_FILE, help="Path to features metadata")
    
    # Settings
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Search Settings")
    num_results = st.sidebar.slider("Number of Results", min_value=1, max_value=10, value=5, help="How many similar images to retrieve")
    show_all_categories = st.sidebar.checkbox("Show All Categories", value=False, help="Display all available categories in sidebar")
    
    # Upload image section
    st.markdown("---")
    st.markdown("## 📤 Upload Your Image")
    st.markdown("Upload an image and find similar ones from the database")
    
    # Create columns for upload and buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], help="Supported formats: JPG, JPEG, PNG")
        
    with col2:
        search_button = st.button("🔎 Search", use_container_width=True, type="primary")
        
    with col3:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    # Initialize session state for results
    if 'results_displayed' not in st.session_state:
        st.session_state.results_displayed = False
        
    # Display all categories if requested
    if show_all_categories and os.path.exists(features_paths_file):
        try:
            with open(features_paths_file, 'r') as f:
                indexed_paths = json.load(f)
                
            # Extract all unique categories
            categories = set()
            for item in indexed_paths:
                if "category" in item:
                    categories.add(item["category"])
            
            # Display categories
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 📂 Available Categories")
            st.sidebar.caption(f"Total: {len(categories)} categories")
            
            # Create a grid display for categories
            category_list = sorted(list(categories))
            rows = [category_list[i:i+2] for i in range(0, len(category_list), 2)]
            
            for row in rows:
                cols = st.sidebar.columns(2)
                for i, category in enumerate(row):
                    cols[i].markdown(f"• `{category}`")
        except Exception as e:
            st.sidebar.warning(f"Could not load categories: {e}")
    
    # Clear button logic
    if clear_button:
        # Reset all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Only load model and data if needed
    if uploaded_file is not None or search_button:
        # Load model and required data
        model, device = load_model(MODEL_PATH)
        if not model:
            st.error("❌ Model not found. Please run precompute.sh first.")
            st.stop()
        
        # Try to load FAISS index
        faiss_index = load_faiss_index(faiss_index_path)
        if faiss_index is None:
            st.error(f"❌ FAISS index not found at {faiss_index_path}. Please run precompute.sh first.")
            st.stop()
            
        # Try to load paths with metadata
        if os.path.exists(features_paths_file):
            try:
                with open(features_paths_file, 'r') as f:
                    indexed_paths = json.load(f)
                # Display system info in sidebar
                st.sidebar.markdown("---")
                st.sidebar.markdown("### ℹ️ System Information")
                st.sidebar.caption(f"🖥️ Device: {device}")
                st.sidebar.success(f"✅ Index ready: {len(indexed_paths)} images")
            except Exception as e:
                st.warning(f"Error loading features paths: {e}")
                indexed_paths = None
        else:
            st.warning(f"Features paths file not found at {features_paths_file}. Results may not be accurate.")
            indexed_paths = None
        
        # Check if we have paths data
        if not indexed_paths:
            st.error("❌ Required files not found. Please run precompute.sh first.")
            st.stop()
        
        # Only perform search when button is clicked AND there's an uploaded file
        if search_button and uploaded_file:
            # Process image and display results
            query_image = Image.open(uploaded_file).convert('RGB')
            
            # Display the uploaded image in a container
            st.markdown("---")
            st.markdown("## 📷 Your Uploaded Image")
            with st.container():
                col_img, col_info = st.columns([2, 1])
                with col_img:
                    st.image(query_image, caption="Query Image", width=300)
                with col_info:
                    st.markdown("### Image Info")
                    st.text(f"Size: {query_image.size[0]} x {query_image.size[1]}")
                    st.text(f"Mode: {query_image.mode}")
            
            # Process the query image
            with st.spinner("🔍 Searching for similar images..."):
                # Process the query image with the model
                query_tensor = preprocess_image(query_image, device)
                query_feature = extract_features(model, query_tensor, device)
                
                # Search for similar images
                similarities, indices = search_similar_images(query_feature, faiss_index, k=num_results)
                
                # Display results
                display_results(similarities, indices, indexed_paths)
                
                # Mark that we've displayed results
                st.session_state.results_displayed = True
        elif search_button and not uploaded_file:
            st.warning("⚠️ Please upload an image first before searching.")
    
    # Footer
    st.markdown("---")
    

if __name__ == "__main__":
    main()
