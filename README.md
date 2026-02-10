# Image Retrieval

A compact image retrieval demo built with Streamlit. This repository contains the app, indexing utilities, and example scripts to compute image embeddings, build a similarity index, and run a visual search demo.

---

## Features
- Streamlit-based UI for interactive image search
- Scripts to compute embeddings and build a searchable index
- Docker support for easy deployment
- Simple pipeline for adding new images and reindexing

## Quick Links
- App entry: `app.py`
- Port (default): `8501`
- Run: `streamlit run app.py`

## Repository layout
- `app.py` - Main Streamlit application
- `src/` - Model and core application code
- `utils/` - Utility modules for data processing, image handling, and feature computation
- `data/` - Precomputed features and FAISS index
- `weights/` - Pretrained model weights
- `caltech101/` - Dataset directory
- `requirements.txt` - Python dependencies
- `Procfile` - Deployment configuration for Streamlit Cloud/Heroku

## Prerequisites
- Python 3.8+
- pip
- (Optional) Docker

## Local setup (recommended)
1. Create and activate a virtual environment:
   - python -m venv .venv
   - source .venv/bin/activate  # macOS / Linux
   - .venv\Scripts\activate     # Windows

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

Open http://localhost:8501 in your browser.

## Deployment

### Streamlit Cloud
1. Push to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Create a new app and point to this repository and `app.py`
4. Set up any required secrets in the app settings

### GitHub Pages / Other Platforms
Refer to platform-specific documentation for deploying Streamlit applications.

## Notes
- The app precomputes features and builds a FAISS index on startup
- Model weights are managed via Git LFS (Git Large File Storage)
- Ensure all required dependencies are installed before running

