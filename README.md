# Image Retrieval

A compact image retrieval demo built with Streamlit. This repository contains the app, indexing utilities, and example scripts to compute image embeddings, build a similarity index, and run a visual search demo.

---

## Features
- Streamlit-based UI for interactive image search
- Scripts to compute embeddings and build a searchable index
- Docker support for easy deployment
- Simple pipeline for adding new images and reindexing

## Quick Links
- App entry: `src/streamlit_app.py`
- Port (default): `8501` (configured for Docker in metadata)
- SDK: Docker

## Repository layout
- src/ - Streamlit app and main application code
- notebooks/ - exploratory notebooks (if present)
- data/ - raw and processed image assets (not included)
- models/ - pretrained models or checkpoints (optional)
- scripts/ - indexing, embedding, and utility scripts
- README.md - this file

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
   - pip install -r requirements.txt
   (If requirements.txt is not present, install packages used by your app, e.g., streamlit, torch, torchvision, faiss-cpu, pillow, numpy.)

3. Run the Streamlit app:
   - streamlit run src/streamlit_app.py --server.port 8501

Open http://localhost:8501 in your browser.

## Docker (optional)
Build:
- docker build -t image-retrieval:latest .

Run:
- docker run -p 8501:8501 image-retrieval:latest

Adjust Dockerfile and port mapping to match your environment.

## Data and Indexing
1. Place images in a `data/images/` directory (or point scripts to your dataset).

2. Run the precompute script to compute embeddings and build an index (required before starting the app):
   - Make the script executable:
     - chmod +x scripts/precompute.sh
   - Run the script (example):
     - ./scripts/precompute.sh --images data/images --out models/index.faiss
   Note: some repositories may name the script `scripts/precoumpute.sh`; if so, use that name instead.

3. (Optional) Compute embeddings manually:
   - Use the provided script (e.g., `scripts/compute_embeddings.py`) to convert images to vectors.

4. Build or verify the similarity index:
   - If the precompute script does not build the index, run:
     - python scripts/build_index.py --embeddings <path> --index_out <path>

5. Point the Streamlit app to the index and image folder (via config or environment variables).

