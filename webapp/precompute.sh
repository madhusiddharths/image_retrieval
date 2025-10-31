#!/bin/bash

# Create necessary directories
mkdir -p data
mkdir -p weights

# Default paths
MODEL_PATH="weights/model.pth"
DATA_FILE="train_val.json"
FAISS_PATH="data/faiss_index.bin"
PICKLE_PATH="data/features.pickle"
NUM_PER_CLASS=20

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --num-per-class)
      NUM_PER_CLASS="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Display info
echo "Image Retrieval: Precomputing Features"
echo "====================================="
echo "This script will extract features from your training images and"
echo "create a FAISS index for fast similarity search."
echo "For efficiency, only ${NUM_PER_CLASS} random images per class will be used."
echo

# Check if files exist
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Please place your trained model in the weights directory first."
    exit 1
fi

if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: Data file not found at $DATA_FILE"
    echo "Please ensure your train_val.json file is in the root directory."
    exit 1
fi

# Run the precompute script
echo "Starting feature extraction (this may take a while)..."
echo "Using model: $MODEL_PATH"
echo "Using data: $DATA_FILE"
echo "Creating FAISS index at: $FAISS_PATH"
echo "Saving features dictionary at: $PICKLE_PATH"
echo "Using GPU if available (auto-detected)"
echo

python utils/precompute_features.py \
    --model "$MODEL_PATH" \
    --data "$DATA_FILE" \
    --faiss "$FAISS_PATH" \
    --pickle "$PICKLE_PATH" \
    --num-per-class "$NUM_PER_CLASS"

# Check if successful
if [ $? -eq 0 ]; then
    echo
    echo "Feature extraction completed successfully!"
    echo "You can now run the app using ./run_app.sh"
else
    echo
    echo "Feature extraction failed. Please check the error messages above."
fi