#!/bin/bash

# Define the input CSV files
# This script serves as a tool to assist in testing embedding models with any input data file.
# In the example below, "false_positive_identification_dataset.csv" is utilised.

INPUT_FILES=(
              "./embedding_models/english_fastText_CBOW/fastText_train_epochs_trained_10_false_positive_classifier_radio_broadcast_news_CBOW_embedding_size_300")

# Function to train models
train_models() {
    local infile=$1
    local outfile=$2
    # Train for fastText using SG
    python classify_word_vectors.py --inFilePath "./data/false_positive_identification_dataset.csv" --inModelPath "$infile" --outFileName "$outfile" --baseLine 1

}

# Iterate over input files and train models
for FILE in "${INPUT_FILES[@]}"; do
    OUT_NAME=$(basename "$FILE")  # Extract base name from file for output
    OUT_NAME+="_false_positive.txt"
    train_models "$FILE" "$OUT_NAME"
done

# Optional: List or move output files
echo "Models trained. Check the ./fastText_models and ./word2vec_models directories for output."