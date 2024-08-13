#!/bin/bash

# Step 1: Generate zero predictions
echo "Generating zero predictions..."
python generate_zero_predictions.py -c outlier-challenge-config.json

# Check if the zero predictions file was created successfully
if [ $? -eq 0 ]; then
    echo "Zero predictions generated successfully."
else
    echo "Failed to generate zero predictions. Exiting."
    exit 1
fi

# Step 2: Submit the zero predictions
echo "Submitting zero predictions..."
python submit_outlier_detections.py -c outlier-challenge-config.json

# Check if the submission was successful
if [ $? -eq 0 ]; then
    echo "Submission completed successfully."
else
    echo "Failed to submit predictions. Exiting."
    exit 1
fi