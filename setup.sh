#!/bin/bash

# Create NLTK data directory
mkdir -p ~/.nltk_data

# Download NLTK resources
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

echo "NLTK resources downloaded successfully"