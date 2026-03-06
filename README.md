# Arabic Sentiment Analysis (TF-IDF + PyTorch)

This project implements a binary sentiment classification system for Arabic reviews using traditional NLP feature engineering combined with a neural network classifier.
The model was trained on a balanced dataset of ~150,000 Arabic reviews and optimized using Google Colab due to local memory constraints.

## Methodology

1. **Text Preprocessing**
   - Arabic character normalization
   - Repetition reduction
   - Diacritics removal
   - Non-Arabic character filtering

2. **Feature Extraction**
   - TF-IDF vectorization
   - Unigrams and bigrams
   - Maximum feature size control

3. **Model Architecture**
   - Fully Connected Neural Network (MLP)
   - ReLU activation
   - CrossEntropy loss
   - F1-score evaluation

## Results

The model achieved strong performance on balanced binary classification using F1-score as the primary metric.

## Project Structure

- `sampling.py` – dataset balancing and preprocessing
- `model.py` – TF-IDF vectorization and PyTorch training loop

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
