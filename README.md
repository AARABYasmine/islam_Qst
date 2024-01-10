# Binary Classification with RNN

This repository contains code for a binary classification task using a Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM) and Transformer to determine the correctness of answers based on given questions and corresponding answers in the dataset.

## Overview

The code in this repository encompasses a comprehensive pipeline:

- **Data Loading and Exploration**: Loads the dataset (`data.csv`) containing columns: `Question_id`, `Question`, `Answer`, and `score`. Provides an overview of the dataset statistics, distributions, and missing values.

- **Text Preprocessing and Feature Engineering**: Employs a robust text preprocessing pipeline including tokenization, removal of Arabic diacritics, punctuations, stopwords, and lemmatization. Conducts word frequency analysis and generates TF-IDF, Word2Vec, and FastText embeddings for feature representation.

- **Model Building and Training**: Implements an architecture for binary classification. Evaluates and fine-tunes the model using training and validation sets.

- **Evaluation and Prediction**: Assesses model performance on test data and showcases how to make predictions.

## Requirements

Ensure you have the following installed:

- Python 3.x
- Libraries specified in `requirements.txt`

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/AARABYasmine/islam_Qst.git
   cd islam_Qst
   ```
   
2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```
    
## Usage

1. Ensure the dataset (`data.csv`) is placed in the appropriate directory.

2. Run the main script:

    ```bash
    python -m flask run
    ```
    
3. Modify hyperparameters, experiment with different models, or adjust functionalities as required.

# Project Structure

- `data.csv`: Dataset containing Question_id, Question, Answer, and score columns.
- `main.py`: Main script for data loading, preprocessing, modeling, training, and evaluation.
- `models.py`: Holds classes for RNN, LSTM and Transformer models.
- `utils.py`: Utility functions for text preprocessing, feature engineering, and evaluation.

# Results

- The RNN model achieved an accuracy of 90.62% on the test set.
- Additionally, an LSTM model obtained an accuracy of 87.50% on the same test set.
- Transformer model obtained an accuracy of % on the same test set.

# Notes

- `nlp_pipeline` function streamlines comprehensive text preprocessing steps, ensuring high-quality data for modeling.
- Features TF-IDF, Word2Vec, and FastText embeddings, offering diverse representations for model learning.
- The repository provides flexibility to adjust model architectures, hyperparameters, and experiment with different embedding techniques.

# Contribution Guidelines

Feel free to contribute by:

- Improving existing functionalities
- Adding new features or models
- Enhancing documentation or code readability
