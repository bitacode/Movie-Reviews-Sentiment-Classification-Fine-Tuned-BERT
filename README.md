# Sentiment Classification with Fine-Tuned BERT

## Introduction

BERT stands for Bidirectional Encoder Representations from `transformers` and is a language representation model by Google. It uses two steps, pre-training and fine-tuning, to create state-of-the-art models for a wide range of tasks. BERT architecture is a stack of `transformers`’s Encoder, the Transformer in NLP is a novel architecture that aims to solve sequence-to-sequence tasks while handling long-range dependencies with ease.

`TFBertForSequenceClassification` is a TensorFlow-based model from Hugging Face's `transformers` library, specifically designed for sequence classification tasks (e.g., sentiment analysis, spam detection) using BERT. To use this model for sentiment analysis, fine-tuning is necessary. In the context of NLP, fine-tuning involves taking a BERT model that has been pre-trained on a large corpus of text and further training it on a labeled dataset for sentiment analysis. The model’s parameters are slightly adjusted to specialize in classifying sentiment while retaining the general language understanding learned during pre-training.

## Dataset

This experiment employs a dataset that comprises movie reviews. The dataset used for model training is publicly available on Kaggle and accessible for free on the internet. The link to this data is [Kaggle Rotten Tomatoes EDA](https://www.kaggle.com/code/stefanoleone992/rotten-tomatoes-eda). The movie reviews were annotated using [RoBERTa](https://github.com/bitacode/Labeling-Dataset-For-Sentiment-Analysis.git).

## Results

The model achieved a training loss of 0.0796 and an accuracy of 96.98%, indicating that it performs well on the training data. However, the validation loss is significantly higher at 0.4776, with a validation accuracy of 87.70%. This suggests that while the model generalizes fairly well to unseen data, there may be some overfitting, as indicated by the larger gap between the training and validation loss.

The model achieved an overall accuracy of 84%. The F1-scores for the positive and negative classes were both high at 0.88, indicating a strong performance in these categories, with the positive class having a recall of 0.95, meaning it was particularly effective in identifying true positive examples. The neutral class had a slightly lower F1-score of 0.75, with a recall of 0.70, suggesting that the model struggled more with neutral sentiment classification compared to positive and negative sentiments. The macro and weighted averages for precision, recall, and F1-score were consistent at 0.84, reflecting balanced performance across all classes.

## Prospects

Adding regularization and dropout to a `TFBertForSequenceClassification` model can be beneficial for addressing issues like overfitting. Additionally, adjusting the learning rate scheduler, and the early stopping parameters can help overcome the overfitting observed in the validation results.
