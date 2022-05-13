# Chinese-Financial-News-Sentiment-Analysis

Natural Language Processing (NLP) lets machines understand and interact with humans. 
In the past research, machine learning techniques such as deep neural networks outperformed traditional models in some NLP tasks. 
We wonder how useful it is to apply it to a sentiment analysis task in the financial market.
We apply from rule-based approach to BERT to predict negative financial news.
Unlike most practical works done in English, we particularly focus on Chinese natural language texts.
We show that deep learning methods can do similarly well but no better on Chinese sentiment analysis,
but it is still worth trying in the industry.

1. first, it will preprocess data, which are some Chinese financial news texts;
2. later, several methods will be applied to predict the sentiments of these texts, including a rule-based approach using Lexicon, 
traditional ML approach using Lexicon or Tfidf,
deep learning approach using Recurrent Neural Network (RNN) or LSTM architecture,
and more advanced NLP pre-trained models BERT;
3. lastly, the performances of these methods will be compared and discussed.

Files include:  
* Data: Train_Data.csv, Test_Data.csv, /dictionary/ (to build features)  
* Code: models.py (for neural network models), dataset.py (for converting Vocab to Int, padding, batching data), train_cls.py (for training neural networks)  
* Notebook: Project.ipynb (for preprocessing data, sentiment prediction using different models, and show performances)  
<!-- * Report: Final_Report.pdf (detailed explanation of the project)   -->

Pytorch implementations:
* 1-layer RNN model
* 2-layer LSTM model
* BERT model: Chinese RoBERTa-Base Models for Sequence Classification (Huggingface), finetuned for tasks
