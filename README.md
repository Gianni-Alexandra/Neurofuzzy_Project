#### News Text Classification with Neural Networks

## Project Overview
The goal of this project is to build a classifier capable of categorizing short news articles into hierarchical categories using a neural network model.
The classification involves:
- 17 top-level news categories
- 109 second-level subcategories

The dataset used comes from a collection of **10,917 manually-labeled news articles** based on the NewsCodes Media Topic taxonomy.

## 📁 Dataset
The dataset used for training and evaluation can be downloaded from:
```bash
wget https://courses.e-ce.uth.gr/CE418/nfc_fall23/news-classification.csv
```

Each entry in the dataset includes:
- The news article text
- The top-level and second-level category labels
  
A separate evaluation set was also provided closer to the submission deadline.

## 📊 Evaluation
Model performance was evaluated based on:
- Accuracy on both top-level and second-level category classification
- Training time as a function of dataset size
- Inference latency per article

📚 Requirements
Make sure you have the following installed:
- Python 3.8+
- PyTorch or TensorFlow (depending on your implementation)
- Pandas, NumPy, Scikit-learn
- tqdm, matplotlib (optional for visualizations)

Install dependencies via:
```bash
pip install -r requirements.txt
``` 



