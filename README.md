# Anomaly Detection in Credit Card Fraud

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![scikit--learn](https://img.shields.io/badge/scikit--learn-Latest-green)
![License](https://img.shields.io/badge/license-MIT-blue)

An advanced anomaly detection system for identifying fraudulent credit card transactions using machine learning techniques. This project implements both Isolation Forest and Autoencoder approaches to detect unusual patterns in credit card transactions.

## 🎯 Project Overview

Credit card fraud detection presents a significant challenge due to highly imbalanced datasets where fraudulent transactions are rare. Instead of using traditional classification approaches, this project employs anomaly detection techniques to identify transactions that deviate significantly from normal spending patterns.

### Key Features

- Implementation of two anomaly detection models:
  - Isolation Forest (traditional machine learning approach)
  - Autoencoder (deep learning approach)
- Comprehensive data preprocessing and analysis
- Model performance comparison and evaluation
- Detailed visualization of results
- Scalable and production-ready code structure

## 📊 Dataset

The project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle, which contains:

- 284,315 transactions
- 31 features (28 PCA-transformed features, Time, Amount, and Class)
- Highly imbalanced classes (0.172% fraudulent transactions)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Divinegrace05/Anomaly-Detection-in-Credit-Card-Fraud.git
cd credit-card-fraud-detection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 📈 Results

Our analysis shows that the Autoencoder outperforms the Isolation Forest in detecting fraudulent transactions:

| Model            | AUC-ROC Score | Precision | Recall | F1-Score |
|-----------------|---------------|-----------|---------|-----------|
| Isolation Forest | 0.759        | 0.64      | 0.76    | 0.68     |
| Autoencoder     | 0.899        | 0.51      | 0.95    | 0.97     |

### Key Findings

- Autoencoder showed superior performance in detecting complex fraud patterns
- Model successfully handles the extreme class imbalance (0.172% fraud cases)
- Feature scaling and preprocessing significantly impact model performance
- Neural network-based approach provides better generalization

## 🚀 Usage

1. Prepare your data:
```python
from preprocessing import preprocess_data

X_train, X_test, y_train, y_test = preprocess_data(data_path)
```

2. Train and evaluate models:
```python
from models import IsolationForestModel, AutoencoderModel

# Isolation Forest
iso_forest = IsolationForestModel()
iso_forest.train(X_train)
iso_forest_predictions = iso_forest.predict(X_test)

# Autoencoder
autoencoder = AutoencoderModel()
autoencoder.train(X_train)
autoencoder_predictions = autoencoder.predict(X_test)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- Credit Card Fraud Detection dataset from Kaggle
- Inspired by research in anomaly detection techniques
- Built with scikit-learn and TensorFlow
