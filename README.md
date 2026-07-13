# Default Prediction Model

A machine learning project for predicting loan defaults using classification algorithms.

## Overview

This project implements a credit default prediction system that assesses the likelihood of a borrower defaulting on a loan. The model uses historical data to identify patterns and risk factors associated with loan defaults, enabling better lending decisions and risk management.

## Features

- **Data Processing**: Handles missing values, encodes categorical variables, and scales numerical features
- **Feature Engineering**: Creates meaningful features from raw data to improve model performance
- **Model Training**: Implements multiple classification algorithms for comparison
- **Model Evaluation**: Provides comprehensive evaluation metrics including accuracy, precision, recall, F1-score, and AUC-ROC
- **Prediction Pipeline**: End-to-end pipeline from data ingestion to predictions

## Project Structure

```
default_prediction_model/
├── README.md           # Project documentation
├── LICENSE            # GNU GPL v3 License
├── data/              # Raw and processed data files
├── notebooks/         # Jupyter notebooks for exploration
├── src/               # Source code modules
│   ├── data/          # Data processing scripts
│   ├── features/      # Feature engineering
│   ├── models/        # Model training and evaluation
│   └── visualization/ # Data visualization
├── models/            # Saved model files
└── tests/             # Unit tests
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip or conda package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/username/default_prediction_model.git
cd default_prediction_model
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

Place your loan data in the `data/raw/` directory. The model expects a CSV file with columns including:
- Loan amount
- Interest rate
- Employment length
- Annual income
- Debt-to-income ratio
- Credit score
- Loan status (target variable)

### Training the Model

```python
from src.models.train_model import train_model
from src.data.load_data import load_data

# Load and preprocess data
X_train, X_test, y_train, y_test = load_data()

# Train the model
model = train_model(X_train, y_train)
```

### Making Predictions

```python
from src.models.predict import predict_default

# Predict default probability for a new applicant
probability = predict_default(model, applicant_features)
```

## Algorithms

The project implements and compares the following algorithms:

- **Logistic Regression**: Baseline model for binary classification
- **Random Forest**: Ensemble method for improved accuracy
- **Gradient Boosting**: XGBoost/LightGBM for high performance
- **Neural Network**: Deep learning approach for complex patterns

## Evaluation Metrics

- **Accuracy**: Overall correct predictions
- **Precision**: Ability to avoid false positives
- **Recall**: Ability to detect all defaults
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

For questions or feedback, please open an issue in the repository.
