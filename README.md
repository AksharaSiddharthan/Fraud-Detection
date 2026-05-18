# Fraud-Detection

# Steps to Run the Credit Card Fraud Detection Project
1. Download the Dataset

Download the dataset from:

Kaggle Credit Card Fraud Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download

After downloading:

Extract the ZIP file
Place creditcard.csv inside your project folder

# 2. Create Project Folder

```
credit-card-fraud-detection/
│
├── creditcard.csv
├── model.py
├── README.md

```

# 3. Install Python Libraries

Open terminal/command prompt inside the project folder.

Run:

pip install pandas numpy scikit-learn joblib

# 4. Add the model.py File

Create a file named:

model.py

Paste the fraud detection code into it.

# 5. Run the Model

Execute:

python model.py
# 6. What Happens After Running
```
The script will:

✅ Load the dataset
✅ Split training/testing data
✅ Scale features
✅ Train Random Forest model
✅ Predict fraud transactions
✅ Print evaluation metrics
✅ Save trained model files
```
# 7. Expected Output

You will see:

Accuracy Score
Confusion Matrix
Classification Report
ROC-AUC Score

Example:

Accuracy Score:
0.9995

ROC-AUC Score:
0.97

# 8. Generated Files

After successful execution:

fraud_detection_model.pkl
scaler.pkl

These files store:

trained ML model
feature scaler

You can later use them for:

deployment
APIs
web apps
real-time fraud prediction
