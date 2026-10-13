# CreditCard_Fraud_Detection

## Banking Credit Card Fraud Detection Using LSTM

The complete runnable project, trained model, evaluation reports, prediction CLI, and Streamlit app are in [Banking_Fraud_Detection_LSTM](Banking_Fraud_Detection_LSTM/README.md).

The completed chronological test run achieved 99.9561% accuracy, 93.1034% precision, 72.0000% recall, 81.2030% F1, and 96.4478% ROC-AUC on 56,962 sequences. The Kaggle CSV is stored locally under the project data directory and excluded from Git.

To run the web app after installing dependencies and providing the Kaggle dataset:

```bash
cd Banking_Fraud_Detection_LSTM
python -m src.train
streamlit run app.py
```