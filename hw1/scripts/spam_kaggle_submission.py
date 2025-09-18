from load import load_spam_data
from svm_model import train_svm_model
from save_csv import results_to_csv

X_all, y_all, X_test = load_spam_data('../data/spam-data.npz')
model = train_svm_model(X_all, y_all, C=10.0)
y_test = model.predict(X_test)
print("Starting to save results...")
results_to_csv(y_test, 'spam_submission.csv')
print("Results saved to spam_submission.csv")