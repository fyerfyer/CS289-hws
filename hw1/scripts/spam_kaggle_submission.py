from load import load_spam_data
from svm_model import train_svm_model
from save_csv import results_to_csv
from preprocess import fit_spam_scaler, apply_spam_scaler 

X_all, y_all, X_test = load_spam_data('../data/spam-data.npz')

scaler = fit_spam_scaler(X_all)

X_all_scaled = apply_spam_scaler(X_all, scaler)
X_test_scaled = apply_spam_scaler(X_test, scaler)

model = train_svm_model(X_all_scaled, y_all, C=10.0)

y_test = model.predict(X_test_scaled)

print("Starting to save results...")
results_to_csv(y_test, 'spam_submission.csv')
print("Results saved to spam_submission.csv")