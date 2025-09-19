from save_csv import results_to_csv
from load import load_spam_data
from preprocess import process_spam_features
from model_fitting import fit_gaussian_models
from classifier_model import LDAClassifier

X_train_raw, y_train, X_test_raw = load_spam_data()

X_train = process_spam_features(X_train_raw)
X_test = process_spam_features(X_test_raw)

model_params = fit_gaussian_models(X_train, y_train)
N_train = X_train.shape[0]
lda_model = LDAClassifier(model_params, N_train, y_train)
y_test = lda_model.predict(X_test)
print("Starting to save results...")
results_to_csv(y_test, 'spam_submission.csv')
print("Results saved to spam_submission.csv")