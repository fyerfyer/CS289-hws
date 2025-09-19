from save_csv import results_to_csv
from load import load_mnist_data
from preprocess import process_mnist_features
from model_fitting import fit_gaussian_models
from classifier_model import LDAClassifier

X_train_raw, y_train, X_test_raw = load_mnist_data()

X_train = process_mnist_features(X_train_raw)
X_test = process_mnist_features(X_test_raw)

model_params = fit_gaussian_models(X_train, y_train)
N_train = X_train.shape[0]
lda_model = LDAClassifier(model_params, N_train, y_train)
y_test = lda_model.predict(X_test)
print("Starting to save results...")
results_to_csv(y_test, 'mnist_submission.csv')
print("Results saved to mnist_submission.csv")