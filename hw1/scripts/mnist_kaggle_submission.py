from preprocess import process_mnist_features
from load import load_mnist_data
from save_csv import results_to_csv
from svm_model import train_svm_model

X_train_raw, y_train, X_test_raw = load_mnist_data('../data/mnist-data.npz')

X_train = process_mnist_features(X_train_raw)
X_test = process_mnist_features(X_test_raw)

model = train_svm_model(X_train, y_train, C=10.0)
y_test = model.predict(X_test)
print("Starting to save results...")
results_to_csv(y_test, 'mnist_submission.csv')
print("Results saved to mnist_submission.csv")