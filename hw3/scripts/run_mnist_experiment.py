from load import load_mnist_data
from utils import shuffle_and_split, evaluate, calculate_per_digit_error
from preprocess import process_mnist_features
from model_fitting import fit_gaussian_models
from classifier_model import QDAClassifier, LDAClassifier
from visualization import visualize_covariance 
   
VALIDATION_SET_RATE = 0.2
SEED = 42 

print("Loading and preparing data...")
X_all, y_all, _ = load_mnist_data()

X_train_raw, X_val_raw, y_train, y_val = shuffle_and_split(
    X_all, y_all, validation_size=VALIDATION_SET_RATE, seed=SEED
)

X_train = process_mnist_features(X_train_raw)
X_val = process_mnist_features(X_val_raw)

print(f"Data prepared. Training set: {len(y_train)} samples. Validation set: {len(y_val)} samples.")

print("\nTraining models on the full training set...")

model_params = fit_gaussian_models(X_train, y_train)

N_train = X_train.shape[0]
lda_model = LDAClassifier(model_params, N_train, y_train)
qda_model = QDAClassifier(model_params)
print("Models initialized.")

print("Predicting on the validation set...")
lda_pred = lda_model.predict(X_val)
qda_pred = qda_model.predict(X_val)

print("\n--- Final Performance Report ---")

lda_accuracy = evaluate(lda_pred, y_val)
qda_accuracy = evaluate(qda_pred, y_val)
print(f"Overall LDA Accuracy: {lda_accuracy:.4f} (Error Rate: {1.0 - lda_accuracy:.4f})")
print(f"Overall QDA Accuracy: {qda_accuracy:.4f} (Error Rate: {1.0 - qda_accuracy:.4f})")

lda_digit_errors = calculate_per_digit_error(lda_pred, y_val)
for digit, error in sorted(lda_digit_errors.items()):
    print(f"  Digit {digit}: {error:.4f}")

qda_digit_errors = calculate_per_digit_error(qda_pred, y_val)
for digit, error in sorted(qda_digit_errors.items()):
    print(f"  Digit {digit}: {error:.4f}")

print("\nVisualizing covariance matrix for Digit '5' (as an example)...")
visualize_covariance(model_params[5]['covariance'], '5')

print("\nMNIST experiments finished!")
