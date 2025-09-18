import numpy as np

from load import load_spam_data  
from utils import create_k_folds, evaluate
from preprocess import process_spam_features
from model_fitting import fit_gaussian_models
from classifier_model import LDAClassifier, QDAClassifier

K_FOLDS = 5
print(f"Running SPAM classification with {K_FOLDS}-Fold Cross-Validation.")

X_all, y_all, _ = load_spam_data()
k_folds = create_k_folds(X_all, y_all, k=K_FOLDS, seed=42)

lda_fold_accuracies = []
qda_fold_accuracies = []

for i in range(K_FOLDS):
    print(f"  Processing Fold {i+1}/{K_FOLDS}...")

    X_val_raw, y_val = k_folds[i]
    X_train_raw = np.concatenate([k_folds[j][0] for j in range(K_FOLDS) if j != i])
    y_train = np.concatenate([k_folds[j][1] for j in range(K_FOLDS) if j != i])

    X_train = process_spam_features(X_train_raw)
    X_val = process_spam_features(X_val_raw)

    model_params = fit_gaussian_models(X_train, y_train)

    N_train = X_train.shape[0]
    lda_model = LDAClassifier(model_params, N_train, y_train)
    qda_model = QDAClassifier(model_params)

    lda_pred = lda_model.predict(X_val)
    qda_pred = qda_model.predict(X_val)

    lda_accuracy = evaluate(lda_pred, y_val)
    qda_accuracy = evaluate(qda_pred, y_val)

    lda_fold_accuracies.append(lda_accuracy)
    qda_fold_accuracies.append(qda_accuracy)

print("\n--- Cross-Validation Results ---")

lda_mean_acc = np.mean(lda_fold_accuracies)
lda_std_acc = np.std(lda_fold_accuracies)
qda_mean_acc = np.mean(qda_fold_accuracies)
qda_std_acc = np.std(qda_fold_accuracies)

print(f"LDA Average Accuracy: {lda_mean_acc:.4f} (+/- {lda_std_acc:.4f})")
print(f"QDA Average Accuracy: {qda_mean_acc:.4f} (+/- {qda_std_acc:.4f})")

print("\nSPAM experiments finished!")
