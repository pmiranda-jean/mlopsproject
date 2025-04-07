
from sklearn.metrics import accuracy_score, classification_report

def test_naive_bayes(data, best_model):
    X_valid = data['cleaned_text']
    y_valid = data['label']

    y_pred_valid = best_model.predict(X_valid)

    print("\nAccuracy on Validation Set:", accuracy_score(y_valid, y_pred_valid))
    print("\nClassification Report on Validation Set:")
    print(classification_report(y_valid, y_pred_valid))