from joblib import load #Needed to use the pre-saved model 
from sklearn.metrics import accuracy_score, classification_report #Needed to evaluate the model 

def model_evaluate_from_saved_file(model_path, X_test, y_test, model_name="Model"):
    model = load(model_path) #Take the model 
    y_pred = model.predict(X_test) #Predict with that model 
    print(f"\n Evaluation for {model_name}")
    print("Accuracy:", accuracy_score(y_test, y_pred)) #Print the accuracy of the model
    print("Classification Report:\n", classification_report(y_test, y_pred)) #Print the classification report
