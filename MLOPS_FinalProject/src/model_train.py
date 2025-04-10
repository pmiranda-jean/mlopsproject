from sklearn.pipeline import Pipeline #Needed to train our models 
from sklearn.feature_extraction.text import TfidfVectorizer #Needed to vectorize the text 
from sklearn.naive_bayes import MultinomialNB #Needed to train the Naive Bayes model 
from sklearn.linear_model import LogisticRegression #Needed to train the Logistic Regression model
from sklearn.linear_model import SGDClassifier #Needed to train the SVM model 
from sklearn.neighbors import KNeighborsClassifier #Needed to train the KNN model 
import joblib #Needed to save our models 
import os #Needed to check if the model file exists before trying to import it

def train_nb(X_train, y_train):
    #Create the Pipeline 
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')), #Vectorize the text 
        ('classifier', MultinomialNB()) #Use Naive Bayes model
    ])
    pipeline.fit(X_train, y_train) #Train the model with our training data 
    
    model_name = "naive_bayes" 
    model_path = os.path.join("models", f"{model_name}_model.pkl") 
    joblib.dump(pipeline, model_path) #Save the model with the path 
    print(f"Naive Bayes model saved at {model_path}")
    return pipeline, model_path

def train_lr(X_train, y_train):
    #Create the Pipeline 
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')), #Vectorize the text 
        ('classifier', LogisticRegression(solver='liblinear')) #Use Logistic Regression model
    ])
    pipeline.fit(X_train, y_train)
    
    model_name = "logistic_regression"
    model_path = os.path.join(f"model_{model_name}.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Naive Bayes model saved at {model_path}")
    return pipeline, model_path

def train_svm(X_train, y_train):
    #Create the Pipeline 
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')), #Vectorize the text 
        ('classifier', SGDClassifier(loss='hinge', random_state=42)) #Use SVM model
    ])
    pipeline.fit(X_train, y_train)
    
    model_name = "svm"
    model_path = os.path.join(f"model_{model_name}.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Naive Bayes model saved at {model_path}")
    return pipeline, model_path

def train_knn(X_train, y_train):
    #Create the Pipeline 
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')), #Vectorize the text 
        ('classifier', KNeighborsClassifier(n_neighbors=5)) #Use KNN model
    ])
    pipeline.fit(X_train, y_train)
    
    model_name = "knn"
    model_path = os.path.join(f"model_{model_name}.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Naive Bayes model saved at {model_path}")
    return pipeline, model_path
