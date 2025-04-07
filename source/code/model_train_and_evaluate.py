from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier

def train_naive_bayes(X_train, y_train, X_test, y_test): 
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('classifier', MultinomialNB())
    ])

    param_grid = {
        'tfidf__max_df': [0.8, 0.9, 1.0],  # Ex: a word with 0.8+ frequency will be discarded from the model.
        'tfidf__min_df': [1, 2],  #Ex: If the value chosen is 2, a word that appears in only one document will be removed.
        'tfidf__ngram_range': [(1, 1), (1, 2)],  #To look at combination of words.
        'classifier__alpha': [0.1, 0.5, 1.0],  #To avoid 0 value probability
    }

    #Gridsearch with 5 cross-validation.
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    #Evaluate the best model with the best hyperparameters
    print("Best Parameters:")
    print(grid_search.best_params_)

    best_model_naive_bayes = grid_search.best_estimator_
    y_pred = best_model_naive_bayes.predict(X_test)

    print("\nAccuracy of Best Model:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return best_model_naive_bayes, pipeline

def train_logistic_regression(X_train, y_train, X_test, y_test):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('classifier', LogisticRegression())
    ])

# Define the parameter grid for hyperparameter tuning
    param_grid = {
        'tfidf__max_features': [5000, 10000, 15000],  # Number of features for TF-IDF
        'classifier__C': [0.1, 1, 10],  # Regularization strength for Logistic Regression
        'classifier__penalty': ['l2'],  # Regularization type
        'classifier__solver': ['liblinear', 'saga'],  # Solvers to choose from
    }

    #Gridsearch
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)

    # Fit the model with hyperparameter tuning
    grid_search.fit(X_train, y_train)

    # Best parameters found by GridSearchCV
    print("Best parameters found: ", grid_search.best_params_)

    # Evaluate the best model from GridSearchCV
    y_pred = grid_search.best_estimator_.predict(X_test)

    # Print the classification report
    print(classification_report(y_test, y_pred))
    return grid_search.best_params_, pipeline

def train_SVM(X_train, y_train, X_test, y_test):
    #SVM with SGD classifier
    pipeline_sgd = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('classifier', SGDClassifier(loss='hinge', random_state=42))  # SVM with SGD
    ])

    # Define the parameter grid
    param_grid = {
        'classifier__alpha': [0.00001, 0.0001, 0.001, 0.01], #regularization coefficient
        'classifier__max_iter': [500, 1000, 2000], #max number of iterations the model will do to train
        'classifier__penalty': ['l2', 'elasticnet'], #which penantly to use for regularization
        'classifier__tol': [1e-4], #condition that the model will stop training
        'classifier__learning_rate': ['optimal', 'adaptive'], #strategy for adapting learning rate
    }

    #Gridsearch
    grid_search = GridSearchCV(pipeline_sgd, param_grid, cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Print the best parameters found by grid search
    print("Best parameters found: ", grid_search.best_params_)

    # Evaluate the model with the best parameters
    best_model = grid_search.best_estimator_
    y_pred_sgd = best_model.predict(X_test)

    # Print classification report
    print(classification_report(y_test, y_pred_sgd))
    return best_model, pipeline_sgd

def train_knn(X_train, y_train, X_test, y_test):
    # Define the pipeline with TF-IDF vectorizer and K-Nearest Neighbors classifier
    pipeline_knn = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('classifier', KNeighborsClassifier())
    ])

    # Define the parameter grid for hyperparameter tuning
    param_grid_knn = {
        'tfidf__max_features': [5000, 10000, 15000],  # Number of features for TF-IDF
        'classifier__n_neighbors': [3, 5, 7, 10],  # Number of neighbors to consider
        'classifier__weights': ['uniform', 'distance'],  # How to weight the neighbors
        'classifier__metric': ['euclidean', 'manhattan', 'cosine'],  # Distance metric
    }   

    # Perform GridSearchCV for hyperparameter tuning
    grid_search_knn = GridSearchCV(pipeline_knn, param_grid_knn, cv=3, n_jobs=-1, verbose=1)

    # Fit the model with hyperparameter tuning
    grid_search_knn.fit(X_train, y_train)

    # Best parameters found by GridSearchCV
    print("Best parameters found: ", grid_search_knn.best_params_)

    # Evaluate the best model from GridSearchCV
    y_pred_knn = grid_search_knn.best_estimator_.predict(X_test)

    # Print the classification report
    print(classification_report(y_test, y_pred_knn))
    return grid_search_knn.best_params_, pipeline_knn

