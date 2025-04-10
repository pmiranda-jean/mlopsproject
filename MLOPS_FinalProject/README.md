# README

# Description 
We have developed machine learning models that would predict whether a review given by a user is positive (1) or negative (0). We created 4 simple models (Naïve Bayes, SVM, KNN and Logistic Regression), trained and evaluated them. The project includes an API and a Docker file, and can be ran directly through a command-line interface. 

# Folder Structure 
 MLOPS_FinalProject/
 - data/ # Train and test datasets
 - models/ # Saved trained models (no need to retrain)
 - src/ # Source functions for data and models
   - data_analysis.py # Simple data analysis functions
   - data_cleaning.py # Cleans non-alphabetical text artifacts
   - data_ingestion.py # Function to load the data in a pandas dataframe
   - model_evaluate.py # Function to measure accuracy of our models
   - model_train.py # Functions to train the 4 models
 - api.py # Code for the FastAPI app
 - run_cli.py # Code for the CLI interface
 - pyproject.toml # Poetry dependency file
 - Dockerfile # Code for Docker container

# Dependencies 
All dependencies are managed via Poetry and included in the pyproject.toml file. 
If Poetry is not installed yet, paste the following in your terminal : curl -sSL https://install.python-poetry.org | python3 - 

# Installation 
1. Clone the repository using the following link : https://github.com/pmiranda-jean/mlopsproject.git
   Navigate to your folder cd mlopsproject 

3. OPTIONAL, only if the models are not yet saved within the 'models' folder. 
    In your terminal, modify the values and paste it in your terminal: 
    python cli_run.py --mode train --model svm --train_path source/data/Train.csv --test_path source/data/Test.csv 

    - --mode : whether you want to train or evaluate. You have to train before evaluate. 
    - --model : which model you want to use. Options are naive, knn, svm and logistic. 
    - --train_path: the path of your train dataset
    - --test_path: the path of your test dataset

   This will train or evaluate the selected model and save the model in the 'models' folder. 

4. Set up the environment with Docker 
    1. Build the Docker image by posting the following text in your terminal: 
    text: docker build -t my-fastapi-app .
    2. Run the Docker contained by posting the following text in your terminal: 
    text: docker run -p 8000:8000 my-fastapi-app

5. Use the API in a web browser
    1. Run the application using by posting the following link in an url: 
    link: http://127.0.0.1:8000/docs
    2. Your browser should open a webpage. Click on 'POST'. Then click on 'Try it out'. 
    3. Modify the text between "" for text and model name. 
    An example: 
    { "text": "This movie was really bad",
      "model_name": "svm"
     }
    4. Click on "Execute". You should see whether your comment was negative (prediction = 0) or positive (prediction = 1)
  
# Future Improvements 
1. Deploy Container to allow all users to use the API on their browser.
2. Add Unit Tests.
3. Improve model performance through hyperparameter tuning and crossvalidation. 
