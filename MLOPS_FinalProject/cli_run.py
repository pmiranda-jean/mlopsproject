""" CLI for model training and evaluation. 
CLI = Command Line Interface. This will allow any user to run the code from the command line in the terminal.
Basically, no need to run the whole notebook. """

import argparse #Needed to parse our command line arguments 
import joblib #Needed to save our model 
import os #Needed to make sure our "Model" folder exists 

#Import our functions that we will need
from src.data_ingestion import load_csv_from_path
from src.data_cleaning import clean_data
from src.model_train import train_knn, train_lr, train_svm, train_nb
from src.model_evaluate import model_evaluate_from_saved_file

#We define our CLI arguments 
parser = argparse.ArgumentParser(description="CLI for model training and evaluation")
parser.add_argument("--mode", choices=["train", "evaluate"], required=True, help="Choose 'train' or 'evaluate'") #If we want to train or evaluate
parser.add_argument("--model", choices=["knn", "logistic", "svm", "naive"], required=True) #The Model we want to train 
parser.add_argument("--train_path", type=str, help="The path to the training dataset") #The path to the training dataset
parser.add_argument("--test_path", type=str, required=True, help="The path to the testing dataset") #The path to the testing dataset 

args = parser.parse_args() #read the command line arguments from the user 

#Load the Data 
train_data = load_csv_from_path(args.train_path) #args.train_path is the path to the training dataset added by user 
test_data = load_csv_from_path(args.test_path) #args.test_path is the path to the test dataset added by user 

#Clean the Data
train_cleaned = clean_data(train_data, "text")
test_cleaned = clean_data(test_data, "text")

#Split the Data 
X_train = train_cleaned["text"]
y_train = train_cleaned["label"]
X_test = test_cleaned["text"]
y_test = test_cleaned["label"]

#Dicitionary to Map the models to their training and evaluating functions 
model_map = {
    "knn": train_knn,
    "logistic": train_lr,
    "svm": train_svm,
    "naive": train_nb
}

# Train mode
if args.mode == "train":
    if not args.train_path:
        raise ValueError("Training path must be provided in train mode.")

    os.makedirs('models', exist_ok=True) #Make sure the folder exists 

    if args.model in model_map:
        model = model_map[args.model](X_train, y_train)
        model_path = f"models/model_{args.model}.pkl"
        joblib.dump(model, model_path)
        print(f"{args.model.capitalize()} model trained and saved at {model_path}")
    else:
        print("Invalid model name. Choose from:", list(model_map.keys()))

# Evaluate mode
elif args.mode == "evaluate":
    model_path = f"models/model_{args.model}.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}. Please train the model first.")
    
    # Call the evaluate function you defined
    model_evaluate_from_saved_file(model_path, X_test, y_test, model_name=args.model)

#Example to Train: python cli_run.py --mode train --model svm --train_path source/data/Train.csv --test_path source/data/Test.csv
#Examplte to Evaluate: python cli_run.py --mode evaluate --model knn --train_path source/data/Train.csv --test_path source/data/Test.csv
