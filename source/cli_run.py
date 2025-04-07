""" CLI for model training and evaluation. 
CLI = Command Line Interface. This will allow any user to run the code from the command line in the terminal.
Basically, no need to run the whole notebook. """

import argparse #Needed to parse our command line arguments 

#Import our functions that we will need
from code.data_ingestion import load_csv_from_path
from code.data_cleaning import clean_data
from code.model_train_and_evaluate import train_knn, train_logistic_regression, train_SVM, train_naive_bayes

#We define our CLI arguments 
parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["knn", "logistic", "svm", "naive"], required=True) #The Model we want to train 
parser.add_argument("--train_path", type=str, required=True) #The path to the training dataset
parser.add_argument("--test_path", type=str, required=True) #The path to the testing dataset 

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
    "logistic": train_logistic_regression,
    "svm": train_SVM,
    "naive": train_naive_bayes
}
#Train the selected model 
if args.model in model_map:
    model, _ = model_map[args.model](X_train, y_train, X_test, y_test)
print(f"Model {args.model} trained and evaluated.")

#Example: python cli_run.py --model svm --train_path source/data/Train.csv --test_path source/data/Test.csv