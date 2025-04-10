"""Objective: 
With this, we will have the user post a review of a movie and determine with one of the loaded models if it is a 
positive or negative review."""

from fastapi import FastAPI #To create my web application 
from pydantic import BaseModel #For my incoming requests
import joblib #For my saved models 
import os #To check if the model file exists before trying to import it
from src.data_cleaning import clean_data #Need this function for my prediction 
import pandas as pd

app = FastAPI() #Initiate new API instance 

#This defines the structure the input should have. 
class TextInput(BaseModel):
    text: str 
    model_name: str

#Create my POST endpoint 
@app.post("/predict")

#Function that will predict if the text is positive or negative based on the model selected 
def predict(input: TextInput):
    model_path = f"models/model_{input.model_name}.pkl" #Path to the previously saved model 

    if not os.path.exists(model_path): #To ensure the models have been saved before using the API
        return {"error": f"Model '{input.model_name}' not found at path '{model_path}'."}

    try:
        model = joblib.load(model_path)
    except Exception as e:
        return {"error": f"Error loading the model: {str(e)}"}

    input_data = pd.DataFrame({ 'text': [input.text] })  #Create DataFrame with column 'text'

    cleaned_data = clean_data(input_data, text_column='text') #Clean the input text using the clean_data function
    cleaned_text = cleaned_data['cleaned_text'] #Get cleaned text
    prediction = model.predict(cleaned_text)[0] #Predict using the cleaned text

    return {"prediction": int(prediction)} #Return the prediction as an integer (0 = Negative, 1= Positive)