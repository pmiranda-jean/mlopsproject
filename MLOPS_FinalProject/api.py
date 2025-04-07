from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model and pipeline
model, pipeline = joblib.load("model.pkl")

# Define the input structure
class TextInput(BaseModel):
    text: str

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to the Toxic Comment Classifier API!"}

@app.post("/predict")
def predict(input_data: TextInput):
    # Clean and predict
    transformed = pipeline.transform([input_data.text])
    prediction = model.predict(transformed)
    return {"prediction": int(prediction[0])}
