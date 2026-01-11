from fastapi import FastAPI
from pydantic import BaseModel
import joblib

model = joblib.load("spam_model.pkl")

app = FastAPI()

class EmailRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_spam(email: EmailRequest):
    prediction = model.predict([email.text])[0]
    return {
        "text_received": email.text,
        "prediction": prediction
    }

@app.get("/")
def read_root():
    return {"status": "Model is running"}