from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
from model_eda import model, vect

# Load your trained model
modele = model()
vectorizer = vect()
app = FastAPI()

class Item(BaseModel):
    text: str

@app.post("/predict")
def predict(item: Item):
    # Vectorize the input text
    X = vectorizer.transform([item.text])
    # Make prediction
    y_pred = modele.predict(X)
    return {"prediction": y_pred[0]}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
