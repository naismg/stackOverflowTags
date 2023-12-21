from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
from model_eda import model_, vect_

# Load your trained model
model = model_()
vectorizer = vect_()

app = FastAPI()

class Item(BaseModel):
    text: str

@app.post("/predict")
def predict(item: Item):
    # Vectorize the input text
    X = vectorizer.transform([item.text])
    # Make prediction
    y_pred = model.predict(X)
    return {"prediction": y_pred[0]}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
