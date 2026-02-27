
from fastapi import FastAPI
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

app = FastAPI()

model_path = hf_hub_download(
    repo_id="your-username/visit_with_us_model",
    filename="best_model.pkl"
)

model = joblib.load(model_path)

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return {"prediction": int(pred)}
