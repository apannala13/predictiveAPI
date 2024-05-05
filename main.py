from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
from pydantic import BaseModel
from typing import List
import pandas as pd
from model import *  
from io import StringIO
import logging 

app = FastAPI()
logging.basicConfig(level=logging.DEBUG)

class Model(BaseModel):
    Accuracy: float
    Confusion_Matrix: List[List[int]]

@app.get("/")
def root():
    return {"Test":"One"}

@app.post("/predict-stroke/", response_model=Model)
async def predict_stroke_api(file: UploadFile = File(...)):
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail='invalid file type.')
    
    try:
        content = await file.read()   
        dataset = pd.read_csv(StringIO(content.decode('utf-8')))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'error reading file: {str(e)}')

    try:
        predictor = StrokePredictor()
        predictor.preprocess_data(dataset)
        predictor.apply_smote()
        results = predictor.train_and_evaluate_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail = f'error during model exec: {str(e)}')
    
    return results 



