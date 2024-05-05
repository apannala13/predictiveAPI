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

result_router = APIRouter()
prediction_router = APIRouter()

@result_router.get("/results/", response_model=Model)
async def get_results():
    try:
        # Directly load JSON from file
        with open('model_results.json', 'r') as f:
            response = json.load(f)
            
        model_results = Model(
            Accuracy=response['Accuracy'],
            Confusion_Matrix=response['Confusion_Matrix']
        )
        return model_results
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Error decoding JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@prediction_router.post("/predict-stroke/", response_model=Model)
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


app.include_router(result_router)
app.include_router(prediction_router)

@app.get("/")
def root():
    return {"message": "API - root. navigate to routers.."}

