from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import json

app = FastAPI()

class Model(BaseModel):
    accuracy: float
    confusion_matrix: List[List[int]]

@app.get("/results/", response_model=Model)
async def get_results():
    try:
        # Directly load JSON from file
        with open('model_results.json', 'r') as f:
            response = json.load(f)
            
        model_results = Model(
            accuracy=response['Accuracy'],
            confusion_matrix=response['Confusion_Matrix']
        )
        return model_results
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Error decoding JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
