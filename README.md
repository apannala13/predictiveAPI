# predictiveAPI
Simple Fast API hosting results from an ML model. 

Run in terminal:
(env) user % uvicorn main:app --reload

Swagger UI:
http://127.0.0.1:8000/docs#/

Post CMD:
curl -X POST "http://127.0.0.1:8000/predict-stroke/" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@stroke_dataset.csv;type=text/csv"
