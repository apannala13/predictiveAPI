
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import SMOTE
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

class StrokePredictor:
    def __init__(self):
        self.dataset = None
        self.transformed_dataset = None
        self.X_binary_rounded = None
        self.y_resampled = None

    def preprocess_data(self, dataset):
        self.dataset = dataset
        self.dataset['gender'].replace('Other', np.nan, inplace=True)

        transformation = ColumnTransformer([
            ('numerical', make_pipeline(SimpleImputer(strategy='mean'), StandardScaler()), ['bmi']),
            ('gender_transform', make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(drop='first')), ['gender']),
            ('binary_encoding', OneHotEncoder(drop='first'), ['ever_married', 'Residence_type']),
            ('categorical_encoding', OneHotEncoder(), ['work_type', 'smoking_status']),
            ('numeric_standardization', StandardScaler(), ['age', 'avg_glucose_level'])
        ], remainder='passthrough')

        pipeline = Pipeline([
            ('Transformation', transformation),
            ("Scaler", StandardScaler())
        ])

        transformed_data = transformation.fit_transform(self.dataset)

        transformed_column_names = (['BMI', 'Gender'] +
                                    ['Married', 'Residence_type', 'Private Sector', 'Self-employed', 'Govt_job',
                                     'Children', 'Never_worked', 'Formerly_smoked', 'Never_smoked', 'Smokes',
                                     'Unknown'] +
                                    self.dataset.columns.drop(['bmi', 'gender', 'ever_married', 'Residence_type',
                                                               'work_type', 'smoking_status']).tolist())

        self.transformed_dataset = pd.DataFrame(transformed_data, columns=transformed_column_names)

    def apply_smote(self):
        X = self.transformed_dataset.iloc[:, :-1]
        y = self.transformed_dataset.iloc[:, -1]

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        threshold = 0.5
        self.X_binary_rounded = X_resampled.copy()

        binary_columns = ['Gender', 'Residence_type', 'Private Sector', 'Govt_job', 'Children', 'Never_worked',
                          'Formerly_smoked', 'Never_smoked', 'Smokes', 'Unknown', 'heart_disease']

        for col in binary_columns:
            self.X_binary_rounded[col] = self.X_binary_rounded[col].apply(lambda x: 1 if x >= threshold else 0)

        self.y_resampled = y_resampled

    def train_and_evaluate_model(self):
        """
        pass 
        """
        pca = PCA(n_components=10)
        selector = SelectKBest(f_classif, k=5)

        pipeline = Pipeline([
            ('pca', pca),
            ('feature_selector', selector)
        ])

        pipelined_X = pipeline.fit_transform(self.X_binary_rounded, self.y_resampled)

        X_train, X_test, y_train, y_test = train_test_split(pipelined_X, self.y_resampled, test_size=0.2, random_state=42)

        rf_classifier = RandomForestClassifier()
        rf_classifier.fit(X_train, y_train)

        y_pred = rf_classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        conf_matrix_list = conf_matrix.tolist()

        results = {
            "Accuracy": accuracy,
            "Confusion_Matrix": conf_matrix_list
        }
        return results

def predict_stroke(dataset):
    predictor = StrokePredictor()
    predictor.preprocess_data(dataset)
    predictor.apply_smote()
    results = predictor.train_and_evaluate_model()
    return json.dumps(results)