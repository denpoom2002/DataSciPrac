# Bring in 
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder

import pandas as pd
import numpy as np
from joblib import load
from autogluon.features.generators import AutoMLPipelineFeatureGenerator

app = FastAPI()

class InfoItem(BaseModel):
    Month: str
    WeekOfMonth: int
    DayOfWeek: str
    Make: str
    AccidentArea: str
    DayOfWeekClaimed: str
    MonthClaimed: str
    WeekOfMonthClaimed: int
    Sex: str
    MaritalStatus: str
    Age: int
    Fault: str
    PolicyType: str
    VehicleCategory: str
    VehiclePrice: str
    PolicyNumber: int
    RepNumber: int
    Deductible: int
    DriverRating: int
    Days_Policy_Accident: str
    Days_Policy_Claim: str
    PastNumberOfClaims: str
    AgeOfVehicle: str
    AgeOfPolicyHolder: str
    PoliceReportFiled: str
    WitnessPresent: str
    AgentType: str
    NumberOfSuppliments: str  
    AddressChange_Claim: str
    NumberOfCars: str
    Year: int
    BasePolicy: str

df = pd.read_csv("fraud_oracle (1).csv")
X = df.drop(columns=['FraudFound_P'])
auto_ml_pipeline_feature_generator = AutoMLPipelineFeatureGenerator()
auto_ml_pipeline_feature_generator.fit_transform(X=X)
model = load("fraud.joblib")

@app.post('/')
async def scoring_endpoint(item:InfoItem):
    data = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    data_transform = auto_ml_pipeline_feature_generator.transform(X=data)
    prediction = model.predict(data_transform)
    pred_label = np.rint(prediction).astype(int)
    response = {"prediction": int(pred_label[0])}
    return response
