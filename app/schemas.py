from pydantic import BaseModel, field_validator
from typing import Optional

class CustomerFeatures(BaseModel):
    gender: Optional[str] = None
    SeniorCitizen: Optional[int] = None
    Partner: Optional[str] = None
    Dependents: Optional[str] = None
    tenure: Optional[int] = None
    PhoneService: Optional[str] = None
    MultipleLines: Optional[str] = None
    InternetService: Optional[str] = None
    OnlineSecurity: Optional[str] = None
    OnlineBackup: Optional[str] = None
    DeviceProtection: Optional[str] = None
    TechSupport: Optional[str] = None
    StreamingTV: Optional[str] = None
    StreamingMovies: Optional[str] = None
    Contract: Optional[str] = None
    PaperlessBilling: Optional[str] = None
    PaymentMethod: Optional[str] = None
    MonthlyCharges: Optional[float] = None
    TotalCharges: Optional[float] = None

    @field_validator("SeniorCitizen")
    @classmethod
    def senior_zero_one(cls, v):
        if v is None:
            return v
        if v not in (0, 1):
            raise ValueError("SeniorCitizen must be 0 or 1")
        return v

class PredictionRequest(BaseModel):
    records: list[CustomerFeatures]

class PredictionResponse(BaseModel):
    probabilities: list[float]
    predictions: list[int]  # 1 = churn, 0 = no churn
