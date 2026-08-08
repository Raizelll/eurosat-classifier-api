from pydantic import BaseModel, Field


# Response for POST /predict
class PredictionResponse(BaseModel):
    predicted_class: str = Field(..., description="Most likely land-use class")
    # ge/le enforce that confidence stays a valid probability
    confidence: float = Field(..., ge=0.0, le=1.0, description="Probability of the predicted class")
    probabilities: dict[str, float] = Field(..., description="Probability for every class")


# Response for GET /health
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# Response for GET /metadata
class MetadataResponse(BaseModel):
    classes: list[str]
    input_size: list[int]
    model_format: str