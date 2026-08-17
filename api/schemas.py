from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    filename: str