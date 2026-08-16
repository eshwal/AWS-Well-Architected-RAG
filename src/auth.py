from fastapi import Header, HTTPException
from src.config import settings

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != settings.DEMO_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")