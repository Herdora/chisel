import os
from decimal import Decimal
from typing import Optional

from database import get_db, init_db
from fastapi import Depends, FastAPI, HTTPException
from models import UsageHistory, User
from pydantic import BaseModel
from sqlalchemy.orm import Session

app = FastAPI(title="Chisel Backend API")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Your actual DigitalOcean token
DO_TOKEN = os.getenv("DO_API_TOKEN", "your_digitalocean_token_here")

# GPU pricing (per hour)
GPU_RATES = {
    "amd-mi300x": Decimal("1.99"),
    "nvidia-h100": Decimal("4.89"),
    "nvidia-l40s": Decimal("2.50"),
}


class TokenValidateResponse(BaseModel):
    valid: bool
    credits_remaining: Optional[float] = None
    warning: Optional[str] = None


class DOTokenResponse(BaseModel):
    do_token: Optional[str] = None
    error: Optional[str] = None


class UsageTrackRequest(BaseModel):
    chisel_token: str
    droplet_id: str
    gpu_type: str
    duration_minutes: int


@app.on_event("startup")
async def startup_event():
    init_db()


@app.get("/")
async def root():
    return {"message": "Chisel Backend API", "status": "running"}


@app.post("/auth/validate", response_model=TokenValidateResponse)
async def validate_token(request: dict, db: Session = Depends(get_db)):
    """Validate chisel token and return credits info"""
    chisel_token = request.get("chisel_token")
    if not chisel_token:
        raise HTTPException(status_code=400, detail="chisel_token required")

    user = db.query(User).filter(User.chisel_token == chisel_token).first()
    if not user:
        return TokenValidateResponse(valid=False)

    # Check for 80% warning
    warning = None
    if user.credits_remaining <= Decimal("2.00") and not user.warned_80_percent:
        warning = f"⚠️ Low credits: ${user.credits_remaining:.2f} remaining"
        user.warned_80_percent = True
        db.commit()

    return TokenValidateResponse(
        valid=True,
        credits_remaining=float(user.credits_remaining),
        warning=warning,
    )


@app.post("/auth/do-token", response_model=DOTokenResponse)
async def get_do_token(request: dict, db: Session = Depends(get_db)):
    """Exchange chisel token for DO token if user has credits"""
    chisel_token = request.get("chisel_token")
    gpu_type = request.get("gpu_type", "nvidia-h100")

    if not chisel_token:
        raise HTTPException(status_code=400, detail="chisel_token required")

    user = db.query(User).filter(User.chisel_token == chisel_token).first()
    if not user:
        return DOTokenResponse(error="Invalid token")

    # Check if user has enough credits for at least 1 hour
    estimated_cost = GPU_RATES.get(gpu_type, GPU_RATES["nvidia-h100"])
    if user.credits_remaining < estimated_cost:
        return DOTokenResponse(
            error=f"Insufficient credits. Need ${estimated_cost}/hour, have ${user.credits_remaining}"
        )

    return DOTokenResponse(do_token=DO_TOKEN)


@app.post("/usage/track")
async def track_usage(
    request: UsageTrackRequest, db: Session = Depends(get_db)
):
    """Log usage and deduct from user credits"""
    user = (
        db.query(User).filter(User.chisel_token == request.chisel_token).first()
    )
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Calculate cost
    hourly_rate = GPU_RATES.get(request.gpu_type, GPU_RATES["nvidia-h100"])
    hours = Decimal(request.duration_minutes) / Decimal("60")
    cost = hours * hourly_rate

    # Deduct from credits
    user.credits_remaining -= cost
    user.total_spent += cost

    # Log usage
    usage = UsageHistory(
        chisel_token=request.chisel_token,
        droplet_id=request.droplet_id,
        gpu_type=request.gpu_type,
        duration_minutes=request.duration_minutes,
        cost=cost,
    )

    db.add(usage)
    db.commit()

    return {
        "success": True,
        "cost": float(cost),
        "credits_remaining": float(user.credits_remaining),
    }


@app.get("/users/{chisel_token}/status")
async def get_user_status(chisel_token: str, db: Session = Depends(get_db)):
    """Get user status - for debugging"""
    user = db.query(User).filter(User.chisel_token == chisel_token).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "chisel_token": user.chisel_token,
        "credits_remaining": float(user.credits_remaining),
        "total_spent": float(user.total_spent),
        "created_at": user.created_at,
    }

@app.post("/users/add")
async def add_user(request: dict, db: Session = Depends(get_db)):
    """Add a new user with credits"""
    chisel_token = request.get("chisel_token")
    credits = request.get("credits_remaining", 10.00)
    
    if not chisel_token:
        raise HTTPException(status_code=400, detail="chisel_token required")
    
    # Check if user already exists
    existing = db.query(User).filter(User.chisel_token == chisel_token).first()
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")
    
    user = User(chisel_token=chisel_token, credits_remaining=Decimal(str(credits)))
    db.add(user)
    db.commit()
    
    return {
        "success": True, 
        "chisel_token": chisel_token, 
        "credits_remaining": float(credits),
        "message": f"User {chisel_token} created with ${credits:.2f} credits"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
