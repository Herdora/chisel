import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, User

# Use SQLite file
DATABASE_URL = "sqlite:///./chisel.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Create tables and add demo users"""
    Base.metadata.create_all(bind=engine)
    
    # Add demo users
    db = SessionLocal()
    try:
        # Check if demo users already exist
        existing_user = db.query(User).filter(User.chisel_token == "chisel-demo-001").first()
        if not existing_user:
            demo_users = [
                User(chisel_token="chisel-demo-001", credits_remaining=10.00),
                User(chisel_token="chisel-demo-002", credits_remaining=10.00),
                User(chisel_token="chisel-demo-003", credits_remaining=10.00),
            ]
            db.add_all(demo_users)
            db.commit()
            print("✅ Demo users created")
    finally:
        db.close()

def get_db():
    """Dependency for FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()