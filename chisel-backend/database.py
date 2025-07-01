import os
from decimal import Decimal
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, User

# Use PostgreSQL if available, fallback to SQLite for local development
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chisel.db")

# Handle SQLite-specific connection args
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Create tables and add demo users"""
    Base.metadata.create_all(bind=engine)
    
    # Ensure demo users always exist
    db = SessionLocal()
    try:
        demo_users_data = [
            ("chisel-demo-001", 10.00),
            ("chisel-demo-002", 10.00),
            ("chisel-demo-003", 10.00),
            ("aseem", 20.00),  # Add your regular users here
        ]
        
        for token, credits in demo_users_data:
            existing_user = db.query(User).filter(User.chisel_token == token).first()
            if not existing_user:
                user = User(chisel_token=token, credits_remaining=Decimal(str(credits)))
                db.add(user)
                print(f"✅ Created user: {token} with ${credits} credits")
        
        db.commit()
    finally:
        db.close()

def get_db():
    """Dependency for FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()