from sqlalchemy import Boolean, Column, DateTime, Integer, String, Numeric
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    chisel_token = Column(String, primary_key=True)
    credits_remaining = Column(Numeric(10, 2), default=10.00)
    total_spent = Column(Numeric(10, 2), default=0.00)
    warned_80_percent = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())


class ActiveDroplet(Base):
    __tablename__ = "active_droplets"

    do_droplet_id = Column(String, primary_key=True)
    chisel_token = Column(String, nullable=False)
    gpu_type = Column(String, nullable=False)
    hourly_rate = Column(Numeric(10, 2), nullable=False)
    created_at = Column(DateTime, default=func.now())
    last_activity = Column(DateTime, default=func.now())


class UsageHistory(Base):
    __tablename__ = "usage_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chisel_token = Column(String, nullable=False)
    droplet_id = Column(String, nullable=False)
    gpu_type = Column(String, nullable=False)
    duration_minutes = Column(Integer, nullable=False)
    cost = Column(Numeric(10, 2), nullable=False)
    created_at = Column(DateTime, default=func.now())
