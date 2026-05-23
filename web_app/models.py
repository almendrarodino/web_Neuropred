from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    dni = Column(String, nullable=True)
    profession = Column(String, nullable=True)
    email = Column(String, nullable=True, index=True)

    patients = relationship("Patient", back_populates="doctor")

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    full_name = Column(String, index=True)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    age = Column(Integer)
    dni = Column(String, nullable=True, index=True)
    email = Column(String, nullable=True)
    notes = Column(String, nullable=True)

    doctor = relationship("User", back_populates="patients")
    predictions = relationship("Prediction", back_populates="patient", cascade="all, delete-orphan")

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    image_path = Column(String)
    original_image_path = Column(String, nullable=True)
    original_filename = Column(String, nullable=True)
    source_format = Column(String, nullable=True)
    p_tumor = Column(Float)
    final_class = Column(String)
    probabilities_json = Column(String) # Store JSON string of the probabilities dictionary
    timestamp = Column(DateTime, default=datetime.utcnow)
    study_notes = Column(String, nullable=True)

    patient = relationship("Patient", back_populates="predictions")
