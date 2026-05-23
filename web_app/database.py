from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from web_app.models import Base

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "app.db"
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH.as_posix()}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
    migrate_sqlite()


def migrate_sqlite():
    user_columns = {
        "first_name": "VARCHAR",
        "last_name": "VARCHAR",
        "dni": "VARCHAR",
        "profession": "VARCHAR",
        "email": "VARCHAR",
    }
    patient_columns = {
        "first_name": "VARCHAR",
        "last_name": "VARCHAR",
        "dni": "VARCHAR",
        "email": "VARCHAR",
    }
    prediction_columns = {
        "original_image_path": "VARCHAR",
        "original_filename": "VARCHAR",
        "source_format": "VARCHAR",
    }

    with engine.begin() as conn:
        existing_users = {row[1] for row in conn.execute(text("PRAGMA table_info(users)"))}
        for column, definition in user_columns.items():
            if column not in existing_users:
                conn.execute(text(f"ALTER TABLE users ADD COLUMN {column} {definition}"))

        existing_patients = {row[1] for row in conn.execute(text("PRAGMA table_info(patients)"))}
        for column, definition in patient_columns.items():
            if column not in existing_patients:
                conn.execute(text(f"ALTER TABLE patients ADD COLUMN {column} {definition}"))

        existing_predictions = {row[1] for row in conn.execute(text("PRAGMA table_info(predictions)"))}
        for column, definition in prediction_columns.items():
            if column not in existing_predictions:
                conn.execute(text(f"ALTER TABLE predictions ADD COLUMN {column} {definition}"))
