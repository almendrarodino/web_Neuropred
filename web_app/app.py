import os
import io
import json
import re
import zipfile
import tempfile
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from sqlalchemy import or_
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras as K

from web_app import models, database

# ================= CONFIGURACIÓN =================
SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret_key_change_for_production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # 1 day

IMG_SIZE = 224
ROI_MARGIN = 8
T_STAR = 0.0610
CLASS_NAMES_3 = ["meningioma", "glioma", "pituitary"]
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".mat", ".dcm", ".dicom", ".zip"}

BASE_DIR = Path(__file__).resolve().parent.parent
STAGE1_WEIGHTS = BASE_DIR / "models_binario_full" / "binary_vgg16_finetuned.weights.h5"
STAGE2_WEIGHTS = BASE_DIR / "models_stage2_final_full" / "stage2_vgg16_final_finetuned.weights.h5"
UPLOADS_DIR = BASE_DIR / "web_app" / "uploads"
LOGO_PATH = BASE_DIR / "web_app" / "static" / "logo.png"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Inicializar Base de datos
database.init_db()

# Inicializar FastAPi
app = FastAPI(title="Tumor Prediction API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "web_app" / "static")), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")


# ================= SEGURIDAD Y AUTH =================

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(database.get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(models.User).filter(models.User.username == username).first()
    if user is None:
        raise credentials_exception
    return user


# ================= MODELOS IA (Cargados en memoria) =================

stage1 = None
stage2 = None

def build_stage1_vgg16_binary():
    inp = K.Input((IMG_SIZE, IMG_SIZE, 3))
    base = K.applications.VGG16(include_top=False, weights=None, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = K.applications.vgg16.preprocess_input(inp)
    base.trainable = False
    x = base(x, training=False)
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dropout(0.3)(x)
    out = K.layers.Dense(1, activation="sigmoid")(x)
    return K.Model(inp, out)

def build_stage2_vgg16_3class():
    inp = K.Input((IMG_SIZE, IMG_SIZE, 3))
    base = K.applications.VGG16(include_top=False, weights=None, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = K.applications.vgg16.preprocess_input(inp)
    base.trainable = False
    x = base(x, training=False)
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dropout(0.3)(x)
    out = K.layers.Dense(3, activation="softmax")(x)
    return K.Model(inp, out)

def load_vgg16_weights_manually(model, weights_path: Path):
    import h5py

    if not weights_path.exists():
        raise FileNotFoundError(f"No se encontraron los pesos del modelo: {weights_path}")

    with h5py.File(weights_path, "r") as f:
        root_keys = list(f.keys())
        sep = "\\" if any("\\" in key for key in root_keys) else "/"

        vgg_base = next(
            (layer for layer in model.layers if layer.name.startswith("vgg16")),
            None
        )
        conv_layers = [
            layer for layer in (vgg_base.layers if vgg_base is not None else model.layers)
            if isinstance(layer, K.layers.Conv2D)
        ]
        dense_layers = [
            layer for layer in model.layers
            if isinstance(layer, K.layers.Dense)
        ]

        conv_keys = ["conv2d"] + [f"conv2d_{i}" for i in range(1, 13)]
        for layer, key in zip(conv_layers, conv_keys):
            group = f[f"layers{sep}functional{sep}layers{sep}{key}"]["vars"]
            layer.set_weights([group["0"][()], group["1"][()]])

        if not dense_layers:
            raise ValueError("No se encontro la capa Dense final del modelo.")

        dense_group = f[f"layers{sep}dense"]["vars"]
        dense_layers[0].set_weights([dense_group["0"][()], dense_group["1"][()]])

@app.on_event("startup")
def load_models():
    global stage1, stage2
    print("Cargando modelo Stage 1 VGG16...")
    stage1 = build_stage1_vgg16_binary()
    load_vgg16_weights_manually(stage1, STAGE1_WEIGHTS)
    
    print("Cargando modelo Stage 2 VGG16...")
    stage2 = build_stage2_vgg16_3class()
    load_vgg16_weights_manually(stage2, STAGE2_WEIGHTS)
    print("Modelos cargados exitosamente.")

# ================= PREPROCESAMIENTO IA =================

def zscore(img: np.ndarray):
    img = img.astype(np.float32)
    return (img - img.mean()) / (img.std() + 1e-6)

def to_rgb_resized(img2d: np.ndarray):
    x = tf.convert_to_tensor(img2d[..., None])
    x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE))
    x = tf.repeat(x, repeats=3, axis=-1)
    return x.numpy().astype(np.float32)

def crop_roi(image2d: np.ndarray, mask2d: np.ndarray, margin=8):
    if mask2d is None or mask2d.max() == 0:
        return image2d
    ys, xs = np.where(mask2d > 0)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    y0 = max(0, y0 - margin); y1 = min(image2d.shape[0]-1, y1 + margin)
    x0 = max(0, x0 - margin); x1 = min(image2d.shape[1]-1, x1 + margin)
    return image2d[y0:y1+1, x0:x1+1]

def normalize_to_uint8(img: np.ndarray):
    img = img.astype(np.float32)
    low, high = np.percentile(img, [1, 99])
    if high <= low:
        low, high = float(img.min()), float(img.max())
    if high <= low:
        return np.zeros(img.shape, dtype=np.uint8)
    img = np.clip((img - low) / (high - low), 0, 1)
    return (img * 255).astype(np.uint8)

def dicom_text_score(ds):
    parts = [
        str(getattr(ds, "SeriesDescription", "")),
        str(getattr(ds, "ProtocolName", "")),
        str(getattr(ds, "SequenceName", "")),
        " ".join(str(v) for v in getattr(ds, "ImageType", [])),
    ]
    text = " ".join(parts).lower()
    score = 0
    for token in ["t1", "t1c", "post", "contrast", "contraste", "gd", "gad", "gadolinium", "+c", "ce"]:
        if token in text:
            score += 2
    for token in ["t2", "flair", "adc", "dwi", "diff", "swi", "tirm"]:
        if token in text:
            score -= 3
    return score

def dicom_instance_number(ds):
    try:
        return int(getattr(ds, "InstanceNumber", 0))
    except (TypeError, ValueError):
        return 0

def dicom_to_uint8(ds):
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[arr.shape[0] // 2]
    if arr.ndim > 2:
        arr = arr[..., 0]

    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    arr = arr * slope + intercept

    center = getattr(ds, "WindowCenter", None)
    width = getattr(ds, "WindowWidth", None)
    try:
        if isinstance(center, (list, tuple)):
            center = center[0]
        if isinstance(width, (list, tuple)):
            width = width[0]
        center = float(center)
        width = float(width)
        low, high = center - width / 2, center + width / 2
        if high > low:
            arr = np.clip((arr - low) / (high - low), 0, 1) * 255
            out = arr.astype(np.uint8)
        else:
            out = normalize_to_uint8(arr)
    except (TypeError, ValueError):
        out = normalize_to_uint8(arr)

    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        out = 255 - out
    return out

def read_dicom_from_bytes(file_bytes: bytes):
    import pydicom

    return pydicom.dcmread(io.BytesIO(file_bytes), force=True)

def choose_dicom_from_zip(file_bytes: bytes):
    import pydicom

    candidates = []
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
        for name in archive.namelist():
            if name.endswith("/") or "__MACOSX" in name:
                continue
            try:
                raw = archive.read(name)
                ds = pydicom.dcmread(io.BytesIO(raw), force=True)
                if not hasattr(ds, "PixelData"):
                    continue
                candidates.append((name, ds))
            except Exception:
                continue

    if not candidates:
        raise ValueError("No se encontraron imagenes DICOM validas dentro del ZIP.")

    series = {}
    for name, ds in candidates:
        key = str(getattr(ds, "SeriesInstanceUID", name))
        series.setdefault(key, []).append((name, ds))

    def series_score(items):
        return max(dicom_text_score(ds) for _, ds in items)

    chosen_items = max(series.values(), key=series_score)
    chosen_items = sorted(chosen_items, key=lambda item: dicom_instance_number(item[1]))
    return chosen_items[len(chosen_items) // 2][1]

def save_dicom_as_jpeg(file_bytes: bytes, file_ext: str, output_path: Path):
    ds = choose_dicom_from_zip(file_bytes) if file_ext == ".zip" else read_dicom_from_bytes(file_bytes)
    img = dicom_to_uint8(ds)
    Image.fromarray(img).convert("L").save(output_path, "JPEG", quality=95)
    return img

def save_prediction_preview(x: np.ndarray, output_path: Path):
    x_min, x_max = x.min(), x.max()
    x_norm = (x - x_min) / (x_max - x_min) if x_max > x_min else x
    Image.fromarray((x_norm * 255).astype(np.uint8)).save(output_path)

def preprocess_image(file_bytes: bytes, file_ext: str):
    import h5py
    from scipy.io import loadmat
    
    # Para mat, necesitamos el archivo en disco
    if file_ext == ".mat":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mat") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        try:
            with h5py.File(tmp_path, "r") as f:
                cj = f["cjdata"]
                img = np.array(cj["image"]).T
                mask = (np.array(cj["tumorMask"]).T > 0).astype(np.uint8)
        except OSError:
            mat = loadmat(tmp_path, squeeze_me=True, struct_as_record=False)
            cj = mat["cjdata"]
            img = np.array(cj.image)
            mask = (np.array(cj.tumorMask) > 0).astype(np.uint8)
            
        os.remove(tmp_path)
        img = crop_roi(img, mask, margin=ROI_MARGIN)
        img = zscore(img)
        return to_rgb_resized(img)
    elif file_ext in [".dcm", ".dicom"]:
        img = dicom_to_uint8(read_dicom_from_bytes(file_bytes))
        img = zscore(img)
        return to_rgb_resized(img)
    elif file_ext == ".zip":
        img = dicom_to_uint8(choose_dicom_from_zip(file_bytes))
        img = zscore(img)
        return to_rgb_resized(img)
    else:
        im = Image.open(io.BytesIO(file_bytes)).convert("L")
        img = zscore(np.array(im).astype(np.float32))
        return to_rgb_resized(img)

def predict_cascade(stage1, stage2, x):
    p_tumor = float(stage1.predict(x[None, ...], verbose=0).ravel()[0])

    if p_tumor < T_STAR:
        return p_tumor, "no_tumor", None

    probs3 = stage2.predict(x[None, ...], verbose=0).ravel()
    cls = int(np.argmax(probs3))
    subtype_probs = {CLASS_NAMES_3[i]: float(probs3[i]) for i in range(3)}
    return p_tumor, CLASS_NAMES_3[cls], subtype_probs


# ================= PREDANTIC SCHEMAS =================

class UserCreate(BaseModel):
    username: str
    password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    dni: Optional[str] = None
    profession: Optional[str] = None
    email: Optional[str] = None

class UserUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    dni: Optional[str] = None
    profession: Optional[str] = None
    email: Optional[str] = None

class PatientCreate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    full_name: Optional[str] = None
    age: int
    dni: Optional[str] = None
    email: Optional[str] = None
    notes: Optional[str] = None

class PatientUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    full_name: Optional[str] = None
    age: Optional[int] = None
    dni: Optional[str] = None
    email: Optional[str] = None
    notes: Optional[str] = None

class NotesUpdate(BaseModel):
    notes: Optional[str] = None

class PatientResponse(BaseModel):
    id: int
    user_id: int
    full_name: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    age: int
    dni: Optional[str] = None
    email: Optional[str] = None
    notes: Optional[str] = None

    class Config:
        from_attributes = True

def build_full_name(first_name: Optional[str], last_name: Optional[str], fallback: Optional[str] = None):
    joined = " ".join(part.strip() for part in [first_name or "", last_name or ""] if part and part.strip())
    if joined:
        return joined
    if fallback and fallback.strip():
        return fallback.strip()
    return "Paciente sin nombre"

def split_full_name(full_name: Optional[str]):
    if not full_name:
        return None, None
    parts = full_name.strip().split()
    if not parts:
        return None, None
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])

def patient_to_dict(patient: models.Patient):
    first_name = patient.first_name
    last_name = patient.last_name
    if not first_name and patient.full_name:
        first_name, last_name = split_full_name(patient.full_name)
    full_name = build_full_name(first_name, last_name, patient.full_name)
    return {
        "id": patient.id,
        "user_id": patient.user_id,
        "full_name": full_name,
        "first_name": first_name or "",
        "last_name": last_name or "",
        "age": patient.age,
        "dni": patient.dni or "",
        "email": patient.email or "",
        "notes": patient.notes,
    }

def user_to_dict(user: models.User):
    display_name = build_full_name(user.first_name, user.last_name, user.username)
    return {
        "id": user.id,
        "username": user.username,
        "first_name": user.first_name or "",
        "last_name": user.last_name or "",
        "dni": user.dni or "",
        "profession": user.profession or "",
        "email": user.email or "",
        "display_name": display_name,
    }

def get_owned_patient(patient_id: int, current_user: models.User, db: Session):
    patient = db.query(models.Patient).filter(
        models.Patient.id == patient_id,
        models.Patient.user_id == current_user.id
    ).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient

def relative_upload_path(path: Optional[str]):
    if not path:
        return None
    return path if path.startswith("/uploads/") else f"/uploads/{Path(path).name}"

def upload_file_from_relative(path: Optional[str]):
    if not path:
        return None
    name = Path(path).name
    candidate = UPLOADS_DIR / name
    try:
        candidate.resolve().relative_to(UPLOADS_DIR.resolve())
    except ValueError:
        return None
    return candidate

def delete_upload_if_exists(path: Optional[str]):
    candidate = upload_file_from_relative(path)
    if candidate and candidate.exists():
        try:
            candidate.unlink()
        except OSError:
            pass

def prediction_to_dict(prediction: models.Prediction):
    probs = json.loads(prediction.probabilities_json) if prediction.probabilities_json else None
    return {
        "id": prediction.id,
        "image_path": prediction.image_path,
        "original_image_path": prediction.original_image_path,
        "original_filename": prediction.original_filename,
        "source_format": prediction.source_format,
        "p_tumor": prediction.p_tumor,
        "final_class": prediction.final_class,
        "probabilities_json": probs,
        "timestamp": prediction.timestamp.isoformat(),
        "study_notes": prediction.study_notes,
    }

def class_label(final_class: str):
    labels = {
        "glioma": "Glioma",
        "meningioma": "Meningioma",
        "pituitary": "Tumor pituitario",
        "no_tumor": "Sin tumor evidente",
    }
    return labels.get(final_class, final_class or "Desconocido")

def generate_patient_report_pdf(patient: models.Patient, predictions: List[models.Prediction]):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import Image as ReportImage
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Muted", parent=styles["Normal"], textColor=colors.HexColor("#64748b")))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontSize=13, spaceBefore=12, spaceAfter=8))

    patient_data = patient_to_dict(patient)
    story = []
    header_cells = []
    if LOGO_PATH.exists():
        header_cells.append(ReportImage(str(LOGO_PATH), width=1.2 * cm, height=1.2 * cm))
    header_cells.append(Paragraph("<b>NeuroPred</b>", styles["Title"]))
    story.append(Table([header_cells], colWidths=[1.5 * cm, 15 * cm] if LOGO_PATH.exists() else [16.5 * cm]))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Datos del paciente", styles["Section"]))
    patient_rows = [
        ["Nombre", patient_data["first_name"] or "-"],
        ["Apellido", patient_data["last_name"] or "-"],
        ["Edad", str(patient_data["age"] or "-")],
        ["DNI", patient_data["dni"] or "-"],
        ["Email", patient_data["email"] or "-"],
        ["Notas clínicas", patient_data["notes"] or "-"],
    ]
    story.append(Table(patient_rows, colWidths=[4 * cm, 12 * cm], style=[
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f8fafc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))

    story.append(Paragraph("Diagnósticos registrados", styles["Section"]))
    if predictions:
        rows = [["Fecha", "Resultado", "P(Tumor)", "Notas"]]
        for prediction in predictions:
            rows.append([
                prediction.timestamp.strftime("%d/%m/%Y %H:%M"),
                class_label(prediction.final_class),
                f"{prediction.p_tumor * 100:.2f}%",
                prediction.study_notes or "-",
            ])
        table = Table(rows, colWidths=[3.2 * cm, 4.3 * cm, 2.5 * cm, 6 * cm], repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0ea5e9")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(table)
    else:
        story.append(Paragraph("No hay diagnósticos cargados para este paciente.", styles["Muted"]))

    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        "Este informe es generado automáticamente por NeuroPred y debe ser interpretado por un profesional de la salud.",
        styles["Muted"],
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ================= RUTAS =================

@app.get("/")
def read_root():
    return FileResponse(str(BASE_DIR / "web_app" / "static" / "index.html"))

@app.post("/register")
def register(user: UserCreate, db: Session = Depends(database.get_db)):
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    new_user = models.User(
        username=user.username,
        password_hash=hashed_password,
        first_name=user.first_name,
        last_name=user.last_name,
        dni=user.dni,
        profession=user.profession,
        email=user.email,
    )
    db.add(new_user)
    db.commit()
    return {"message": "User created successfully"}

@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/me")
def read_users_me(current_user: models.User = Depends(get_current_user)):
    return user_to_dict(current_user)

@app.put("/me")
def update_users_me(payload: UserUpdate, current_user: models.User = Depends(get_current_user), db: Session = Depends(database.get_db)):
    data = payload.model_dump(exclude_unset=True)
    for field in ["first_name", "last_name", "dni", "profession", "email"]:
        if field in data:
            setattr(current_user, field, data[field])
    db.commit()
    db.refresh(current_user)
    return user_to_dict(current_user)

@app.get("/patients")
def get_patients(
    search: Optional[str] = None, 
    tumor_type: Optional[str] = None,
    sort: Optional[str] = "recent",
    current_user: models.User = Depends(get_current_user), 
    db: Session = Depends(database.get_db)
):
    query = db.query(models.Patient).filter(models.Patient.user_id == current_user.id)
    
    if search:
        pattern = f"%{search}%"
        query = query.filter(or_(
            models.Patient.full_name.ilike(pattern),
            models.Patient.first_name.ilike(pattern),
            models.Patient.last_name.ilike(pattern),
            models.Patient.dni.ilike(pattern),
            models.Patient.email.ilike(pattern),
        ))

    sort_value = (sort or "recent").lower()
    if sort_value == "alphabetic":
        query = query.order_by(models.Patient.full_name.asc(), models.Patient.id.asc())
    else:
        query = query.order_by(models.Patient.id.desc())
        
    patients = query.all()
    
    # Filter by tumor type if requested (in predictions)
    if tumor_type:
        filtered_patients = []
        for p in patients:
            has_tumor = any(pred.final_class == tumor_type for pred in p.predictions)
            if has_tumor:
                filtered_patients.append(p)
        return [patient_to_dict(p) for p in filtered_patients]
        
    return [patient_to_dict(p) for p in patients]

@app.get("/patients/{patient_id}")
def get_patient(patient_id: int, current_user: models.User = Depends(get_current_user), db: Session = Depends(database.get_db)):
    patient = get_owned_patient(patient_id, current_user, db)
        
    predictions = db.query(models.Prediction).filter(models.Prediction.patient_id == patient_id).order_by(models.Prediction.timestamp.desc()).all()
        
    return {
        "patient": patient_to_dict(patient),
        "predictions": [prediction_to_dict(p) for p in predictions]
    }

@app.post("/patients")
def create_patient(patient: PatientCreate, current_user: models.User = Depends(get_current_user), db: Session = Depends(database.get_db)):
    first_name = patient.first_name
    last_name = patient.last_name
    if not first_name and patient.full_name:
        first_name, last_name = split_full_name(patient.full_name)
    db_patient = models.Patient(
        user_id=current_user.id,
        first_name=first_name,
        last_name=last_name,
        full_name=build_full_name(first_name, last_name, patient.full_name),
        age=patient.age,
        dni=patient.dni,
        email=patient.email,
        notes=patient.notes,
    )
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return patient_to_dict(db_patient)

@app.put("/patients/{patient_id}")
def update_patient(patient_id: int, payload: PatientUpdate, current_user: models.User = Depends(get_current_user), db: Session = Depends(database.get_db)):
    patient = get_owned_patient(patient_id, current_user, db)
    data = payload.model_dump(exclude_unset=True)

    if "full_name" in data and "first_name" not in data and "last_name" not in data:
        data["first_name"], data["last_name"] = split_full_name(data["full_name"])

    for field in ["first_name", "last_name", "age", "dni", "email", "notes"]:
        if field in data:
            setattr(patient, field, data[field])

    patient.full_name = build_full_name(patient.first_name, patient.last_name, data.get("full_name", patient.full_name))
    db.commit()
    db.refresh(patient)
    return patient_to_dict(patient)

@app.patch("/patients/{patient_id}/notes")
def update_patient_notes(patient_id: int, payload: NotesUpdate, current_user: models.User = Depends(get_current_user), db: Session = Depends(database.get_db)):
    patient = get_owned_patient(patient_id, current_user, db)
    patient.notes = payload.notes
    db.commit()
    db.refresh(patient)
    return patient_to_dict(patient)

@app.delete("/patients/{patient_id}")
def delete_patient(patient_id: int, current_user: models.User = Depends(get_current_user), db: Session = Depends(database.get_db)):
    patient = get_owned_patient(patient_id, current_user, db)
    paths = []
    for prediction in patient.predictions:
        paths.extend([prediction.image_path, prediction.original_image_path])
    db.delete(patient)
    db.commit()
    for path in set(filter(None, paths)):
        delete_upload_if_exists(path)
    return {"message": "Paciente eliminado"}

@app.post("/patients/{patient_id}/predict")
async def create_prediction(
    patient_id: int, 
    file: UploadFile = File(...), 
    notes: Optional[str] = Form(None),
    current_user: models.User = Depends(get_current_user), 
    db: Session = Depends(database.get_db)
):
    patient = get_owned_patient(patient_id, current_user, db)

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Formato de archivo no soportado. Use JPG, PNG, MAT, DICOM o ZIP DICOM.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_stem = f"patient_{patient_id}_{timestamp}"
    file_bytes = await file.read()

    if ext in [".dcm", ".dicom", ".zip"]:
        converted_filename = f"{safe_stem}_dicom.jpg"
        converted_path = UPLOADS_DIR / converted_filename
        try:
            save_dicom_as_jpeg(file_bytes, ext, converted_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"No se pudo procesar el archivo DICOM: {exc}")
        original_relative_path = f"/uploads/{converted_filename}"
    else:
        safe_filename = f"{safe_stem}{ext}"
        physical_path = UPLOADS_DIR / safe_filename
        with open(physical_path, "wb") as f:
            f.write(file_bytes)
        original_relative_path = f"/uploads/{safe_filename}"

    # Preprocesar y predecir
    try:
        if ext in [".dcm", ".dicom", ".zip"]:
            x = preprocess_image((UPLOADS_DIR / Path(original_relative_path).name).read_bytes(), ".jpg")
        else:
            x = preprocess_image(file_bytes, ext)
    except Exception as exc:
        delete_upload_if_exists(original_relative_path)
        raise HTTPException(status_code=400, detail=f"No se pudo preprocesar la imagen: {exc}")

    p_tumor, final_class, probs = predict_cascade(stage1, stage2, x)

    # Save preview image for UI display
    preview_filename = f"preview_{patient_id}_{timestamp}.png"
    preview_physical_path = UPLOADS_DIR / preview_filename
    save_prediction_preview(x, preview_physical_path)

    relative_preview_path = f"/uploads/{preview_filename}"

    # Guardar en base de datos
    probs_json = json.dumps(probs) if probs else "{}"
    
    prediction = models.Prediction(
        patient_id=patient.id,
        image_path=relative_preview_path,
        original_image_path=original_relative_path,
        original_filename=file.filename,
        source_format=ext.lstrip("."),
        p_tumor=p_tumor,
        final_class=final_class,
        probabilities_json=probs_json,
        study_notes=notes
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    return {
        "id": prediction.id,
        "p_tumor": p_tumor,
        "final_class": final_class,
        "probs": probs,
        "image_path": relative_preview_path,
        "original_image_path": original_relative_path,
        "timestamp": prediction.timestamp
    }

@app.delete("/patients/{patient_id}/predictions/{prediction_id}")
def delete_prediction(
    patient_id: int,
    prediction_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    get_owned_patient(patient_id, current_user, db)
    prediction = db.query(models.Prediction).filter(
        models.Prediction.id == prediction_id,
        models.Prediction.patient_id == patient_id
    ).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    paths = [prediction.image_path, prediction.original_image_path]
    db.delete(prediction)
    db.commit()
    for path in set(filter(None, paths)):
        delete_upload_if_exists(path)
    return {"message": "Diagnóstico eliminado"}

@app.get("/predictions/{prediction_id}/image/download")
def download_prediction_image(
    prediction_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    prediction = db.query(models.Prediction).join(models.Patient).filter(
        models.Prediction.id == prediction_id,
        models.Patient.user_id == current_user.id
    ).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    relative_path = prediction.original_image_path or prediction.image_path
    path = upload_file_from_relative(relative_path)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    filename = prediction.original_filename or path.name
    if prediction.source_format in ["dcm", "dicom", "zip"]:
        filename = f"{Path(filename).stem}_anonimizado.jpg"
    return FileResponse(str(path), filename=filename)

@app.get("/patients/{patient_id}/report")
def download_patient_report(
    patient_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    patient = get_owned_patient(patient_id, current_user, db)
    predictions = db.query(models.Prediction).filter(
        models.Prediction.patient_id == patient_id
    ).order_by(models.Prediction.timestamp.desc()).all()
    pdf_bytes = generate_patient_report_pdf(patient, predictions)
    filename = f"informe_neuropred_paciente_{patient_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
