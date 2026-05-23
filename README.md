# NeuroPred

Plataforma web local para gestionar pacientes, cargar estudios de resonancia cerebral y realizar predicciones con modelos VGG16 entrenados para detección y clasificación de tumores.

> Esta herramienta no reemplaza el criterio médico. Los resultados deben ser interpretados por un profesional de la salud.

## Requisitos

- Windows.
- Python 3.11 recomendado. No usar Python 3.14, porque TensorFlow no tiene paquete compatible para ese entorno.
- Git.
- Git LFS para descargar los pesos del modelo.

## Estructura importante

Los pesos deben quedar en estas rutas:

```text
models_binario_full/binary_vgg16_finetuned.weights.h5
models_stage2_final_full/stage2_vgg16_final_finetuned.weights.h5
```

Ambos archivos pesan mas de 100 MB, por eso el repositorio usa Git LFS.

## Instalación local

```powershell
git clone https://github.com/almendrarodino/web_Neuropred.git
cd web_Neuropred
git lfs pull
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\iniciar_web.ps1
```

Luego abrir:

```text
http://127.0.0.1:8000/
```

## Variables opcionales

Para un uso real, conviene definir una clave propia para los tokens:

```powershell
$env:SECRET_KEY="cambiar-por-una-clave-larga-y-privada"
```

Si no se define, la app usa una clave de desarrollo local.

## Qué no se sube a GitHub

La configuración del repositorio excluye automáticamente:

- `.venv/`
- `app.db`
- `web_app/uploads/`
- logs de ejecución
- archivos `.env`
- `smtp_config.env`

Esto evita subir datos de pacientes, estudios cargados, entornos virtuales y credenciales.

## Subir este proyecto a GitHub

Desde esta carpeta:

```powershell
git init
git lfs install
git add .
git commit -m "Initial NeuroPred platform"
git branch -M main
git remote add origin https://github.com/almendrarodino/web_Neuropred.git
git push -u origin main
```

Antes del `push`, crear en GitHub un repositorio vacío llamado `web_Neuropred` o el nombre que prefieras.
