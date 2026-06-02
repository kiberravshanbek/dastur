# ============================================================
# A0/A1 YOLO MODELLARINI GOOGLE COLAB DA O'QITISH
# ============================================================
# 1. https://colab.research.google.com ga kiring
# 2. Runtime -> Change runtime type -> GPU (T4) tanlang
# 3. dataset.zip ni Colab ga yuklang (chap paneldagi papka belgisi)
# 4. Quyidagi kodni katakka qo'yib ishga tushiring
# ============================================================

# --- 1-katak: O'rnatish va ochish ---
!pip install ultralytics -q
!unzip -o dataset.zip -d /content/ > /dev/null
print("Tayyor")

# --- 2-katak: data.yaml yo'llarini Colab uchun tuzatish ---
import re
for ds in ['area', 'circle']:
    path = f'/content/data/{ds}/data.yaml'
    txt = open(path).read()
    txt = re.sub(r'^path:.*$', f'path: /content/data/{ds}', txt, flags=re.M)
    open(path, 'w').write(txt)
    print(open(path).read())

# --- 3-katak: CIRCLE modelini o'qitish (doirachalar) ---
from ultralytics import YOLO
model_c = YOLO('yolov8n.pt')
model_c.train(
    data='/content/data/circle/data.yaml',
    epochs=60,
    imgsz=1280,      # doirachalar kichik - katta o'lcham kerak
    batch=8,
    device=0,        # GPU
    project='/content/runs', name='circle', exist_ok=True,
    patience=20,
)
print("CIRCLE TUGADI:", model_c.trainer.best)

# --- 4-katak: AREA modelini o'qitish (bloklar) ---
model_a = YOLO('yolov8n.pt')
model_a.train(
    data='/content/data/area/data.yaml',
    epochs=60,
    imgsz=960,
    batch=16,
    device=0,
    project='/content/runs', name='area', exist_ok=True,
    patience=20,
)
print("AREA TUGADI:", model_a.trainer.best)

# --- 5-katak: Modellarni yuklab olish ---
from google.colab import files
import shutil
shutil.copy('/content/runs/circle/weights/best.pt', '/content/circle_a0a1.pt')
shutil.copy('/content/runs/area/weights/best.pt', '/content/area_a0a1.pt')
files.download('/content/circle_a0a1.pt')
files.download('/content/area_a0a1.pt')
# Bu ikkala faylni dasturning models/ papkasiga qo'ying:
#   circle_a0a1.pt -> models/circle_a0a1/best.pt
#   area_a0a1.pt   -> models/area_a0a1/best.pt
