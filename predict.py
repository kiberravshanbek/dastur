from ultralytics import YOLO

model = YOLO(
    r"F:\YALO_A0A1\runs\detect\runs\circle_a0a1-4\weights\best.pt"
)

results = model.predict(
    source=r"F:\YALO_A0A1\test.jpg",
    imgsz=1280,
    conf=0.25,
    save=True
)

print("Prediction finished")