from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset/area/data.yaml",
        epochs=100,
        imgsz=1280,
        batch=16,
        device=0,
        workers=8,
        project="runs",
        name="area_a0a1"
    )