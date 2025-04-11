from ultralytics import YOLO

model = YOLO("turing_canet_v2.pt")

# while True:
#     model.predict(source="test.png", show=True)

model.export(format="engine", imgsz=640, dynamic=False, half=True, device="cuda")