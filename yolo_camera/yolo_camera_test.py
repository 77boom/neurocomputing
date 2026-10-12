from ultralytics import YOLO

# 加载 COCO 预训练的 YOLOv8 nano 模型
model = YOLO("yolov8n.pt")

# 0 = Mac 默认摄像头
model.predict(
    source=1,
    show=True,
    conf=0.4
)