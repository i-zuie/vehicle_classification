from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")  # 加载预训练模型
model.train(data="data",  # 数据集配置文件
            epochs=10,
            imgsz=640,
            batch=64,
            device='0'
            ) 
model.save("yolo/model_saved/yolov8_best_model.pt")