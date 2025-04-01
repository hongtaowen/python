from ultralytics import YOLO
if __name__=='__main__':

    # Load a model
    model = YOLO(r"D:\code pro1\pythonProject\gitbase\ultralytics\ultralytics\cfg\models\v8\yolov8n-seg.yaml")  # build a new model from YAML
    model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)
    # model = YOLO(r"D:\code pro1\pythonProject\ultralytics\ultralytics\cfg\models\v8\yolov8n-seg.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data=r"D:\code pro1\pythonProject\gitbase\ultralytics\ultralytics\cfg\datasets\coco128-seg.yaml", epochs=500, nms=True, imgsz=640)