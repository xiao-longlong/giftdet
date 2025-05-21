# train_custom.py
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.yaml").load("yolo11n.pt")

    result = model.train(
        data="/home/wxl/kuaishou/GiftDetect/ultralytics/train/yaml/giftyolo.yaml",
        epochs=500,
        imgsz=1080,
        batch=2,
        device=0,
        name="521_saved",
        project="runs/train",

        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        flipud=0.0,
        fliplr=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        auto_augment='',  # 禁用自动增强策略
    )
