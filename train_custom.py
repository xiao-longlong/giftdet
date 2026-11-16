# train_custom.py
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("/home/wxl/kuaishou/GiftDetect/ultralytics/train/cfg/yolov11tinyscale.yaml")

    result = model.train(
        data="/home/wxl/kuaishou/GiftDetect/ultralytics/train/yaml/giftyolo.yaml",
        epochs=500,
        imgsz=1080,
        batch=1,
        device=0,
        name="522_try",
        project="runs/train",
        cls=1,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.015,  # 色调变化：小幅扰动，防止改变物体主色调
        hsv_s=0.7,  # 饱和度扰动：中等强度，增强颜色多样性
        hsv_v=0.4,  # 明度扰动：适中，模拟亮度差异
        flipud=0.0,
        fliplr=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        auto_augment="",  # 禁用自动增强策略
    )
