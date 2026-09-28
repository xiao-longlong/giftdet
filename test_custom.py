import os

import cv2

from ultralytics import YOLO

# 加载训练好的模型
model = YOLO("/home/wxl/kuaishou/GiftDetect/ultralytics/runs/train/exp_custom2/weights/best.pt")

# 推理源目录
source_dir = "/home/wxl/kuaishou/GiftDetect/ultralytics/datasets/giftyolo/test/images"
# 保存结果的目录
save_dir = "/home/wxl/kuaishou/GiftDetect/ultralytics/runs/train/exp_custom2/inference_results"
os.makedirs(save_dir, exist_ok=True)

# 获取所有图片路径
image_files = [f for f in os.listdir(source_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

# 遍历图像进行推理并保存结果
for image_file in image_files:
    image_path = os.path.join(source_dir, image_file)
    results = model.predict(source=image_path, conf=0.25, verbose=False)

    # 获取带框的图像（r.plot() 是 OpenCV 格式）
    for r in results:
        annotated_img = r.plot()
        save_path = os.path.join(save_dir, image_file)
        cv2.imwrite(save_path, annotated_img)
        print(f"Saved: {save_path}")
