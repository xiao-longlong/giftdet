import os
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt

from ultralytics import YOLO


# === IOU计算函数 ===
def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa1, ya1, xa2, ya2 = x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2
    xb1, yb1, xb2, yb2 = x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2

    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area, box2_area = w1 * h1, w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


# === 1. 保存带预测框的可视化图像 ===
def visualize_and_save(model, image_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        results = model.predict(source=image_path, conf=0.25, verbose=False)
        for r in results:
            annotated_img = r.plot()
            save_path = os.path.join(save_dir, image_file)
            cv2.imwrite(save_path, annotated_img)
            print(f"Saved: {save_path}")


# === 2. 分类正确/错误/漏检/误检统计并输出文本 ===
def evaluate_and_save_stats(model, image_dir, label_dir, save_dir):
    os.makedirs(os.path.join(save_dir, "data"), exist_ok=True)

    accuracy_file = os.path.join(save_dir, "data", "detection_class_accuracy.txt")
    miss_file = os.path.join(save_dir, "data", "missed_detections.txt")
    false_file = os.path.join(save_dir, "data", "false_detections.txt")

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    stats = defaultdict(lambda: {"total": 0, "correct": 0, "wrong": 0, "missed": 0, "false": 0})

    with open(accuracy_file, "w") as acc_f, open(miss_file, "w") as miss_f, open(false_file, "w") as false_f:
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            label_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + ".txt")
            if not os.path.exists(label_path):
                continue

            results = model.predict(source=image_path, conf=0.25, verbose=False)
            pred = results[0].boxes
            pred_classes = pred.cls.cpu().numpy().astype(int)
            pred_boxes = pred.xywhn.cpu().numpy()

            gt_boxes = []
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    cls = int(parts[0])
                    box = list(map(float, parts[1:]))
                    gt_boxes.append((cls, box))

            total = len(gt_boxes)
            correct = 0
            wrong_preds, missed, false = [], [], []
            matched_preds, matched_gts = set(), set()

            for i, (gt_cls, gt_box) in enumerate(gt_boxes):
                stats[gt_cls]["total"] += 1
                found = False
                for j, (pred_cls, pred_box) in enumerate(zip(pred_classes, pred_boxes)):
                    if j in matched_preds:
                        continue
                    iou_val = iou(gt_box, pred_box)
                    if iou_val > 0.3:
                        matched_preds.add(j)
                        matched_gts.add(i)
                        found = True
                        if pred_cls == gt_cls:
                            correct += 1
                            stats[gt_cls]["correct"] += 1
                        else:
                            wrong_preds.append((pred_cls, gt_cls, *gt_box))
                            stats[gt_cls]["wrong"] += 1
                        break
                if not found:
                    missed.append((gt_cls, *gt_box))
                    stats[gt_cls]["missed"] += 1

            for j, (pred_cls, pred_box) in enumerate(zip(pred_classes, pred_boxes)):
                if j not in matched_preds:
                    false.append((pred_cls, *pred_box))
                    stats[pred_cls]["false"] += 1

            # 写入准确性统计
            acc_f.write(f"{image_file} {total} {correct} {correct / total:.4f}")
            for item in wrong_preds:
                acc_f.write(" " + " ".join(map(str, item)))
            acc_f.write("\n")

            # 写入漏检信息
            miss_f.write(f"{image_file} {total} {len(missed)} {len(missed) / total:.4f}")
            for item in missed:
                miss_f.write(" " + " ".join(map(str, item)))
            miss_f.write("\n")

            # 写入误检信息
            false_total = len(pred_boxes)
            false_f.write(
                f"{image_file} {false_total} {len(false)} {len(false) / false_total:.4f}"
                if false_total > 0
                else f"{image_file} 0 0 0.0000"
            )
            for item in false:
                false_f.write(" " + " ".join(map(str, item)))
            false_f.write("\n")

    return stats


# === 3. 绘图统计 ===
def plot_class_statistics(stats, save_dir):
    def plot(key, title, filename, color):
        from matplotlib import colors as mcolors

        classes = sorted(stats.keys())
        totals = [stats[c]["total"] for c in classes]
        values = [stats[c][key] for c in classes]

        base_color = mcolors.to_rgba(color, alpha=0.3)
        main_color = mcolors.to_rgba(color, alpha=1.0)

        plt.figure(figsize=(12, 6))
        plt.bar(classes, totals, color=base_color, label="Total")
        plt.bar(classes, values, color=main_color, label=key.capitalize())
        plt.xlabel("Class ID")
        plt.ylabel("Count")
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.savefig(os.path.join(save_dir, "data", filename))
        plt.close()

    plot("correct", "Correct Detections per Class", "correct.png", "green")
    plot("wrong", "Wrong Class Predictions per Class", "wrong.png", "orange")
    plot("missed", "Missed Detections per Class", "missed.png", "red")
    plot("false", "False Detections per Class", "false.png", "blue")


# === 主执行函数 ===
def run_inference_and_evaluation(model_path, image_dir, label_dir, save_dir):
    model = YOLO(model_path)
    visualize_and_save(model, image_dir, save_dir)
    stats = evaluate_and_save_stats(model, image_dir, label_dir, save_dir)
    plot_class_statistics(stats, save_dir)


# === 主函数入口 ===
if __name__ == "__main__":
    # 修改此处路径以适配你的目录结构
    model_path = "/home/wxl/kuaishou/GiftDetect/ultralytics/runs/train/521_saved_train/weights/best.pt"
    image_dir = "/home/wxl/kuaishou/GiftDetect/dataset/giftyolo/val/images"
    label_dir = "/home/wxl/kuaishou/GiftDetect/dataset/giftyolo/val/labels"
    save_dir = "/home/wxl/kuaishou/GiftDetect/ultralytics/vis/val"

    run_inference_and_evaluation(model_path, image_dir, label_dir, save_dir)
