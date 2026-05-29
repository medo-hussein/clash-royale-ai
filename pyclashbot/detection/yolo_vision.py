import numpy as np
from ultralytics import YOLO
from os.path import abspath, dirname, join
import os

# تحديد مسار ملف الموديل بدقة جوه المشروع
current_dir = dirname(abspath(__file__))
model_path = join(current_dir, "models", "yolov8_cr.pt")

# تحميل الموديل مرة واحدة بس في الذاكرة عشان ميبطأش اللعب
# ضفنا السطر ده عشان نتأكد إن الموديل موجود قبل ما يحمله
if not os.path.exists(model_path):
    print(f"⚠️ تحذير: ملف الموديل مش موجود في المسار: {model_path}")
else:
    model = YOLO(model_path)

def get_yolo_predictions(image: np.ndarray):
    """
    الدالة دي بتاخد سكرين شوت من اللعبة، وتدخلها للموديل،
    وترجع قائمة بالكروت والقوات اللي الموديل شافها.
    """
    if not os.path.exists(model_path):
        return []

    # تشغيل الموديل على الصورة
    results = model.predict(source=image, conf=0.5, verbose=False)
    
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # إحداثيات المربع
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # نسبة التأكد
            conf = float(box.conf[0])
            # اسم الكارت أو القوات
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            
            detections.append({
                "name": cls_name,
                "confidence": conf,
                "box": (int(x1), int(y1), int(x2), int(y2)),
                "center": (int((x1+x2)/2), int((y1+y2)/2))
            })
    return detections