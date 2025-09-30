import os
from ultralytics import YOLO

# הגדרות נתיבים
# נתיב המודל שיירד אוטומטית בעת הפעולה הראשונה
MODEL_NAME = 'yolov8m.pt'
# נתיב לתיקייה המכילה את התמונות שלך
SOURCE_DIR = r"D:\ofir\create_ship\output1"
# נתיב לתיקיית היעד, בה יישמרו התמונות עם המסגרות וקבצי ה-TXT
OUTPUT_DIR = r"D:\ofir\create_ship\output1"

# 1. טעינת המודל
model = YOLO(MODEL_NAME)

# 2. הרצת הזיהוי
# ה-output יישמר אוטומטית בתיקייה 'runs/detect/predictX'
# אנו מגדירים את התיקייה 'project' ואת התיקייה 'name' כדי לשלוט בנתיב היעד
results = model.predict(
    source=SOURCE_DIR,
    save=True,               # שמירת התוצאות (תמונות עם מסגרות)
    save_txt=True,           # שמירת הקואורדינטות כקבצי TXT בפורמט YOLO
    conf=0.5,                # רמת ביטחון מינימלית (50%)
    project=OUTPUT_DIR,      # תיקיית הפרויקט הראשי
    name='ships_detection',  # שם הריצה בתוך תיקיית הפרויקט
    exist_ok=True            # מאפשר לשמור בתיקייה קיימת
)

print(f"✅ הזיהוי הסתיים בהצלחה!")
print(f"קבצי ה-TXT נשמרו בתוך: {OUTPUT_DIR}\ships_detection\labels")