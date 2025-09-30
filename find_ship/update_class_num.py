import os

# --- הגדרות ---
# הנתיב לתיקייה המכילה את קבצי ה-TXT של YOLO.
# לדוגמה: D:\ofir\detection_results\ships_detection\labels
LABELS_DIR = r"D:\ofir\create_ship\output1"

# ה-Class ID החדש שברצונך להגדיר (לדוגמה: 1)
NEW_CLASS_ID = "0"


# ----------------

def update_yolo_class_ids(labels_dir, new_class_id):
    """
    עובר על כל קבצי ה-TXT בתיקייה ומשנה את מזהה ה-class (המספר הראשון)
    בכל שורה ל-NEW_CLASS_ID.
    """
    files_processed = 0

    print(f"מתחיל עיבוד קבצים בנתיב: {labels_dir}...")

    # עובר על כל הקבצים בתיקייה
    for filename in os.listdir(labels_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(labels_dir, filename)
            updated_lines = []

            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                # עובר על כל שורה בקובץ
                for line in lines:
                    parts = line.strip().split()

                    if parts:
                        # מחליף את המספר הראשון (class ID) ב-ID החדש
                        # שאר הקואורדינטות (x_center, y_center, width, height) נשמרות
                        new_line = f"{new_class_id} {' '.join(parts[1:])}\n"
                        updated_lines.append(new_line)

                # כותב את התוכן המעודכן חזרה לקובץ
                with open(file_path, 'w') as f:
                    f.writelines(updated_lines)

                files_processed += 1

            except Exception as e:
                print(f"שגיאה בעיבוד קובץ {filename}: {e}")

    print("-" * 30)
    print(f"✅ סיום! {files_processed} קבצי TXT עובדו בהצלחה.")
    print(f"כל מזהי ה-Class הוחלפו ל: {new_class_id}")


# הרצת הפונקציה
update_yolo_class_ids(LABELS_DIR, NEW_CLASS_ID)