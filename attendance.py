import cv2
import os
import glob
import sqlite3
from datetime import datetime, date
import pandas as pd

CASCADE_PATH = "haarcascade_frontalface_default.xml"
TRAINER_PATH = "trainer/trainer.yml"
DB_NAME = "attendance.db"


# ------------------ HELPER FUNCTIONS ------------------ #

def delete_student(student_id):
    """Delete a student completely: photos + attendance + student record."""
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    # Delete attendance records of student
    cur.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))

    # Delete student record
    cur.execute("DELETE FROM students WHERE id = ?", (student_id,))

    conn.commit()
    conn.close()

    # Delete face images from dataset folder
    files = glob.glob(f"dataset/user.{student_id}.*.jpg")
    for f in files:
        os.remove(f)

    print(f"[INFO] Student {student_id} deleted successfully — photos + attendance removed.")
    print("[INFO] After deleting a student, run train_model.py again to update the model.")


def get_student_name(student_id):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT name FROM students WHERE id = ?", (student_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return row[0]
    return None


def mark_attendance(student_id):
    today = date.today().isoformat()
    current_time = datetime.now().strftime("%H:%M:%S")

    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    # Check if attendance already marked today
    cur.execute("""
        SELECT * FROM attendance
        WHERE student_id = ? AND date = ?
    """, (student_id, today))

    already = cur.fetchone()

    if already:
        conn.close()
        return False
    else:
        cur.execute("""
            INSERT INTO attendance (student_id, timestamp, date)
            VALUES (?, ?, ?)
        """, (student_id, current_time, today))
        conn.commit()
        conn.close()
        return True


def export_attendance_to_excel(date_str):
    """Export attendance for a given date (YYYY-MM-DD) to Excel file."""
    # Ensure exports folder exists
    if not os.path.exists("exports"):
        os.makedirs("exports")

    conn = sqlite3.connect(DB_NAME)
    query = """
        SELECT a.id, s.id as student_id, s.name, a.date, a.timestamp
        FROM attendance a
        JOIN students s ON a.student_id = s.id
        WHERE a.date = ?
        ORDER BY a.timestamp
    """
    df = pd.read_sql_query(query, conn, params=(date_str,))
    conn.close()

    if df.empty:
        print("[INFO] No attendance data for that date.")
        return

    filename = f"exports/attendance_{date_str}.xlsx"
    df.to_excel(filename, index=False)
    print(f"[INFO] Attendance exported to {filename}")


# ------------------ MAIN PROGRAM ------------------ #

if __name__ == "__main__":

    if not os.path.exists(TRAINER_PATH):
        print("[ERROR] Trainer file not found. Run train_model.py first.")
        exit(0)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_PATH)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    cam = cv2.VideoCapture(0)

    print("[INFO] Starting attendance. Press 'q' to quit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(100, 100)
        )

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            student_id, confidence = recognizer.predict(face_img)

            # Lower confidence = better match
            if confidence < 70:
                name = get_student_name(student_id)
                if name:
                    newly_marked = mark_attendance(student_id)
                    label = f"{name}"

                    if newly_marked:
                        print(f"[ATTENDANCE] Marked for {name}")
                else:
                    label = "Unknown"
            else:
                name = "Unknown"
                label = "Unknown"

            # Draw rectangle & label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Face Recognition Attendance - Press q to Quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    # After camera closes, ask for export and optional delete
    choice = input("Do you want to export today's attendance to Excel? (y/n): ").strip().lower()
    if choice == 'y':
        today_str = date.today().isoformat()
        export_attendance_to_excel(today_str)

    # Optional: ask if you want to delete a student
    del_choice = input("Do you want to delete any student? (y/n): ").strip().lower()
    if del_choice == 'y':
        try:
            sid = int(input("Enter Student ID to delete: ").strip())
            delete_student(sid)
        except ValueError:
            print("[ERROR] Invalid ID. Must be an integer.")
