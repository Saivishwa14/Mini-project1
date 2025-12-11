import cv2
import os
import sqlite3


CASCADE_PATH = "haarcascade_frontalface_default.xml"
DATASET_DIR = "dataset"
DB_NAME = "attendance.db"


def init_db():
    """Create database and tables if not exist."""
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        )
    """)
    
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            timestamp TEXT,
            date TEXT,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    """)
    
    conn.commit()
    conn.close()

def insert_or_update_student(student_id, name):
    """Insert new student or update if ID already exists."""
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    
    cur.execute("SELECT * FROM students WHERE id = ?", (student_id,))
    data = cur.fetchone()
    
    if data is None:
        cur.execute("INSERT INTO students (id, name) VALUES (?, ?)", (student_id, name))
        print("[INFO] New student added to database.")
    else:
        cur.execute("UPDATE students SET name = ? WHERE id = ?", (name, student_id))
        print("[INFO] Student name updated in database.")
    
    conn.commit()
    conn.close()

def capture_faces(student_id):
    """Capture face images for the given student ID."""
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cam = cv2.VideoCapture(0)

    print("[INFO] Starting face capture. Look at the camera...")
    count = 0
    TOTAL_IMAGES = 50 

    while True:
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(100, 100)
        )

        for (x, y, w, h) in faces:
            count += 1
            
            face_img = gray[y:y+h, x:x+w]
            
            img_path = os.path.join(DATASET_DIR, f"user.{student_id}.{count}.jpg")
            cv2.imwrite(img_path, face_img)

            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
            cv2.putText(frame, f"Image {count}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Capture Faces - Press q to Quit", frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif count >= TOTAL_IMAGES:
            print(f"[INFO] Captured {TOTAL_IMAGES} images.")
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    init_db()
    
    try:
        student_id = int(input("Enter Student ID (integer): "))
    except ValueError:
        print("Invalid ID. Must be an integer.")
        exit(0)

    name = input("Enter Student Name: ")

    insert_or_update_student(student_id, name)
    capture_faces(student_id)
    print("[INFO] Dataset creation completed.")
