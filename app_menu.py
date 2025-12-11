import subprocess
import sys
import sqlite3
import os
import glob

DB_NAME = "attendance.db"

# ------------ DELETE STUDENT FUNCTION ------------ #

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
    print("[INFO] Model will be retrained now so this student is no longer recognized.")


# ------------ EXISTING FUNCTIONS ------------ #

def run_create_and_train():
    """Register a new student and retrain the model."""
    print("\n[STEP 1] Registering new student and capturing faces...\n")
    subprocess.run([sys.executable, "create_dataset.py"])

    print("\n[STEP 2] Training face recognition model...\n")
    subprocess.run([sys.executable, "train_model.py"])

    print("\n[INFO] Student added and model trained successfully!\n")


def run_attendance():
    """Start attendance system."""
    print("\n[INFO] Starting attendance system...\n")
    subprocess.run([sys.executable, "attendance.py"])
    print("\n[INFO] Attendance session finished.\n")


# ------------ MAIN MENU ------------ #

def main_menu():
    while True:
        print("!!!!!!*********************!!!!!!!!!!!")
        print("  SMART ATTENDANCE SYSTEM - MAIN MENU ")
        print("!!!!!!*********************!!!!!!!!!!!")
        print("1. Register NEW student")
        print("2. Start attendance")
        print("3. Delete student")
        print("4. Exit")
        print("!!!!!!*********************!!!!!!!!!!!")

        choice = input("Enter your choice (1/2/3/4): ").strip()

        if choice == "1":
            run_create_and_train()

        elif choice == "2":
            run_attendance()

        elif choice == "3":
            try:
                sid = int(input("Enter the Student ID to delete: ").strip())
                delete_student(sid)
                # Retrain model after deleting student
                subprocess.run([sys.executable, "train_model.py"])
                print("[INFO] Model retrained successfully after deletion.\n")
            except ValueError:
                print("[ERROR] Invalid Student ID. Please enter a number.\n")

        elif choice == "4":
            print("Exiting... Have a good day!")
            break

        else:
            print("\n[ERROR] Invalid input. Please enter 1, 2, 3, or 4.\n")


if __name__ == "__main__":
    main_menu()
