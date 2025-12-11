import cv2
import os
import numpy as np

DATASET_DIR = "dataset"
TRAINER_DIR = "trainer"

def get_images_and_labels(dataset_path):
    """Read images from dataset and return face samples and IDs."""
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
    
    face_samples = []
    ids = []

    for image_path in image_paths:
        
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        
        try:
            filename = os.path.split(image_path)[-1]
            parts = filename.split(".")
            
            student_id = int(parts[1])
        except:
            print(f"[WARN] Skipping file: {image_path}")
            continue

        face_samples.append(np.array(img, dtype="uint8"))
        ids.append(student_id)

    return face_samples, ids

if __name__ == "__main__":
    if not os.path.exists(DATASET_DIR):
        print("[ERROR] Dataset directory not found. Run create_dataset.py first.")
        exit(0)

    if not os.path.exists(TRAINER_DIR):
        os.makedirs(TRAINER_DIR)

    print("[INFO] Training faces. This may take a few seconds...")

    faces, ids = get_images_and_labels(DATASET_DIR)

    if len(faces) == 0:
        print("[ERROR] No faces found in dataset.")
        exit(0)

    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    
    
    trainer_path = os.path.join(TRAINER_DIR, "trainer.yml")
    recognizer.save(trainer_path)

    print(f"[INFO] Training completed. Model saved at {trainer_path}")
