import os
import cv2
import numpy as np
import time
from datetime import datetime
from insightface.app import FaceAnalysis


# --- Configuration ---
dataset_path = "C:/Users/Allen/Desktop/FaceRecognition/dataset"  # dataset folder
similarity_threshold = 0.5
cooldown_time = 10

# --- Initialize InsightFace ---
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# --- Step 1: Load known faces ---
known_embeddings = []
known_names = []

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    for img_file in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        faces = app.get(img)
        if len(faces) == 0:
            continue

        known_embeddings.append(faces[0].embedding)
        known_names.append(person_name)

print(f"Loaded {len(known_embeddings)} known faces.")

# --- Step 2: Define similarity functions ---

def cosine_similarity(a, b):
    """Compute cosine similarity between two embeddings."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def activated_similarity(a, b, k=8.0):
    """
    Apply sigmoid activation to enhance similarity discrimination.
    Compresses low similarities and emphasizes confident matches.
    """
    cos_sim = cosine_similarity(a, b)
    # Shift and scale sigmoid curve to emphasize mid-high similarity
    activated = 1 / (1 + np.exp(-k * (cos_sim - 0.5)))
    return activated


# --- Step 3: Start webcam ---
cap = cv2.VideoCapture(0)

# Tracking
last_detected_name = None
last_detection_time = 0
door_state = {}  # Keeps track of each person's door state

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        # Compute similarity scores with activation
        sim_scores = [activated_similarity(face.embedding, known_emb) for known_emb in known_embeddings]
        if len(sim_scores) == 0:
            continue

        best_idx = np.argmax(sim_scores)
        best_score = sim_scores[best_idx]

        if best_score > similarity_threshold:
            name = known_names[best_idx]

            current_time = time.time()
            dt = datetime.fromtimestamp(current_time)
            formatted_time = dt.strftime("%a, %b. %d, %Y %I:%M%p ").replace("AM", "am").replace("PM", "pm")

            # Only print if it's a new person or cooldown passed
            if name != last_detected_name or (current_time - last_detection_time) > cooldown_time:
                if door_state.get(name) == "unlocked":
                    # Person scanned again — class ended
                    print("\n" + "=" * 50)
                    print(f" DOOR LOCKED (at {formatted_time})\n Person identified: {name}")
                    print(f" Similarity score: {best_score:.3f}")
                    print("=" * 50 + "\n")
                    door_state[name] = "locked"
                else:
                    # First scan — unlock
                    print("\n" + "=" * 50)
                    print(f" DOOR UNLOCKED (at {formatted_time})\n Person identified: {name}")
                    print(f" Similarity score: {best_score:.3f}")
                    print("=" * 50 + "\n")
                    door_state[name] = "unlocked"

                last_detected_name = name
                last_detection_time = current_time
        else:
            name = "Unknown"

        # Draw bounding box and name
        x1, y1, x2, y2 = [int(i) for i in face.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({best_score:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Facial Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
