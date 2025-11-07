import cv2
import numpy as np
from tensorflow.keras.models import load_model

# -------- Configuration --------
MODEL_PATH = "best_ddd_model_realistic.h5"   # trained model file
IMG_SIZE = (128, 128)                               # image size for model
CLASSES = ["awake", "drowsy", "sleep", "yawn"]      # output labels
# --------------------------------

# Load model
print("Loading trained model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Load face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# This loads a Haar Cascade Classifier, a pre-trained face detection model available in OpenCV.
# It helps the system detect faces in each frame captured from the webcam.
# The file "haarcascade_frontalface_default.xml" contains patterns to identify human faces.

def preprocess(img):
    """Resize and normalize the face image before giving it to the model."""
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # If img was shape (96, 96, 3), it becomes (1, 96, 96, 3).
    return img

# ----------- MAIN PROGRAM -----------
cap = cv2.VideoCapture(0)  # open webcam

if not cap.isOpened():
    print("Error: Cannot access camera.")
    exit()

print("Starting detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 5)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Crop the face region
            face = frame[y:y+h, x:x+w]

            # Preprocess the image for prediction
            processed = preprocess(face)
            prediction = model.predict(processed, verbose=0)

            # Get predicted label
            label_index = np.argmax(prediction)
            label = CLASSES[label_index]

            # Display driver state
            if label == "awake":
                text = f"Driver State: NORMAL"
                color = (0, 255, 0)
            else:
                text = f"Driver State:ABNORMAL"
                color = (0, 0, 255)

            cv2.putText(frame, text, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    else:
        cv2.putText(frame, "Driver State:ABNORMAL", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Show the output frame
    cv2.imshow("Driver Drowsiness Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Detection stopped.")
