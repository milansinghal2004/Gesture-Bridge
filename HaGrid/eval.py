import cv2
import numpy as np
import pickle
from collections import deque
from statistics import mode
import mediapipe as mp
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model and label encoder
with open('Hagrid/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('Hagrid/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Store predictions and actual labels for evaluation
predictions = []
actual_labels = []

# Define a deque for smoothing predictions
predictions_window = deque(maxlen=10)

# Function to normalize landmarks
def normalize_landmarks(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    mid_tip = hand_landmarks.landmark[12]

    x_scale = mid_tip.x if abs(mid_tip.x) > 1e-6 else 1e-6
    y_scale = mid_tip.y if abs(mid_tip.y) > 1e-6 else 1e-6

    normalized = []
    for lm in hand_landmarks.landmark:
        x_norm = (lm.x - wrist.x) / x_scale
        y_norm = (lm.y - wrist.y) / y_scale
        z_norm = lm.z  # Keep z as is
        normalized.extend([x_norm, y_norm, z_norm])

    return normalized

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Normalize landmarks and predict gesture
            processed = normalize_landmarks(hand_landmarks)

            try:
                pred = model.predict([processed])[0]
                gesture = le.inverse_transform([pred])[0]
                predictions_window.append(gesture)
                current_gesture = mode(predictions_window) if predictions_window else gesture
            except:
                current_gesture = "Unknown"

            # Collect predictions and ground truth for evaluation
            predictions.append(current_gesture)
            # Actual label here should be defined, assuming you have a method to map ground truth to frames
            # Example: actual_labels.append(actual_label)

            # Draw landmarks and gesture
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display prediction on screen
            cv2.putText(frame, current_gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# After capturing frames, evaluate the model's performance
# Example: Assuming you have the actual labels (you can collect them manually or from a dataset)
# actual_labels = ['gesture1', 'gesture2', ...]

# Evaluation Metrics
if len(predictions) == len(actual_labels):
    # Calculate accuracy
    accuracy = accuracy_score(actual_labels, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(actual_labels, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

else:
    print("Error: The number of predictions does not match the number of actual labels.")
