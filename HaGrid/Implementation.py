import cv2
import numpy as np
from collections import deque
from statistics import mode
import mediapipe as mp
import pickle


# # Normalize landmarks exactly as done in training:

# In[6]:


def normalize_landmarks(hand_landmarks):
    """
    Normalize landmarks exactly as done in training:
    1. Make wrist (landmark 0) the origin
    2. Scale x by mid finger tip x (landmark 12 x)
    3. Scale y by mid finger tip y (landmark 12 y)
    4. Leave z coordinates unchanged
    """
    # Get wrist and middle finger tip landmarks
    wrist = hand_landmarks.landmark[0]
    mid_tip = hand_landmarks.landmark[12]

    # Calculate scale factors with small epsilon to avoid division by zero
    x_scale = mid_tip.x if abs(mid_tip.x) > 1e-6 else 1e-6
    y_scale = mid_tip.y if abs(mid_tip.y) > 1e-6 else 1e-6

    normalized = []
    for lm in hand_landmarks.landmark:
        # Recenter to wrist and scale by mid finger tip
        x_norm = (lm.x - wrist.x) / x_scale
        y_norm = (lm.y - wrist.y) / y_scale
        z_norm = lm.z  # Keep z as is

        normalized.extend([x_norm, y_norm, z_norm])

    return normalized


# # Initialize MediaPipe

# In[7]:


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# # Load model and label encoder

# In[8]:


with open('C:/Users/singh/Documents/.vscode/Gesture Bridge/Hagrid/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('C:/Users/singh/Documents/.vscode/Gesture Bridge/Hagrid/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)


# # Initialize video capture

# In[ ]:


# Color settings (BGR format)
DARK = (223, 191, 159)
LIGHT = (115, 77, 38)
TEXT_COLOR = (255, 255, 255)  # White color for sentence display

# Initialize video and sentence tracking
cap = cv2.VideoCapture(0)
predictions_window = deque(maxlen=10)
current_sentence = []  # Store words in the sentence
last_gesture = None
gesture_cooldown = 0  # Cooldown counter to prevent rapid word addition
COOLDOWN_FRAMES = 30  # Adjust this value to control word addition speed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    # Create a clean space for displaying the sentence
    sentence_display = " ".join(current_sentence)
    cv2.rectangle(frame, (10, 10), (frame.shape[1]-10, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Sentence: {sentence_display}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2, cv2.LINE_AA)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Normalize exactly like training data
            processed = normalize_landmarks(hand_landmarks)

            # Predict gesture
            try:
                pred = model.predict([processed])[0]
                gesture = le.inverse_transform([pred])[0]
                predictions_window.append(gesture)
                current_gesture = mode(predictions_window) if predictions_window else gesture
                
                # Add word to sentence if gesture is stable and cooldown is complete
                if gesture_cooldown == 0 and current_gesture != last_gesture:
                    current_sentence.append(current_gesture)
                    gesture_cooldown = COOLDOWN_FRAMES
                    last_gesture = current_gesture
                
            except:
                current_gesture = "Unknown"

            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=LIGHT, thickness=3, circle_radius=3),
                mp_drawing.DrawingSpec(color=DARK, thickness=3, circle_radius=1.5)
            )

            # Display current gesture
            h, w = frame.shape[:2]
            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
            text_x, text_y = min(x_coords), max(10, min(y_coords) - 30)

            cv2.rectangle(frame, (text_x-5, text_y-25), 
                         (text_x + len(current_gesture)*20, text_y+10), (0,0,0), -1)
            cv2.putText(frame, current_gesture, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, LIGHT, 2, cv2.LINE_AA)

    # Decrease cooldown counter
    if gesture_cooldown > 0:
        gesture_cooldown -= 1

    # Display controls info
    controls_text = "Controls: Q-Quit, C-Clear, Backspace-Remove last word"
    cv2.putText(frame, controls_text, (20, frame.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)

    cv2.imshow('Hand Gesture Recognition', frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('c'):  # Clear sentence
        current_sentence = []
        last_gesture = None
    elif key == 8:  # Backspace to remove last word
        if current_sentence:
            current_sentence.pop()
            last_gesture = None if not current_sentence else current_sentence[-1]

cap.release()
cv2.destroyAllWindows()