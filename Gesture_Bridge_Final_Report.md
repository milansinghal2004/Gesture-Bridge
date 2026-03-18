# Gesture-Bridge: Product Requirements Document (PRD) & Final Technical Report

## 1. Executive Summary

**Gesture-Bridge** is a comprehensive computer vision and machine learning-based application designed to bridge the communication gap using hand gestures. The project integrates state-of-the-art object detection, hand tracking, and natural language processing to facilitate real-time sign language recognition (for alphabets and words) and interactive virtual drawing (Air Canvas). It serves as a unified platform accessible through a graphical centralized launcher.

## 2. Problem Statement

Communication for speech and hearing-impaired individuals relies heavily on sign language, which is often not understood by the broader public. Furthermore, interactive touch-free interfaces are becoming increasingly relevant in human-computer interaction (HCI). Gesture-Bridge solves these issues by providing:

1. Real-time translation of Sign Language (Alphabets and Words) to English text.
2. An assistive sentence construction mechanism to bridge gestures into complete sentences.
3. A touchless, gesture-controlled virtual drawing environment (Air Canvas).

## 3. System Architecture & Core Modules

The system is divided into three primary functional modules and a centralized Graphical User Interface (GUI) launcher.

### 3.1. GUI Launcher (`launcher.py`)

Serves as the entry point of the application built using `tkinter`. It provides a simple, intuitive dashboard to launch specific modules:

- **Detect** -> Options for recognizing **Words** (HaGRID dataset) or **Alphabets** (YOLOv8 + DistilGPT-2).
- **Draw** -> Launches the **Air Canvas** application.

### 3.2. Alphabet Gesture Recognition & Sentence Construction (`button.py`)

This module recognizes American Sign Language (ASL) alphabets in real-time and forms coherent sentences.

- **Gesture Detection:** Uses a custom-trained **YOLOv8** model (`best.pt`) running on the CPU to detect alphabet gestures with high accuracy.
- **Word Formation:** Accumulates stabilized letter detections to construct words, applying logic like "WORD_BREAK_FRAMES" to distinguish between words.
- **Sentence Construction & Autocompletion:** Integrates **DistilGPT-2** (via Hugging Face `transformers`) to contextually complete and format sentences (e.g., adding verbs like "is/are", capitalizing, and predicting next words).

### 3.3. Word / Static Gesture Classification (`HaGrid` Module)

This module is tailored for recognizing 18 distinct hand gestures based on the HaGRID dataset.

- **Hand Tracking:** Utilizes **MediaPipe Holistic/Hands** to extract 21 3D hand landmarks in real-time.
- **Data Normalization:** Landmarks are recently scaled (wrist as origin, scaled by mid-finger tip) exactly as trained.
- **Classification:** Uses a trained **Support Vector Classifier (SVC)** (`best_model.pkl`) mapped via a Label Encoder to classify the specific gesture.
- **Sentence Tracking:** Utilizes a `deque` memory buffer to take the mode of the last 10 frames, ensuring stable predictions, and appends the detected gestures into a sequence (sentence display).

### 3.4. Air Canvas (`AirCanvas` Module)

A touch-free interactive drawing board that interprets hand gestures as drawing commands.

- **Tracking & Recognition:** Employs **MediaPipe** to track the index finger tip.
- **Gesture States:** Identifies gestures representing different tools (`SELECT`, `DRAW`, `ERASE`).
- **Functionality:** Users can select colors from a top-bar UI, draw on a virtual canvas, or use the eraser. The canvas binary mask is dynamically overlaid onto the webcam feed using OpenCV bitwise operations.

## 4. Technical Stack

### 4.1 Programming Language & Libraries

- **Python 3.x**
- **OpenCV (`cv2`)**: For video capture, image processing, bitwise masking, and UI rendering on frames.
- **MediaPipe**: For highly accurate spatial hand landmark tracking (used in both HaGRID and Air Canvas).
- **Ultralytics (YOLOv8)**: For bounding-box object detection of alphabet gestures.
- **Hugging Face Transformers**: For the `distilgpt2` language model.
- **Scikit-Learn**: For training and utilizing the SVC model.
- **Tkinter**: For the desktop GUI launcher.

### 4.2 Machine Learning Models

- `best.pt`: YOLOv8 weights trained specifically for alphabet sign language.
- `best_model.pkl`: Scikit-Learn SVC trained on the HaGRID dataset landmarks.
- `distilgpt2`: Lightweight causal language model for real-time text generation and grammar improvement.

## 5. Technical Pipeline & Working Workflow

The system employs a multi-staged pipeline that spans from image acquisition to ML inference and feedback rendering.

### 5.1 System Initiation Pipeline

1. **Execution**: The user executes `launcher.py`.
2. **GUI Rendering**: Tkinter initializes the main menu (`root.mainloop()`).
3. **Subprocess Dispatch**: Clicking a button invokes `subprocess.run()`, which spawns independent Python environments to run either `button.py` (Alphabets), `Implementation.py` (Words), or `main1.py` (Air Canvas).

### 5.2 Alphabet & Sentence Construction Pipeline (`button.py`)

1. **Frame Capture & Preprocessing**:
   - `cv2.VideoCapture(0)` pulls the webcam feed at 640x480 resolution.
2. **YOLOv8 Object Detection**:
   - The frame is passed directly to the `YOLO('best.pt')` model loaded into the CPU.
   - The model returns bounding boxes (`xyxy`), detection classes (class ID mapping to alphabet letters), and confidence scores.
   - Filtering: Only detections with confidence `> LETTER_CONF_THRESHOLD` (0.5) are processed.
3. **Temporal Stabilization (Word Formation)**:
   - To prevent flickering, a letter is only registered if it remains constant for `STABLE_LETTER_FRAMES` (5 frames).
   - These stable letters are accumulated into a `current_word_letters` array.
4. **Word Boundary Detection**:
   - If no gesture is detected for `WORD_BREAK_FRAMES` (50 frames), the array is joined into a string, converted to lowercase (except for 'I' and the first word), and pushed to `recognized_words`.
5. **NLP Sentence Autocompletion Pipeline**:
   - The raw sentence (e.g., "Hi name John") is tokenized and passed into **DistilGPT-2**.
   - The model generates exactly one new token/word (`max_new_tokens=1`).
   - Post-processing logic explicitly checks the model output to intelligently inject verbs like "is", "are", or "to" (e.g., converting "Hi name John" -> "Hi name is John").
6. **Output Rendering**:
   - Bounding boxes and labels are drawn using OpenCV. The final stabilized, NLP-corrected sentence is overlaid onto the video feed before being presented through `cv2.imshow()`.

### 5.3 Static Gesture Recognition Pipeline (`Implementation.py`)

1. **Frame Capture**: Webcam feed is captured and transformed from BGR to RGB (`cv2.cvtColor`).
2. **MediaPipe Landmark Extraction**:
   - The frame passes through `mp_hands.Hands()`.
   - If a hand is detected, an array of 21 3D coordinates (x, y, z) is returned.
3. **Coordinate Normalization Pipeline**:
   - **Translation**: The wrist (Landmark 0) coordinates are subtracted from all other 20 landmarks, making the wrist the absolute origin (0,0).
   - **Scaling**: All (x, y) coordinates are divided by the absolute distance to the middle finger tip (Landmark 12), ensuring scale invariance (distance from camera). The z-value is preserved.
4. **Machine Learning Inference (SVC classification)**:
   - The flattened 63-value array (21 landmarks x 3 axes) is fed to `model.predict()` (Scikit-Learn SVC).
   - The numerical output is decoded to a string using `label_encoder.pkl`.
5. **Moving Mode Stabilization Buffer**:
   - The string prediction is appended to a fixed-size `deque` of size 10.
   - The statistical `mode` of the deque determines the ultimate "stable gesture" to reduce jitter.
6. **Sentence Tracking**:
   - Once stabilized and exceeding a `COOLDOWN_FRAMES` gap (30 frames), the word is appended to the `current_sentence` array.
7. **Rendering**: OpenCV draws the dynamic MediaPipe landmarks, the current single gesture box, and the ongoing sentence at the top of the GUI.

### 5.4 Air Canvas Rendering Pipeline (`main1.py`)

1. **Frame Capture & Hand Isolation**: Webcam imports the frame. `HandTracker` isolates the hand using MediaPipe.
2. **Keypoint Extraction**: Specifically extracts the tip of the index finger (Landmark 8).
3. **Gesture Mode Evaluation Pipeline**:
   - `GestureRecogniser` calculates the Euclidean distance between specific fingers (e.g., thumb vs index vs middle).
   - Based on thresholds, it returns a categorical `GestureType`: `SELECT` (two fingers raised), `DRAW` (one finger raised), or `ERASE` (three fingers raised).
4. **State Machine & Canvas Update**:
   - `DrawingCanvas` maintains a state matrix mimicking the frame size.
   - If `SELECT`: Checks coordinates against the UI Color Bar to swap `current_color`.
   - If `DRAW`/`ERASE`: Draws lines (or thick erased paths) on the background state matrix between the last known finger coordinate and the current one.
5. **Alpha Masking Integration**:
   - The canvas matrix is converted into a binary mask (`cv2.cvtColor` -> `cv2.threshold`).
   - The mask splits the live webcam feed into foreground (drawn pixels) and background (webcam feed).
   - `cv2.add(frame_bg, drawing_display)` perfectly overlays the continuous drawing on top of the live feed without lagging.

## 6. Evaluation & Metrics

- The **YOLOv8** model includes an `evaluate_yolo.py` script calculating standard object detection metrics (Precision, Recall, F1 Score, mAP@0.5, mAP@0.5:0.95).
- The **HaGRID SVC** model has been aggressively tuned, achieving:
  - **Accuracy:** ~98.1%
  - **Precision/Recall/F1-score:** ~0.981

## 7. Conclusion & Future Enhancements

**Gesture-Bridge** successfully creates a multi-modal interface leveraging both deterministic rules (AirCanvas distance mapping), ML classification (SVC on MediaPipe coordinates), and Deep Learning (YOLOv8 and Transformers).

**Potential Future Enhancements:**

- Integration of dynamic/moving signs (e.g., using LSTMs or 3D CNNs) rather than strictly static alphabets and words.
- GPU acceleration optimization for YOLOv8 deployment (currently explicitly set to CPU for broad compatibility).
- A unified UI incorporating all sub-modules under one continuous video feed application rather than separate pop-up scripts.

