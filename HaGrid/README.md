
# 🤖 Hand Gesture Classification Using MediaPipe & HaGRID Dataset

This project focuses on classifying 18 different hand gestures using landmark coordinates extracted via MediaPipe from the HaGRID dataset. A trained machine learning model then classifies the gestures in real-time using a webcam.

## 📁 Project Structure

```
Hand-Gesture-Classification/
│
├── Dataset/
│   └── haGRID_landmarks.csv          # Preprocessed landmark data from MediaPipe
│
├── best_model.pkl                    # Final trained model
├── label_encoder.pkl                 # Label encoder to map class numbers to labels
│
├── Data_Preparation.ipynb            # Data cleaning, preprocessing, and normalization
├── Training_and_Evaluation.ipynb     # Training and tuning of multiple ML models
├── Implementation_using_Mediapipe.ipynb   # Real-time implementation using MediaPipe & webcam
│
├── README.md                         # Project documentation (you're here!)
```

## 🧠 Project Highlights

- **Dataset**: HaGRID Dataset with 21 hand landmarks per gesture image (https://www.kaggle.com/datasets/kapitanov/hagrid)
- **Preprocessing**: Re-centering to wrist and normalization with middle finger tip
- **Models Tested**: Logistic Regression, Decision Tree, Random Forest, SVC, KNN, XGBoost
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score
- **Best Model**: Support Vector Classifier (SVC) after tuning

## 🎯 Results Summary

### ✅ Before Tuning:

| Model                  | Accuracy | Precision | Recall | F1-score |
|------------------------|----------|-----------|--------|----------|
| LogisticRegression     | 0.799    | 0.798     | 0.799  | 0.797    |
| DecisionTreeClassifier | 0.854    | 0.856     | 0.854  | 0.854    |
| RandomForestClassifier | 0.939    | 0.940     | 0.939  | 0.939    |
| SVC                    | 0.836    | 0.857     | 0.836  | 0.839    |
| KNeighborsClassifier   | 0.908    | 0.910     | 0.908  | 0.908    |
| XGBClassifier          | 0.970    | 0.970     | 0.970  | 0.970    |

### ✅ After Tuning:

| Model                  | Accuracy | Precision | Recall | F1-score |
|------------------------|----------|-----------|--------|----------|
| LogisticRegression     | 0.931    | 0.931     | 0.931  | 0.930    |
| DecisionTreeClassifier | 0.868    | 0.869     | 0.868  | 0.868    |
| RandomForestClassifier | 0.953    | 0.953     | 0.953  | 0.953    |
| SVC                    | 0.981    | 0.981     | 0.981  | 0.981    |
| KNeighborsClassifier   | 0.932    | 0.933     | 0.932  | 0.933    |
| XGBClassifier          | 0.978    | 0.978     | 0.978  | 0.978    |

## 🔧 How to Use (Live Webcam Mode)

1. Make sure you have a webcam enabled and your environment supports OpenCV and MediaPipe (e.g., local machine or Colab with webcam access like Colab Pro or Jupyter).
2. Clone the repo or upload it to your Colab/local environment.
3. Run the `Implementation_using_Mediapipe.ipynb`.
4. Inside the notebook:
   - It uses MediaPipe to detect hand landmarks in real-time from webcam frames
   - Landmarks are preprocessed and passed to the trained model
   - The prediction label is displayed on the frame in real-time
   - Press `q` to exit the webcam window

## 🧰 Requirements

- Python 3.x
- OpenCV
- MediaPipe
- scikit-learn
- joblib
- xgboost (optional)
- matplotlib (for visualization)
- numpy, pandas

## 💡 Key Techniques Used

- **Hand Landmark Detection**: MediaPipe's holistic hand solution
- **Normalization**: Recenter on wrist, scale using mid-finger
- **ML Algorithms**: SVM, Random Forest, XGBoost, etc.
- **Model Tuning**: GridSearchCV for hyperparameter optimization
- **Stabilization**: Optional window mode smoothing of predictions

## 📹 Demo (Optional)

https://drive.google.com/file/d/1q6TI4d9Br_06sElh-FZQEbUmZkOWUFOo/view?usp=drive_link

