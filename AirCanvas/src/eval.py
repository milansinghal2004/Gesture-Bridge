import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
# from jiwer import wer

# -----------------------------
# Sample Data - Replace with logs
# -----------------------------
# Ground truth gestures (from test session or dataset)
true_gestures = [
    "DRAW", "DRAW", "SELECT", "ERASE", "DRAW", "SELECT", "ERASE", "DRAW", "DRAW", "SELECT"
]

# Predicted gestures from the application
predicted_gestures = [
    "DRAW", "DRAW", "SELECT", "DRAW", "DRAW", "SELECT", "ERASE", "DRAW", "ERASE", "SELECT"
]

# Voice command evaluation
true_commands = [
    "clear", "red", "blue", "green", "yellow", "exit"
]

predicted_commands = [
    "clear", "read", "blue", "grain", "yellow", "exist"
]

# -----------------------------
# Gesture Recognition Evaluation
# -----------------------------
print("\n🎯 Gesture Recognition Evaluation:")
print(classification_report(true_gestures, predicted_gestures))

cm = confusion_matrix(true_gestures, predicted_gestures, labels=["DRAW", "SELECT", "ERASE"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["DRAW", "SELECT", "ERASE"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Gesture Confusion Matrix")
plt.show()

# -----------------------------
# Voice Command Evaluation
# -----------------------------
# print("\n🎤 Voice Command Evaluation:")
# for gt, pred in zip(true_commands, predicted_commands):
#     print(f"GT: {gt:7s} | Predicted: {pred:7s} | WER: {wer(gt, pred):.2f}")

# overall_wer = wer(" ".join(true_commands), " ".join(predicted_commands))
# print(f"\n🧠 Overall WER: {overall_wer:.2f}")

# -----------------------------
# Summary
# -----------------------------
gesture_accuracy = np.mean(np.array(true_gestures) == np.array(predicted_gestures))
print(f"\n✅ Gesture Accuracy: {gesture_accuracy * 100:.2f}%")
