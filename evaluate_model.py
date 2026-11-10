import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
import matplotlib.pyplot as plt


# ============================================================
# Smart Krishi Mitra
# Sugarcane Disease Model Evaluation
# ============================================================

TEST_DIR = r"I:\development\SmartKrishiDataset\test"
MODEL_PATH = r"I:\development\SmartKrishiModel\sugarcane_disease_model.keras"
OUTPUT_DIR = r"I:\development\SmartKrishiModel"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

CLASS_NAMES = [
    "Healthy",
    "Mosaic",
    "RedRot",
    "Rust",
    "Yellow",
]


# ============================================================
# Load Model
# ============================================================

print("=" * 60)
print(" Smart Krishi Mitra - Model Evaluation")
print("=" * 60)

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# Load Test Dataset
# ============================================================

print("\nLoading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("Test dataset loaded successfully.")

print(f"Test images: {len(test_dataset.file_paths)}")


# ============================================================
# Make Predictions
# ============================================================

print("\nRunning predictions...")

y_true = []
y_pred = []

for images, labels in test_dataset:
    predictions = model.predict(
        images,
        verbose=0,
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1,
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# Overall Accuracy
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred,
)

print("\n" + "=" * 60)
print(" OVERALL RESULTS")
print("=" * 60)

print(f"\nAccuracy: {accuracy * 100:.2f}%")


# ============================================================
# Classification Report
# ============================================================

print("\n" + "=" * 60)
print(" CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
)

print("\n" + report)


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
)

print("=" * 60)
print(" CONFUSION MATRIX")
print("=" * 60)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(" " * 12 + " ".join(
    f"{name:>10}"
    for name in CLASS_NAMES
))

for i, row in enumerate(cm):
    print(
        f"{CLASS_NAMES[i]:>12}"
        + " ".join(
            f"{value:>10}"
            for value in row
        )
    )


# ============================================================
# Per-Class Accuracy
# ============================================================

print("\n" + "=" * 60)
print(" PER-CLASS ACCURACY")
print("=" * 60)

for i, class_name in enumerate(CLASS_NAMES):

    total = cm[i].sum()

    correct = cm[i, i]

    if total > 0:
        class_accuracy = (
            correct / total
        ) * 100
    else:
        class_accuracy = 0

    print(
        f"{class_name:>10}: "
        f"{class_accuracy:.2f}% "
        f"({correct}/{total})"
    )


# ============================================================
# Save Classification Report
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt",
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "Smart Krishi Mitra\n"
        "Sugarcane Disease Model Evaluation\n"
        "\n"
    )

    file.write(
        f"Overall Accuracy: "
        f"{accuracy * 100:.2f}%\n\n"
    )

    file.write(
        "Classification Report\n"
        "=====================\n\n"
    )

    file.write(report)

    file.write(
        "\n\nConfusion Matrix\n"
        "================\n\n"
    )

    file.write(
        np.array2string(cm)
    )


print(
    f"\nClassification report saved to:\n"
    f"{report_path}"
)


# ============================================================
# Create Confusion Matrix Image
# ============================================================

plt.figure(figsize=(8, 7))

plt.imshow(cm)

plt.title(
    "Smart Krishi Mitra - Disease Classification"
)

plt.xlabel("Predicted Class")

plt.ylabel("Actual Class")

plt.xticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES,
    rotation=45,
    ha="right",
)

plt.yticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES,
)

# Add values inside the matrix
for i in range(len(CLASS_NAMES)):
    for j in range(len(CLASS_NAMES)):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
        )


plt.tight_layout()


cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png",
)

plt.savefig(
    cm_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


print(
    f"Confusion matrix saved to:\n"
    f"{cm_path}"
)


# ============================================================
# Finished
# ============================================================

print("\n" + "=" * 60)
print(" EVALUATION COMPLETE")
print("=" * 60)

print("\nNext, we will use these results to decide")
print("whether the model needs improvement before")
print("converting it to TensorFlow Lite.")