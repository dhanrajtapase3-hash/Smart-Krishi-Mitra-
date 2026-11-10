import os
import tensorflow as tf


# ============================================================
# SMART KRISHI MITRA
# Convert Fine-Tuned Sugarcane Disease Model to TFLite
# ============================================================

MODEL_PATH = (
    r"I:\development\SmartKrishiModel"
    r"\fine_tuned_best_model.keras"
)

OUTPUT_DIR = (
    r"I:\development\SmartKrishiModel"
)

TFLITE_MODEL_PATH = (
    os.path.join(
        OUTPUT_DIR,
        "sugarcane_disease_model.tflite"
    )
)

CLASS_NAMES_PATH = (
    os.path.join(
        OUTPUT_DIR,
        "class_names.txt"
    )
)


# ------------------------------------------------------------
# Start
# ------------------------------------------------------------

print("=" * 70)
print(" Smart Krishi Mitra - TFLite Conversion")
print("=" * 70)

print()
print("TensorFlow version:", tf.__version__)

print()
print("Loading model:")
print(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Fine-tuned model not found:\n"
        f"{MODEL_PATH}"
    )


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# Load Keras model
# ------------------------------------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH
)

print()
print("Model loaded successfully.")

print()
print("Input shape:")
print(model.input_shape)

print()
print("Output shape:")
print(model.output_shape)


# ------------------------------------------------------------
# Convert to TensorFlow Lite
# ------------------------------------------------------------

print()
print("=" * 70)
print(" Converting to TensorFlow Lite")
print("=" * 70)
print()

converter = tf.lite.TFLiteConverter.from_keras_model(
    model
)

# Standard TFLite conversion.
# We keep float32 here for maximum compatibility
# and to avoid changing model behavior before Flutter integration.

tflite_model = converter.convert()


# ------------------------------------------------------------
# Save TFLite model
# ------------------------------------------------------------

with open(
    TFLITE_MODEL_PATH,
    "wb"
) as file:

    file.write(
        tflite_model
    )


# ------------------------------------------------------------
# Save class names
# ------------------------------------------------------------

class_names = [
    "Healthy",
    "Mosaic",
    "RedRot",
    "Rust",
    "Yellow",
]


with open(
    CLASS_NAMES_PATH,
    "w",
    encoding="utf-8"
) as file:

    for name in class_names:
        file.write(
            name + "\n"
        )


# ------------------------------------------------------------
# Information
# ------------------------------------------------------------

model_size_mb = (
    os.path.getsize(
        TFLITE_MODEL_PATH
    )
    / (1024 * 1024)
)


print()
print("=" * 70)
print(" Conversion Successful")
print("=" * 70)

print()
print("TFLite model:")
print(TFLITE_MODEL_PATH)

print()
print(
    f"TFLite model size: "
    f"{model_size_mb:.2f} MB"
)

print()
print("Class names:")
for index, name in enumerate(class_names):
    print(
        f"  {index}: {name}"
    )

print()
print("=" * 70)
print(" Ready for Flutter Integration")
print("=" * 70)
