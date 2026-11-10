import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# ============================================================
# SMART KRISHI MITRA
# Sugarcane Disease Detection
# Controlled MobileNetV2 Fine-Tuning
# ============================================================

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

TRAIN_DIR = r"I:\development\SmartKrishiDataset\train"
VAL_DIR = r"I:\development\SmartKrishiDataset\validation"
TEST_DIR = r"I:\development\SmartKrishiDataset\test"

BASELINE_MODEL_PATH = (
    r"I:\development\SmartKrishiModel\best_model.keras"
)

OUTPUT_DIR = r"I:\development\SmartKrishiModel"

FINE_TUNED_BEST_MODEL_PATH = (
    os.path.join(
        OUTPUT_DIR,
        "fine_tuned_best_model.keras"
    )
)

FINE_TUNED_FINAL_MODEL_PATH = (
    os.path.join(
        OUTPUT_DIR,
        "fine_tuned_sugarcane_disease_model.keras"
    )
)

CLASS_NAMES_PATH = (
    os.path.join(
        OUTPUT_DIR,
        "fine_tuned_class_names.txt"
    )
)

HISTORY_PATH = (
    os.path.join(
        OUTPUT_DIR,
        "fine_tuning_history.txt"
    )
)

ACCURACY_GRAPH_PATH = (
    os.path.join(
        OUTPUT_DIR,
        "fine_tuning_validation_accuracy.png"
    )
)

LOSS_GRAPH_PATH = (
    os.path.join(
        OUTPUT_DIR,
        "fine_tuning_validation_loss.png"
    )
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

EPOCHS = 10

# Only the upper portion of MobileNetV2 will be fine-tuned.
FINE_TUNE_LAST_LAYERS = 30

LEARNING_RATE = 0.00001

SEED = 42


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ------------------------------------------------------------
# Create output directory
# ------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 70)
print(" Smart Krishi Mitra - MobileNetV2 Fine-Tuning")
print("=" * 70)

print()
print("TensorFlow version:", tf.__version__)

print()
print("Loading baseline model:")
print(BASELINE_MODEL_PATH)


# ------------------------------------------------------------
# Load baseline model
# ------------------------------------------------------------

if not os.path.exists(BASELINE_MODEL_PATH):
    raise FileNotFoundError(
        "Baseline model not found:\n"
        f"{BASELINE_MODEL_PATH}\n\n"
        "Make sure the previous training completed successfully."
    )

model = keras.models.load_model(
    BASELINE_MODEL_PATH
)

print()
print("Baseline model loaded successfully.")


# ------------------------------------------------------------
# Find MobileNetV2 base model
# ------------------------------------------------------------

base_model = None

for layer in model.layers:
    if isinstance(layer, keras.Model):
        if "mobilenetv2" in layer.name.lower():
            base_model = layer
            break


if base_model is None:
    raise RuntimeError(
        "Could not find the MobileNetV2 base model "
        "inside the saved model."
    )


print()
print("MobileNetV2 base model found:")
print("Name:", base_model.name)
print("Total layers:", len(base_model.layers))


# ------------------------------------------------------------
# Freeze the complete base model first
# ------------------------------------------------------------

base_model.trainable = False

for layer in base_model.layers:
    layer.trainable = False


# ------------------------------------------------------------
# Unfreeze only the final layers
# ------------------------------------------------------------

total_layers = len(base_model.layers)

start_index = max(
    0,
    total_layers - FINE_TUNE_LAST_LAYERS
)

trainable_count = 0

for index, layer in enumerate(base_model.layers):

    if index >= start_index:

        # Keep BatchNormalization frozen.
        # This gives more stable fine-tuning on a small dataset.
        if isinstance(
            layer,
            layers.BatchNormalization
        ):
            layer.trainable = False

        else:
            layer.trainable = True
            trainable_count += 1

    else:
        layer.trainable = False


print()
print("=" * 70)
print(" Fine-Tuning Configuration")
print("=" * 70)

print(
    "Total MobileNetV2 layers :",
    total_layers
)

print(
    "Layers considered for tuning :",
    FINE_TUNE_LAST_LAYERS
)

print(
    "Trainable MobileNetV2 layers :",
    trainable_count
)

print(
    "Learning rate :",
    LEARNING_RATE
)

print(
    "Epochs :",
    EPOCHS
)

print()


# ------------------------------------------------------------
# Print trainable layers
# ------------------------------------------------------------

print("Trainable layers:")

for layer in base_model.layers:

    if layer.trainable:
        print(
            "  ✓",
            layer.name,
            "(",
            layer.__class__.__name__,
            ")"
        )


# ------------------------------------------------------------
# Recompile model
# ------------------------------------------------------------

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)


# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------

print()
print("Loading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED,
)

print()
print("Loading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

print()
print("Loading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)


# ------------------------------------------------------------
# Class names
# ------------------------------------------------------------

class_names = train_dataset.class_names

print()
print("Classes:")
for index, name in enumerate(class_names):
    print(
        f"  {index}: {name}"
    )


# ------------------------------------------------------------
# Performance pipeline
# ------------------------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    AUTOTUNE
)


# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------

checkpoint = keras.callbacks.ModelCheckpoint(
    FINE_TUNED_BEST_MODEL_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=3,
    restore_best_weights=True,
    verbose=1,
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=1,
    min_lr=1e-7,
    verbose=1,
)


# ------------------------------------------------------------
# Fine-tuning
# ------------------------------------------------------------

print()
print("=" * 70)
print(" Starting Fine-Tuning")
print("=" * 70)
print()

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr,
    ],
)


# ------------------------------------------------------------
# Evaluate on test set
# ------------------------------------------------------------

print()
print("=" * 70)
print(" Final Fine-Tuned Test Evaluation")
print("=" * 70)
print()

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1,
)

print()
print(
    f"Fine-Tuned Test Loss     : {test_loss:.4f}"
)

print(
    f"Fine-Tuned Test Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)


# ------------------------------------------------------------
# Save final model
# ------------------------------------------------------------

model.save(
    FINE_TUNED_FINAL_MODEL_PATH
)

print()
print(
    "Fine-tuned final model saved to:"
)

print(
    FINE_TUNED_FINAL_MODEL_PATH
)


# ------------------------------------------------------------
# Save class names
# ------------------------------------------------------------

with open(
    CLASS_NAMES_PATH,
    "w",
    encoding="utf-8",
) as file:

    for name in class_names:
        file.write(
            name + "\n"
        )


print()
print(
    "Class names saved to:"
)

print(
    CLASS_NAMES_PATH
)


# ------------------------------------------------------------
# Save training history
# ------------------------------------------------------------

with open(
    HISTORY_PATH,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "Smart Krishi Mitra - Fine-Tuning History\n"
    )

    file.write(
        "=" * 55 + "\n\n"
    )

    file.write(
        f"Test Loss: {test_loss:.4f}\n"
    )

    file.write(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%\n\n"
    )

    file.write(
        "Epoch-by-epoch results:\n\n"
    )

    for index in range(
        len(history.history["loss"])
    ):

        epoch_number = index + 1

        train_loss = (
            history.history["loss"][index]
        )

        train_accuracy = (
            history.history["accuracy"][index]
        )

        val_loss = (
            history.history["val_loss"][index]
        )

        val_accuracy = (
            history.history["val_accuracy"][index]
        )

        file.write(
            f"Epoch {epoch_number}: "
            f"Train Loss={train_loss:.4f}, "
            f"Train Accuracy={train_accuracy * 100:.2f}%, "
            f"Val Loss={val_loss:.4f}, "
            f"Val Accuracy={val_accuracy * 100:.2f}%\n"
        )


# ------------------------------------------------------------
# Plot validation accuracy
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["val_accuracy"],
    marker="o",
)

plt.title(
    "Smart Krishi Mitra - Fine-Tuning Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Validation Accuracy"
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    ACCURACY_GRAPH_PATH,
    dpi=150,
)

plt.close()


# ------------------------------------------------------------
# Plot validation loss
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["val_loss"],
    marker="o",
)

plt.title(
    "Smart Krishi Mitra - Fine-Tuning Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Validation Loss"
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    LOSS_GRAPH_PATH,
    dpi=150,
)

plt.close()


# ------------------------------------------------------------
# Completion
# ------------------------------------------------------------

print()
print("=" * 70)
print(" Fine-Tuning Complete")
print("=" * 70)

print()
print(
    "Baseline model was NOT overwritten."
)

print()
print(
    "Baseline model:"
)

print(
    BASELINE_MODEL_PATH
)

print()
print(
    "Fine-tuned best model:"
)

print(
    FINE_TUNED_BEST_MODEL_PATH
)

print()
print(
    "Fine-tuned final model:"
)

print(
    FINE_TUNED_FINAL_MODEL_PATH
)

print()
print(
    "Validation accuracy graph:"
)

print(
    ACCURACY_GRAPH_PATH
)

print()
print(
    "Validation loss graph:"
)

print(
    LOSS_GRAPH_PATH
)

print()
print("=" * 70)