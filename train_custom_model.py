from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

DATASET_DIR = Path("dataset")
MODEL_DIR = Path("custom_model")
MODEL_PATH = MODEL_DIR / "model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.txt"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10


def main():
    if not DATASET_DIR.exists():
        print("Dataset directory not found.")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="both",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    train_data, val_data = train_ds

    class_names = train_data.class_names
    num_classes = len(class_names)

    if num_classes < 2:
        print("At least two classes are required.")
        return

    with open(CLASS_NAMES_PATH, "w") as f:
        for name in class_names:
            f.write(name + "\n")

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS
    )

    model.save(MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Classes: {class_names}")


if __name__ == "__main__":
    main()
