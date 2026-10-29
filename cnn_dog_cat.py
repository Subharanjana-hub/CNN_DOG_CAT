import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 30
LOCAL_DATA_DIR = "dataset"
DATA_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
OUT_DIR = "cnn_dog_cat_outputs"
CLASS_NAMES = ["cat", "dog"]

keras.utils.set_random_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)


def get_data_dirs():
    local_train = os.path.join(LOCAL_DATA_DIR, "train")
    local_val = os.path.join(LOCAL_DATA_DIR, "validation")
    if os.path.isdir(local_train) and os.path.isdir(local_val):
        print(f"Using your own dataset from '{LOCAL_DATA_DIR}/'")
        return local_train, local_val
    print("Downloading the cats and dogs dataset (about 65 MB) ...")
    zip_path = keras.utils.get_file("cats_and_dogs_filtered.zip", origin=DATA_URL, extract=True)
    base = os.path.join(os.path.dirname(zip_path), "cats_and_dogs_filtered")
    return os.path.join(base, "train"), os.path.join(base, "validation")


train_dir, val_dir = get_data_dirs()

train_ds = keras.utils.image_dataset_from_directory(
    train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="binary", shuffle=True, seed=SEED)
val_ds = keras.utils.image_dataset_from_directory(
    val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="binary", shuffle=False)

print("Classes:", train_ds.class_names, "(cat = 0, dog = 1)")

plt.figure(figsize=(10, 6))
for images, labels in train_ds.take(1):
    for i in range(8):
        ax = plt.subplot(2, 4, i + 1)
        ax.imshow(images[i].numpy().astype("uint8"))
        ax.set_title(CLASS_NAMES[int(labels[i][0])])
        ax.axis("off")
plt.suptitle("Sample training images")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/samples.png", dpi=120)
plt.close()

train_ds = train_ds.cache().prefetch(buffer_size=2)
val_ds = val_ds.cache().prefetch(buffer_size=2)


def build_cnn(input_shape=(150, 150, 3)):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.Rescaling(1.0 / 255),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="cnn_dog_cat")
    return model


model = build_cnn(IMG_SIZE + (3,))
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                  restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                      patience=3, min_lr=1e-6, verbose=1),
    keras.callbacks.ModelCheckpoint(f"{OUT_DIR}/best_cnn.keras",
                                    monitor="val_accuracy",
                                    save_best_only=True, verbose=0),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2,
)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history["accuracy"], label="train")
ax[0].plot(history.history["val_accuracy"], label="validation")
ax[0].set_title("Accuracy")
ax[0].set_xlabel("Epoch")
ax[0].legend()
ax[1].plot(history.history["loss"], label="train")
ax[1].plot(history.history["val_loss"], label="validation")
ax[1].set_title("Loss")
ax[1].set_xlabel("Epoch")
ax[1].legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/training_curves.png", dpi=120)
plt.close()

val_loss, val_acc = model.evaluate(val_ds, verbose=0)
print(f"\nValidation loss    : {val_loss:.4f}")
print(f"Validation accuracy: {val_acc:.4f}")

y_true = np.concatenate([y.numpy().ravel() for _, y in val_ds]).astype(int)
y_prob = model.predict(val_ds, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

print("\nClassification report:\n")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(4.5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_xticklabels(CLASS_NAMES)
ax.set_yticks([0, 1])
ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.colorbar(im)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=120)
plt.close()

fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(5, 4.5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curve.png", dpi=120)
plt.close()
print(f"ROC AUC: {roc_auc:.4f}")

plt.figure(figsize=(13, 7))
offset = 0
for images, labels in val_ds.take(1):
    probs = model.predict(images, verbose=0).ravel()
    for i in range(min(12, len(images))):
        ax = plt.subplot(3, 4, i + 1)
        ax.imshow(images[i].numpy().astype("uint8"))
        pred = int(probs[i] >= 0.5)
        conf = probs[i] if pred == 1 else 1 - probs[i]
        true = int(labels[i][0])
        ax.set_title(f"{CLASS_NAMES[pred]} ({conf:.0%})\ntrue: {CLASS_NAMES[true]}",
                     color="green" if pred == true else "red", fontsize=9)
        ax.axis("off")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/predictions.png", dpi=120)
plt.close()


def predict_image(path):
    img = keras.utils.load_img(path, target_size=IMG_SIZE)
    arr = np.expand_dims(keras.utils.img_to_array(img), axis=0)
    p = float(model.predict(arr, verbose=0)[0][0])
    label = CLASS_NAMES[int(p >= 0.5)]
    conf = p if p >= 0.5 else 1 - p
    print(f"{path} -> {label.upper()} ({conf:.1%})")
    return label


model.save(f"{OUT_DIR}/cnn_dog_cat.keras")
print(f"\nAll outputs saved in: {OUT_DIR}/")
print("To test your own picture, add this line at the end of the script:")
print('    predict_image("my_pet.jpg")')
