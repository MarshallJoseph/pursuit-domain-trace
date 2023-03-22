import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import os

from keras import layers
from keras import metrics
from keras.models import Sequential

# We only care about high priority log errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Parameters
batch_size = 32
img_height = 200
img_width = 200

# Load in files
train_ds = tf.keras.utils.image_dataset_from_directory('data',
                                                       validation_split=0.2,
                                                       subset="training",
                                                       seed=3,
                                                       color_mode="grayscale",
                                                       image_size=(img_height, img_width),
                                                       batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory('data',
                                                     validation_split=0.2,
                                                     subset="validation",
                                                     seed=3,
                                                     color_mode="grayscale",
                                                     image_size=(img_height, img_width),
                                                     batch_size=batch_size)

class_names = train_ds.class_names
print("Classes = " + str(class_names))

for image_batch, labels_batch in train_ds:
    print("Image Batch Shape = " + str(image_batch.shape))
    # print(labels_batch.shape)
    break

AUTOTUNE = tf.data.AUTOTUNE
# Dataset.cache keeps the images in memory after they're loaded off disk during the first epoch.
# This ensures the dataset does not become a bottleneck while training the model.
# Dataset.prefect overlaps data preprocessing and model execution while training.
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Set up model
num_classes = len(class_names)

# Create checkpoint path
checkpoint_path = "cnn/checkpoints/cnn-{epoch:02d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a tf.keras.callbacks.ModelCheckpoint callback that saves weights only during training
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq='epoch'
)


def create_model():
    cnn = Sequential([
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 1)),
        layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(rate=0.25),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(rate=0.5),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(rate=0.5),
        layers.BatchNormalization(),
        layers.Dense(num_classes, name="outputs", activation='relu')
    ])

    # Compile the model
    cnn.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return cnn


# Create an instance of the cnn model
model = create_model()

# View all the layers of the network using Keras Model.summary method
model.summary()

# Save the weights using the 'checkpoint_path' format
model.save_weights(checkpoint_path.format(epoch=0))

# Train the model
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[cp_callback]
)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
