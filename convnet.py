import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from keras import layers
from keras import metrics
from keras.models import Sequential

# Parameters
batch_size = 64
img_height = 200
img_width = 200

# Load in files
train_ds = tf.keras.utils.image_dataset_from_directory('data',
                                                       validation_split=0.2,
                                                       subset="training",
                                                       seed=1,
                                                       color_mode="grayscale",
                                                       image_size=(img_height, img_width),
                                                       batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory('data',
                                                     validation_split=0.2,
                                                     subset="validation",
                                                     seed=1,
                                                     color_mode="grayscale",
                                                     image_size=(img_height, img_width),
                                                     batch_size=batch_size)

class_names = train_ds.class_names
print("Classes = " + str(class_names))

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")

# plt.show()

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

model = Sequential([
    layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 1)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# View all the layers of the network using Keras Model.summary method
model.summary()

# Train the model
epochs = 5
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Test the model
for i in range(10):
    img = tf.keras.utils.load_img(
        "unique/unique/test_trace_" + str(i) + ".png", color_mode="grayscale", target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print("Image = test_trace_" + str(i) + ".png")
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
#
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs_range = range(epochs)
#
# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()
