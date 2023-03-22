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
img_height = 200
img_width = 200
class_names = ['arched-line-ricochet', 'classic-pursuit', 'large-circle-pursuit',
               'medium-circle-pursuit', 'small-circle-pursuit', 'straight-line-ricochet']

# Set up model
num_classes = len(class_names)


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


# Load checkpoint from model to pull in weights
checkpoint_dir = 'cnn/checkpoints/cnn-02.ckpt'

# Create a new model instance
model = create_model()

# Load the previously saved weights from checkpoint
model.load_weights(checkpoint_dir).expect_partial()

# Print a summary
model.summary()

# Test the model
for i in range(50):
    img = tf.keras.utils.load_img(
        "unique/unique/test" + str(i) + ".png", color_mode="grayscale", target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print("Length of score vector = " + str(len(score)))

    print("Image = test" + str(i) + ".png")
    print("Predictions:")
    for j in range(len(score)):
        print(
            "\t Class = {} -> Confidence = {:.2f}"
            .format(class_names[j], 100 * score[j])
        )