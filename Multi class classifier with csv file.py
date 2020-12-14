import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd

def get_data(filename):
    with open(filename) as training_file:
        csv_reader = csv.reader(training_file, delimiter=",")
        temp_images = []
        temp_labels = []
        flag = True
        for row in csv_reader:
            if flag:
                flag = False
                continue
            else:
                temp_labels.append(row[0])
                image_data = row[1:785]
                image_array = np.array_split(image_data, 28)
                temp_images.append(image_array)
            images = np.array(temp_images).astype('float')
            labels = np.array(temp_labels).astype('float')
        return images, labels

path_sign_mnist_train = f"{getcwd()}/sign_mnist_train.csv"
path_sign_mnist_test = f"{getcwd()}/sign_mnist_test.csv"
training_images, training_labels = get_data(path_sign_mnist_train)
testing_images, testing_labels = get_data(path_sign_mnist_test)

print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)

training_images = np.expand_dims(training_images, axis=-1)
testing_images = np.expand_dims(testing_images, axis=-1)

# Create an ImageDataGenerator and do Image Augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )
train_generator = train_datagen.flow(training_images,
                                     training_labels,
                                     batch_size=32)


validation_datagen = ImageDataGenerator(
     rescale=1. / 255)

valdiation_generator = validation_datagen.flow(testing_images,
                                               testing_labels,
                                               batch_size=32)
# Keep These
print(training_images.shape)
print(testing_images.shape)


# Define the model
# Use no more than 2 Conv2D and 2 MaxPooling2D
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(26, activation=tf.nn.softmax)])

# Compile Model.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the Model
history = model.fit(train_generator, epochs=25, steps_per_epoch=len(training_images)/32, validation_data = valdiation_generator, verbose = 1, validation_steps=len(testing_images)/32)

model.evaluate(testing_images, testing_labels, verbose=0)


# Plot the chart for accuracy and loss on both training and validation
import matplotlib.pyplot as plt
###c
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()