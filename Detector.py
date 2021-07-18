# import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.models import *
from keras.preprocessing import image
from keras.losses import binary_crossentropy

# let's define the Train and Validation dataset
train_path = 'Dataset/Train'
validation_path = 'Dataset/Validation'

# CNN Based model in Keras
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 input_shape=(224, 224, 3)))  # height=224, width=224, no of filter=3
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=binary_crossentropy, optimizer='adam', metrics=['accuracy'])

model.summary()

# Train from scratch:
train_datagen = image.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = image.ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'Dataset/Train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
# print(train_generator.class_indices)
# print(train_generator.image_shape)  # (224, 224, 3)

validation_generator = validation_datagen.flow_from_directory(
    'Dataset/Validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# print(validation_generator.class_indices)
# print(validation_generator.image_shape)  # (224, 224, 3)


"""def plotImages(images):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()

    for img, ax in zip(images, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


sample_training_images, _ = next(train_generator)
plotImages(sample_training_images[:5])"""

# fit_generator--->If we have ImageDataGenerator for our Image dataset creation
hist = model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=25,
    validation_data=validation_generator,
    validation_steps=2
)

model.save('covid19.h5')

evaluate_train = model.evaluate(train_generator)
evaluate_validation = model.evaluate(validation_generator)

accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
ephocs_range = range(25)

plt.figure(figsize=(15, 15))
# plt.subplot(1, 2, 1)
plt.plot(ephocs_range, accuracy, label='Training accuracy')
plt.plot(ephocs_range, val_accuracy, label='Validation accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy')
plt.show()

plt.figure(figsize=(15, 15))
# plt.subplot(1, 2, 2)
plt.plot(ephocs_range, loss, label='Training loss')
plt.plot(ephocs_range, val_loss, label='Validation loss')
plt.legend(loc='upper right')
plt.title('Loss')

plt.show()
