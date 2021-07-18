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
# loss: 0.0628 - accuracy: 0.9839 - val_loss: 0.1658 - val_accuracy: 0.9531

evaluate_train = model.evaluate(train_generator)
print(evaluate_train)  # [0.05581784248352051, 0.9892857074737549]
evaluate_validation = model.evaluate(validation_generator)
print(evaluate_validation)  # [0.10265020281076431, 0.9800000190734863]

"""
using batch size 16:
loss: 0.1526 - accuracy: 0.9500 - val_loss: 0.0815 - val_accuracy: 1.0000
[0.13616831600666046, 0.9607142806053162]
[0.14550650119781494, 0.9700000286102295]

using batch size 32:
loss: 0.0527 - accuracy: 0.9879 - val_loss: 0.0990 - val_accuracy: 0.9688
[0.030146615579724312, 0.9892857074737549]
[0.12272470444440842, 0.949999988079071]
"""

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

"""
Epoch 1/25
8/8 [==============================] - 9s 1s/step - loss: 0.7601 - accuracy: 0.6008 - val_loss: 0.6215 - val_accuracy: 0.5781
Epoch 2/25
8/8 [==============================] - 6s 759ms/step - loss: 0.5497 - accuracy: 0.7339 - val_loss: 0.5059 - val_accuracy: 0.8125
Epoch 3/25
8/8 [==============================] - 6s 787ms/step - loss: 0.3367 - accuracy: 0.8790 - val_loss: 0.4609 - val_accuracy: 0.8906
Epoch 4/25
8/8 [==============================] - 6s 792ms/step - loss: 0.2319 - accuracy: 0.9073 - val_loss: 0.2490 - val_accuracy: 0.9219
Epoch 5/25
8/8 [==============================] - 6s 763ms/step - loss: 0.1912 - accuracy: 0.9274 - val_loss: 0.1915 - val_accuracy: 0.9844
Epoch 6/25
8/8 [==============================] - 6s 775ms/step - loss: 0.1409 - accuracy: 0.9597 - val_loss: 0.2332 - val_accuracy: 0.9531
Epoch 7/25
8/8 [==============================] - 6s 778ms/step - loss: 0.1045 - accuracy: 0.9637 - val_loss: 0.0759 - val_accuracy: 0.9844
Epoch 8/25
8/8 [==============================] - 6s 782ms/step - loss: 0.2039 - accuracy: 0.9609 - val_loss: 0.2658 - val_accuracy: 0.9219
Epoch 9/25
8/8 [==============================] - 6s 759ms/step - loss: 0.0983 - accuracy: 0.9677 - val_loss: 0.1762 - val_accuracy: 0.9688
Epoch 10/25
8/8 [==============================] - 6s 739ms/step - loss: 0.1086 - accuracy: 0.9798 - val_loss: 0.1908 - val_accuracy: 0.9531
Epoch 11/25
8/8 [==============================] - 6s 770ms/step - loss: 0.0673 - accuracy: 0.9879 - val_loss: 0.1325 - val_accuracy: 0.9688
Epoch 12/25
8/8 [==============================] - 6s 776ms/step - loss: 0.1038 - accuracy: 0.9718 - val_loss: 0.1919 - val_accuracy: 0.9375
Epoch 13/25
8/8 [==============================] - 6s 780ms/step - loss: 0.1013 - accuracy: 0.9556 - val_loss: 0.2208 - val_accuracy: 0.9375
Epoch 14/25
8/8 [==============================] - 6s 762ms/step - loss: 0.1412 - accuracy: 0.9597 - val_loss: 0.2685 - val_accuracy: 0.9062
Epoch 15/25
8/8 [==============================] - 6s 763ms/step - loss: 0.0964 - accuracy: 0.9556 - val_loss: 0.1797 - val_accuracy: 0.9375
Epoch 16/25
8/8 [==============================] - 6s 808ms/step - loss: 0.1137 - accuracy: 0.9395 - val_loss: 0.1812 - val_accuracy: 0.9688
Epoch 17/25
8/8 [==============================] - 6s 748ms/step - loss: 0.0771 - accuracy: 0.9758 - val_loss: 0.1685 - val_accuracy: 0.9531
Epoch 18/25
8/8 [==============================] - 6s 770ms/step - loss: 0.0366 - accuracy: 0.9879 - val_loss: 0.1893 - val_accuracy: 0.9531
Epoch 19/25
8/8 [==============================] - 6s 786ms/step - loss: 0.0722 - accuracy: 0.9758 - val_loss: 0.0679 - val_accuracy: 0.9844
Epoch 20/25
8/8 [==============================] - 6s 774ms/step - loss: 0.0342 - accuracy: 0.9960 - val_loss: 0.1473 - val_accuracy: 0.9531
Epoch 21/25
8/8 [==============================] - 6s 786ms/step - loss: 0.0637 - accuracy: 0.9718 - val_loss: 0.1237 - val_accuracy: 0.9531
Epoch 22/25
8/8 [==============================] - 6s 774ms/step - loss: 0.1106 - accuracy: 0.9677 - val_loss: 0.1714 - val_accuracy: 0.9531
Epoch 23/25
8/8 [==============================] - 6s 778ms/step - loss: 0.1002 - accuracy: 0.9718 - val_loss: 0.1605 - val_accuracy: 0.9531
Epoch 24/25
8/8 [==============================] - 6s 796ms/step - loss: 0.0481 - accuracy: 0.9879 - val_loss: 0.0931 - val_accuracy: 0.9844
Epoch 25/25
8/8 [==============================] - 6s 767ms/step - loss: 0.0391 - accuracy: 0.9879 - val_loss: 0.1431 - val_accuracy: 0.9688
9/9 [==============================] - 5s 608ms/step - loss: 0.0558 - accuracy: 0.9893
[0.05581784248352051, 0.9892857074737549]
4/4 [==============================] - 1s 258ms/step - loss: 0.1027 - accuracy: 0.9800
[0.10265020281076431, 0.9800000190734863]
"""
