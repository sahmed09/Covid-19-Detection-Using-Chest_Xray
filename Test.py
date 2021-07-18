import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import cv2
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Testing Data:
model = load_model("covid19.h5")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

img = cv2.imread("Dataset/Test/Covid/yxppt-2020-02-19_00-51-27_287214-day8.jpg")
# print(img.shape)
img = cv2.resize(img, (224, 224))
img = np.reshape(img, (1, 224, 224, 3))
# print(img.shape)

classes = model.predict_classes(img)
classes = int(classes)
# print(classes)
label = ["COVID INFECTED", "NORMAL"]
print(label[classes])
print()

# Creating Confusion matrix:
model = load_model("covid19.h5")

y_actual = []
y_test = []

for i in os.listdir("Dataset/Test/Normal/"):
    img = image.load_img("Dataset/Test/Normal/" + i, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    # print(img.shape)
    p = model.predict_classes(img)
    # print(int(p))
    y_test.append(p[0, 0])
    y_actual.append(1)
# print(y_test, y_actual)

for i in os.listdir("Dataset/Test/Covid/"):
    img = image.load_img("Dataset/Test/Covid/" + i, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    p = model.predict_classes(img)
    # print(int(p))
    y_test.append(p[0, 0])
    y_actual.append(0)
# print(y_test, y_actual)

y_actual = np.array(y_actual)
y_test = np.array(y_test)
print(y_actual, y_test)

cm = confusion_matrix(y_actual, y_test)

sns.heatmap(cm, cmap="Blues", annot=True)
plt.show()

accuracy = accuracy_score(y_actual, y_test)
print("Accuracy :", accuracy)
precision = precision_score(y_actual, y_test)
print("Precision :", precision)
recall = recall_score(y_actual, y_test)
print("Recall :", recall)
f1_score = f1_score(y_actual, y_test)
print("F1_score :", f1_score)
