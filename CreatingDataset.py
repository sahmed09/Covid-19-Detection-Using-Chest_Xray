import pandas as pd
import os
import shutil
import random

# create the dataset for covid positive samples(github)
file_path_covid = "covid-chestxray/metadata.csv"
images_path_covid = "covid-chestxray/images"

df = pd.read_csv(file_path_covid)  # creating dataframe from csv file
# print(df.shape)  # (950, 30)
# print(df["finding"])

target_directory = "Dataset/Covid"

if not os.path.exists(target_directory):
    os.mkdir(target_directory)
    print("Covid folder created")

# pick images from covid chestxray folder and place it to Dataset/Covid folder
count = 0
for (i, row) in df.iterrows():
    if row["finding"] == "Pneumonia/Viral/COVID-19" and row["view"] == "PA":
        # row["view"] == "PA" we only taking those images which have frontal view
        filename = row["filename"]
        image_path = os.path.join(images_path_covid, filename)
        image_copy_path = os.path.join(target_directory, filename)
        shutil.copy2(image_path, image_copy_path)  # copy from source to destination
        # print("Moving image", count)
        count += 1
print(count)

# Sampling of images from kaggle:
kaggle_file_path = "chest_xray_kaggle/train/NORMAL"
target_normal_directory = "Dataset/Normal"

image_names = os.listdir(kaggle_file_path)
random.shuffle(image_names)

for i in range(196):
    image_name = image_names[i]
    image_path = os.path.join(kaggle_file_path, image_name)
    target_path = os.path.join(target_normal_directory, image_name)
    shutil.copy2(image_path, target_path)
