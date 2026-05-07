


# import the modules
import os
import shutil

# source folder containing images
folder_dir = "D:/Maitri folder/Microsoft COCO.v2-raw.yolov11/train_combined"

# destination folder for filtered images
filtered_dir = os.path.join(folder_dir, "filtered_images")
os.makedirs(filtered_dir, exist_ok=True)  # Create the folder if it doesn't exist

# loop through images
for images in os.listdir(folder_dir):
    # check if the image ends with .jpg
    if images.endswith(".jpg"):
        print(images)

        # full path of source image
        source_path = os.path.join(folder_dir, images)

        # full path of destination
        dest_path = os.path.join(filtered_dir, images)

        # move image
        shutil.move(source_path, dest_path)

print("\n✅ All images read and moved to 'filtered_images' folder successfully! 🚀🔥")
