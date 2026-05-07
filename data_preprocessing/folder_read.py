import os

# Specify the folder path
folder_path = "D:/Maitri folder/Microsoft COCO.v2-raw.yolov11"

# List all files and subfolders in the specified directory
try:
    files = os.listdir(folder_path)
    print(f"Contents of '{folder_path}':")
    for file in files:
        print(file)
except FileNotFoundError:
    print("The specified folder does not exist.")
except PermissionError:
    print("You do not have permission to access this folder.")


print("\n✅ All files read successfully! Keep going—you’re creating something awesome! 🚀🔥")