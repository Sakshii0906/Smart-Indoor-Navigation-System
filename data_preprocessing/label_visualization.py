
## ---> labels read

import os

# leble read in train folder
folder_dir = "D:/Maitri folder/Microsoft COCO.v2-raw.yolov11/train/labels"

# Loop through all files in the directory
for file in os.listdir(folder_dir):
    # Check if the file ends with .txt
    if file.endswith(".txt"):
        file_path = os.path.join(folder_dir, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"Contents of {file}:\n{content}\n{'-'*50}\n")
            
print("\n✅ All text files read successfully! Keep going—you’re doing amazing! 🚀🔥")



import os

# leble read in valid folder
folder_dir = "D:/Maitri folder/Microsoft COCO.v2-raw.yolov11/valid/labels"

# Loop through all files in the directory
for file in os.listdir(folder_dir):
    # Check if the file ends with .txt
    if file.endswith(".txt"):
        file_path = os.path.join(folder_dir, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"Contents of {file}:\n{content}\n{'-'*50}\n")
            
print("\n✅ All text files read successfully! Keep going—you’re doing amazing! 🚀🔥")