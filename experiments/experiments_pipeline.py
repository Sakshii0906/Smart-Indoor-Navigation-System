# folder read ##############################################################
import os

folder_path = 'C:/Users/P G Chopda/Desktop/supplier/Microsoft COCO.v2-raw.yolov11'
print(f"Current working directory: {os.getcwd()}")
print(f"Folder exists: {os.path.exists(folder_path)}")
print(f"Files in folder: {os.listdir(folder_path) if os.path.exists(folder_path) else 'Folder not found'}")

#
################################--->  images read ###############################


# import the modules
import os
from os import listdir

# get the path/directory
folder_dir = "C:/Users/P G Chopda/Desktop/supplier/Microsoft COCO.v2-raw.yolov11"
for images in os.listdir(folder_dir):

	# check if the image ends with png
	if (images.endswith(".jpg")):
		print(images)

#
################################ ---> labels read ###############################

import os

# Specify the folder directory
folder_dir = "C:/Users/P G Chopda/Desktop/supplier/Microsoft COCO.v2-raw.yolov11/train/labels"

# Loop through all files in the directory
for file in os.listdir(folder_dir):
    # Check if the file ends with .txt
    if file.endswith(".txt"):
        file_path = os.path.join(folder_dir, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"Contents of {file}:\n{content}\n{'-'*50}\n")
            
print("\n✅ All text files read successfully! Keep going—you’re doing amazing! 🚀🔥")


#
############################### combine all images and labels  ##############################################################



import os
import shutil

# Paths
image_dir = "C:/Users/P G Chopda/Desktop/supplier/Microsoft COCO.v2-raw.yolov11/train/images"
label_dir = "C:/Users/P G Chopda/Desktop/supplier/Microsoft COCO.v2-raw.yolov11/train/labels"
output_dir = "C:/Users/P G Chopda/Desktop/supplier/Microsoft COCO.v2-raw.yolov11/train_combined"

os.makedirs(output_dir, exist_ok=True)

# Get sorted lists
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

min_len = min(len(image_files), len(label_files))

matched_count = 0

for i in range(min_len):
    img_ext = os.path.splitext(image_files[i])[1]
    
    new_basename = f"{i+1}"
    new_image_name = f"{new_basename}{img_ext}"
    new_label_name = f"{new_basename}.txt"
    
    # Full paths
    old_img_path = os.path.join(image_dir, image_files[i])
    old_lbl_path = os.path.join(label_dir, label_files[i])
    new_img_path = os.path.join(output_dir, new_image_name)
    new_lbl_path = os.path.join(output_dir, new_label_name)
    
    # Copy and rename to combined folder
    shutil.copy(old_img_path, new_img_path)
    shutil.copy(old_lbl_path, new_lbl_path)
    
    matched_count += 1

print(f"\n✅ Total matched and renamed pairs: {matched_count}")
print(f"📂 All copied to: {output_dir}")

# check  image are combine or not ##############################################################

import os

# Set the path to your combined folder
combined_folder = "C:/Users/P G Chopda/Desktop/supplier/Microsoft COCO.v2-raw.yolov11/train_combined"

# List image and label files
image_files = [f for f in os.listdir(combined_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
label_files = [f for f in os.listdir(combined_folder) if f.lower().endswith('.txt')]

# Extract base names (filename without extension)
image_basenames = set(os.path.splitext(f)[0] for f in image_files)
label_basenames = set(os.path.splitext(f)[0] for f in label_files)

# Find matches and mismatches
matched = image_basenames & label_basenames
images_without_labels = image_basenames - label_basenames
labels_without_images = label_basenames - image_basenames

# Optional: print details (only if mismatches exist)
if images_without_labels:
    print("\n🔍 Images without matching labels:")
    for img in sorted(images_without_labels):
        print(f"- {img}")

if labels_without_images:
    print("\n🔍 Labels without matching images:")
    for lbl in sorted(labels_without_images):
        print(f"- {lbl}")


# Display summary
print("\n🎯 Final Matching Report:\n")
print(f"✅ Matched pairs: {len(matched)}")
print(f"❌ Images without labels: {len(images_without_labels)}")
print(f"❌ Labels without images: {len(labels_without_images)}")


print("\n🚀 Done! You're good to go, Maitri! 🔥")



###################################   split 100 data    ##########3###############################



import os
import shutil

# Base paths
base_path = r"C:/Users/P G Chopda/Desktop/supplier/Microsoft COCO.v2-raw.yolov11/train"
images_path = os.path.join(base_path, "images")

# Output folder
sample_folder = os.path.join(base_path, "sample_100")
sample_images = os.path.join(sample_folder, "images")
os.makedirs(sample_images, exist_ok=True)

# Get all image files
image_extensions = ('.jpg', '.jpeg', '.png')
image_files = [f for f in os.listdir(images_path) if f.lower().endswith(image_extensions)]

# Take only first 100
image_files = image_files[:100]

# Copy to sample folder
for img in image_files:
    src = os.path.join(images_path, img)
    dst = os.path.join(sample_images, img)
    shutil.copy2(src, dst)

print(f"✅ Copied {len(image_files)} images to: {sample_images}")



###################################  train - test split 100 data    ##########3###############################

import os
import shutil
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# STEP 1: Paths
base_path = r"C:/Users/P G Chopda/Desktop/supplier/Microsoft COCO.v2-raw.yolov11/train"
source_images = os.path.join(base_path, "images")
sample_folder = os.path.join(base_path, "sample_100")
sample_images = os.path.join(sample_folder, "images")

# Create sample_100/images
os.makedirs(sample_images, exist_ok=True)

# Copy 100 images to sample folder
image_extensions = ('.jpg', '.jpeg', '.png')
image_files = [f for f in os.listdir(source_images) if f.lower().endswith(image_extensions)]
random.shuffle(image_files)
image_files = image_files[:100]

for img in image_files:
    shutil.copy2(os.path.join(source_images, img), os.path.join(sample_images, img))

# STEP 2: Create folders for train/test with fake class folders
train_path = os.path.join(sample_folder, "train")
test_path = os.path.join(sample_folder, "test")
classes = ["class1", "class2"]

for cls in classes:
    os.makedirs(os.path.join(train_path, cls), exist_ok=True)
    os.makedirs(os.path.join(test_path, cls), exist_ok=True)

# STEP 3: Split into train/test and assign to classes randomly
all_images = os.listdir(sample_images)
random.shuffle(all_images)
split_index = int(0.8 * len(all_images))
train_imgs = all_images[:split_index]
test_imgs = all_images[split_index:]

for img in train_imgs:
    cls = random.choice(classes)
    shutil.move(os.path.join(sample_images, img), os.path.join(train_path, cls, img))

for img in test_imgs:
    cls = random.choice(classes)
    shutil.move(os.path.join(sample_images, img), os.path.join(test_path, cls, img))

# STEP 4: Image preprocessing
train_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'
)

# STEP 5: Build CNN Model
model = Sequential([
    Input(shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# STEP 6: Train the model
history = model.fit(
    train_generator,
    epochs=5,
    validation_data=test_generator
)

# STEP 7: Evaluate accuracy
loss, acc = model.evaluate(test_generator)
print(f"\n✅ Final Test Accuracy: {acc * 100:.2f}%")



###################################  train - test entire train folder data    ##########3###############################




import os
import shutil
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# STEP 1: Paths
base_path = r"C:/Users/P G Chopda/Desktop/supplier/Microsoft COCO.v2-raw.yolov11/train"
source_images = os.path.join(base_path, "images")
full_data_folder = os.path.join(base_path, "full_dataset")
train_path = os.path.join(full_data_folder, "train")
test_path = os.path.join(full_data_folder, "test")
classes = ["class1", "class2"]

# Clean old dataset folder if exists
if os.path.exists(full_data_folder):
    shutil.rmtree(full_data_folder)

# Create new folders
for cls in classes:
    os.makedirs(os.path.join(train_path, cls), exist_ok=True)
    os.makedirs(os.path.join(test_path, cls), exist_ok=True)

# STEP 2: Load all image files
image_extensions = ('.jpg', '.jpeg', '.png')
all_images = [f for f in os.listdir(source_images) if f.lower().endswith(image_extensions)]
random.shuffle(all_images)

# STEP 3: Split into train/test
split_index = int(0.8 * len(all_images))
train_imgs = all_images[:split_index]
test_imgs = all_images[split_index:]

# STEP 4: Move images to respective folders with random fake class
for img in train_imgs:
    cls = random.choice(classes)
    shutil.copy2(os.path.join(source_images, img), os.path.join(train_path, cls, img))

for img in test_imgs:
    cls = random.choice(classes)
    shutil.copy2(os.path.join(source_images, img), os.path.join(test_path, cls, img))

# STEP 5: Image preprocessing
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# STEP 6: Build CNN Model
model = Sequential([
    Input(shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# STEP 7: Train the model
history = model.fit(
    train_generator,
    epochs=5,
    validation_data=test_generator
)

# STEP 8: Evaluate accuracy
loss, acc = model.evaluate(test_generator)
print(f"\n✅ Final Test Accuracy: {acc * 100:.2f}%")








############################# open camarea adn detect real time photo


from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
import cv2

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # Load your trained model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    outputs = predictor(frame)

    v = Visualizer(frame[:, :, ::-1], scale=1.0)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Obstacle Detection", out.get_image()[:, :, ::-1])
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()














import cv2
import torch
import pyttsx3

# Load pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # You can choose a different model ('yolov5m', 'yolov5l', 'yolov5x')

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Start video capture (use your webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # Object Detection with YOLOv5
    results = model(frame)
    
    # Get detected objects
    detected_objects = results.names
    object_coords = results.xywh[0]  # Bounding box coordinates

    # Visualize detections
    results.render()  # Renders boxes on the image
    cv2.imshow('Detection', frame)
    
    # Check if an obstacle is detected (adjust based on your setup)
    if 'person' in detected_objects:  # Assuming 'person' represents obstacles for a blind person
        engine.say("Obstacle detected ahead")
        engine.runAndWait()

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()













































































































































































































































import os
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torchvision.ops import box_iou

# 🔹 Set image path
image_path = "D:/Maitri folder/Microsoft COCO.v2-raw.yolov11/train/find_image_specific"

# 🔹 Custom Dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted([
            f for f in os.listdir(root) if f.endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.imgs = self.imgs[:10]  # ✅ Limit for testing
        print(f"Total images loaded: {len(self.imgs)}")

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        # ✅ Dummy bounding box (replace with real annotations later)
        boxes = torch.tensor([[30,30,150,150]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

# 🔹 Load full dataset
full_dataset = CustomDataset(image_path, transforms=F.to_tensor)

# 🔹 Train-test split (80-20)
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
test_size = total_size - train_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# 🔹 Load and modify pretrained Faster R-CNN
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

# 🔹 Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# 🔹 Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# 🔹 Training Loop
num_epochs = 2
print("Starting training...\n")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (images, targets) in enumerate(train_loader):
        print(f"Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}")

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    print(f">>> Epoch {epoch+1} finished. Loss: {running_loss:.4f}\n")

# 🔹 Save model
torch.save(model.state_dict(), "faster_rcnn_model.pth")
print("✅ Model saved as faster_rcnn_model.pth")

# 🔹 Evaluation on Test Set (IoU-based)
model.eval()
correct = 0
total = 0

print("Evaluating accuracy on test data...")

with torch.no_grad():
    for i, (images, targets) in enumerate(test_loader):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for pred, target in zip(outputs, targets):
            pred_boxes = pred['boxes'].cpu()
            true_boxes = target['boxes'].cpu()

            if len(pred_boxes) == 0:
                continue

            ious = box_iou(pred_boxes, true_boxes)
            max_iou = ious.max().item()

            if max_iou > 0.5:
                correct += 1
            total += 1

accuracy = correct / total if total > 0 else 0
print(f"✅ Test Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")




