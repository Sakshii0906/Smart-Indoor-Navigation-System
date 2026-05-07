
import os
import shutil
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# STEP 1: Paths
base_path = r"D:/Maitri folder/Microsoft COCO.v2-raw.yolov11/train"
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
print(f"\n✅ Final Test Accuracy sample data 100 : {acc * 100:.2f}%")