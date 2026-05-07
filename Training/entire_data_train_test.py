
import os
import shutil
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# STEP 1: Paths
base_path = r"D:/Maitri folder/Microsoft COCO.v2-raw.yolov11/train/LivingRoom"

# ✅ Updated line — no 'images' subfolder now
source_images = base_path  

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
split_index = int(0.9 * len(all_images))
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































import os
import shutil
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# STEP 1: Paths
base_path = r"D:/Maitri folder/Microsoft COCO.v2-raw.yolov11/train/find_image_specific"
source_images = base_path  
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

# STEP 4: Move images to respective folders with random class
for img in train_imgs:
    cls = random.choice(classes)
    shutil.copy2(os.path.join(source_images, img), os.path.join(train_path, cls, img))

for img in test_imgs:
    cls = random.choice(classes)
    shutil.copy2(os.path.join(source_images, img), os.path.join(test_path, cls, img))

# STEP 5: Data Augmentation + Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
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

# STEP 6: Enhanced CNN Model
model = Sequential([
    Input(shape=(150, 150, 3)),

    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# STEP 7: Compile Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# STEP 8: Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# STEP 9: Train the Model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator,
    callbacks=[early_stop, checkpoint]
)

# STEP 10: Evaluate
loss, acc = model.evaluate(test_generator)
print(f"\n✅ Final Test Accuracy: {acc * 100:.2f}%")





























