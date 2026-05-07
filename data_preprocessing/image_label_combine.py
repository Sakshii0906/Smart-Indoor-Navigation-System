# image and label combine in train_combined folder and check mismatch or not


import os
import shutil

# Paths
image_dir = "D:/Maitri folder/Microsoft COCO.v2-raw.yolov11/train/images"
label_dir = "D:/Maitri folder/Microsoft COCO.v2-raw.yolov11/train/labels"
output_dir = "D:/Maitri folder/Microsoft COCO.v2-raw.yolov11/train_combined"

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

# image and lable are matched report



import os

# Set the path to your combined folder
combined_folder = "D:/Maitri folder/Microsoft COCO.v2-raw.yolov11/train_combined"

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
