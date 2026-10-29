import os
import shutil

# Base dataset path
dataset_path = "./datasets/composite"
num_test_images = 20 # number of images to copy

# Define folders
trainA = os.path.join(dataset_path, "trainA")
trainB = os.path.join(dataset_path, "trainB")
testA = os.path.join(dataset_path, "testA")
testB = os.path.join(dataset_path, "testB")

# Create test folders if they don't exist
os.makedirs(testA, exist_ok=True)
os.makedirs(testB, exist_ok=True)

# Function to copy images
def copy_test_images(src_folder, dst_folder, num_images):
    images = sorted(os.listdir(src_folder))
    if not images:
        print(f"No images found in {src_folder}")
        return
    for img in images[:num_images]:
        shutil.copy(os.path.join(src_folder, img), os.path.join(dst_folder, img))
    print(f"Copied {min(num_images, len(images))} images from {src_folder} to {dst_folder}")

# Copy images
copy_test_images(trainA, testA, num_test_images)
copy_test_images(trainB, testB, num_test_images)

print("Test folders setup complete.")

