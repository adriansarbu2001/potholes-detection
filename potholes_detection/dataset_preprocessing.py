import os
import cv2
import shutil

# Input directories
mask_folder = "C:/UBB/Licenta/pothole-mix/training/cracks-and-potholes-in-road/masks"
image_folder = "C:/UBB/Licenta/pothole-mix/training/cracks-and-potholes-in-road/images"

# Output directories
new_mask_folder = "C:/UBB/Licenta/potholes-detection/potholes_detection/data/potholes_on_road/training/masks"
new_image_folder = "C:/UBB/Licenta/potholes-detection/potholes_detection/data/potholes_on_road/training/images"

# Create the output folders if they don't exist
if not os.path.exists(new_mask_folder):
    os.makedirs(new_mask_folder)
if not os.path.exists(new_image_folder):
    os.makedirs(new_image_folder)

# Get a list of all mask filenames in the mask folder
mask_filenames = [f for f in os.listdir(mask_folder) if f.endswith('.png')]

# Process each mask file and copy the corresponding real image
counter = 0  # Counter for generating the four-digit numbers
for mask_filename in mask_filenames:
    mask_path = os.path.join(mask_folder, mask_filename)
    image = cv2.imread(mask_path)

    # Check if the image contains at least one red pixel
    if cv2.countNonZero(image[:, :, 2]) > 0:
        # Replace red pixels with white and green pixels with black
        red_pixels = (image[:, :, 2] > 0)
        image[red_pixels] = [255, 255, 255]  # Set red pixels to white
        image[~red_pixels] = [0, 0, 0]  # Set non-red pixels to black

        # Generate four-digit number as filename
        filename = "{:04d}".format(counter)

        # Save the processed mask image
        new_mask_filename = filename + ".png"
        new_mask_path = os.path.join(new_mask_folder, new_mask_filename)
        cv2.imwrite(new_mask_path, image)

        # Get the corresponding image file path
        image_filename = os.path.splitext(mask_filename)[0] + ".jpg"
        image_path = os.path.join(image_folder, image_filename)

        # Copy the real image to the new image folder
        new_image_path = os.path.join(new_image_folder, filename + ".jpg")
        shutil.copy(image_path, new_image_path)

        counter += 1  # Increment the counter

print("Preprocessing complete.")
