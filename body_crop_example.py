import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, draw
import ultralytics
from roboflow import Roboflow

ultralytics.checks()


# Function to read the coordinates and omit the first number
def read_coordinates(file_path):
    with open(file_path, "r") as file:
        data = file.read().strip().split()
        if data:
            # Remove the first number (0) and then convert the rest to float
            return np.array(data[1:], dtype=float).reshape(-1, 2)
        else:
            return np.array([])


# Function to crop the area based on given coordinates
def crop_polygon_area(image_path, coordinates):
    # Read the image
    image = io.imread(image_path)
    img_height, img_width = image.shape[:2]

    # Scale the normalized coordinates to the image size
    scaled_coords = np.copy(coordinates)
    scaled_coords[:, 0] *= img_width
    scaled_coords[:, 1] *= img_height
    scaled_coords = np.round(scaled_coords).astype(np.int32)

    # Create a mask with the same dimensions as the image, fill it with zeros (black)
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # Get the rows and cols to fill from the polygon coordinates
    rr, cc = draw.polygon(scaled_coords[:, 1], scaled_coords[:, 0], mask.shape)

    # Fill in the mask with white where the polygon is
    mask[rr, cc] = 1

    # Now, we'll use the mask to extract the relevant area from the image
    # The mask needs to be 3 channels like the image
    if len(image.shape) == 3:
        mask = np.dstack([mask] * image.shape[2])

    # Use the mask to extract the region of interest
    cropped_image = image * mask

    # Display the cropped area
    # plt.imshow(cropped_image)
    # plt.show()

    return cropped_image


if __name__ == "__main__":
    images_dir = "/home/ralcaraz/Documentos/git-repos/lizard-body-crop/datasets/lizard-recognition-2/train/images"
    labels_dir = "/home/ralcaraz/Documentos/git-repos/lizard-body-crop/datasets/lizard-recognition-2/train/labels"
    output_dir = "/home/ralcaraz/Documentos/git-repos/lizard-body-crop/cropped-datasets/cropped_images"

    os.makedirs("datasets", exist_ok=True)
    # %cd datasets in python:
    os.chdir("datasets")
    from roboflow import Roboflow

    rf = Roboflow(api_key="your_roboflow_api_key_here")
    project = rf.workspace("datalab-xz0n5").project("lizard-recognition")
    version = project.version(2)
    dataset = version.download("yolov8")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each file in the images directory
    for image_filename in os.listdir(images_dir):
        if image_filename.endswith(".jpg"):
            base_filename = os.path.splitext(image_filename)[0]
            image_path = os.path.join(images_dir, image_filename)
            label_path = os.path.join(labels_dir, base_filename + ".txt")

            # Check if the corresponding label file exists
            if os.path.exists(label_path):
                norm_coords = read_coordinates(label_path)
                cropped_image = crop_polygon_area(image_path, norm_coords)

                # Save the cropped image
                output_image_path = os.path.join(output_dir, image_filename)
                io.imsave(output_image_path, cropped_image)
                print(f"Cropped image saved to {output_image_path}")
            else:
                print(f"No corresponding label file found for {image_filename}")
