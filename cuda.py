import cv2
import numpy as np
import os
import random
import zipfile

def generate_random_kernel(kernel_size, blur_parameter, center_parameter=1.0):
    """
    Generate a random kernel for convolution.
    """
    kernel = np.random.uniform(low=0, high=blur_parameter, size=(kernel_size, kernel_size))
    if center_parameter is not None:
        center = kernel_size // 2
        kernel[center, center] = center_parameter
    kernel /= np.sum(kernel)  # Normalize kernel
    return kernel

def create_class_filters(num_classes, kernel_size=3, blur_parameter=1.0, center_parameter=1.0):
    """
    Create random filters (kernels) for each class.
    """
    filters = {}
    for cls in range(num_classes):
        filters[cls] = generate_random_kernel(kernel_size, blur_parameter, center_parameter)
    return filters

def apply_class_filter(image, label, filters):
    """
    Apply the filter corresponding to the class label.
    """
    kernel = filters[label]
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image

# Example for custom dataset
def load_custom_dataset(root_folder):
    """
    Load images from a custom dataset folder structure.
    """
    images = []
    labels = []
    class_map = {cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(root_folder)))}
    
    for cls_name, cls_idx in class_map.items():
        cls_folder = os.path.join(root_folder, cls_name)
        for filename in os.listdir(cls_folder):
            file_path = os.path.join(cls_folder, filename)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Only load image files
                img = cv2.imread(file_path)
                if img is not None:
                    images.append(img)
                    labels.append(cls_idx)

    return images, labels, class_map


# Function untuk unzip dataset
def unzip_dataset(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# Main execution
if __name__ == "__main__":
    # Path ke ZIP dataset
    zip_path = "testingDataset.zip"
    dataset_path = "dataset"

    # Unzip dataset
    unzip_dataset(zip_path, dataset_path)

    # Load custom dataset
    images, labels, class_map = load_custom_dataset(dataset_path)
    print(f"Loaded {len(images)} images from {len(class_map)} classes.")

    # Create filters for each class
    num_classes = len(class_map)
    kernel_size = 3
    blur_parameter = 1.0
    center_parameter = 1.0

    filters = create_class_filters(num_classes, kernel_size, blur_parameter, center_parameter)
    print(f"Created filters for {num_classes} classes.")

    # Apply filters to images and save results
    output_path = "./filtered_dataset"
    os.makedirs(output_path, exist_ok=True)

    for i, img in enumerate(images):
        label = labels[i]
        class_name = list(class_map.keys())[label]
        filtered_img = apply_class_filter(img, label, filters)

        # Save the filtered image
        class_folder = os.path.join(output_path, class_name)
        os.makedirs(class_folder, exist_ok=True)
        output_file = os.path.join(class_folder, f"filtered_{i}.jpg")
        cv2.imwrite(output_file, filtered_img)

    print(f"Filtered images saved to {output_path}.")
