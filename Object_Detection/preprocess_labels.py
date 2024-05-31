import pandas as pd
import cv2

def transform_coordinates(first_model_coords, second_model_coords, original_image_size, number_plate_size):
    # Extract coordinates from the first model
    cls1, x1, y1, w1, h1, conf1 = first_model_coords
    original_image_width, original_image_height = original_image_size

    # Transform coordinates from the first model to absolute coordinates in the original image
    absolute_x1 = int(x1 * original_image_width)
    absolute_y1 = int(y1 * original_image_height)
    absolute_w1 = int(w1 * original_image_width)
    absolute_h1 = int(h1 * original_image_height)

    # Extract coordinates from the second model
    cls2, x2, y2, w2, h2, conf2 = second_model_coords
    number_plate_width, number_plate_height = number_plate_size

    # Transform coordinates from the second model to absolute coordinates in the number plate image
    absolute_x2 = int(x2 * number_plate_width)
    absolute_y2 = int(y2 * number_plate_height)
    absolute_w2 = int(w2 * number_plate_width)
    absolute_h2 = int(h2 * number_plate_height)

    # Convert second model coordinates to the format of the first model
    transformed_x = absolute_x2 + absolute_x1
    transformed_y = absolute_y2 + absolute_y1
    transformed_w = absolute_w2 * absolute_w1
    transformed_h = absolute_h2 * absolute_h1

    return transformed_x, transformed_y, transformed_w, transformed_h

# Example usage:
img_path = "E:/Devanagari Number Plate Recognition/Project_All/Project_Host/Streamlit_App/img/uploaded/img.jpg"
num_path = "E:/Devanagari Number Plate Recognition/Project_All/Project_Host/Streamlit_App/inference/numberplate/box_0.jpg"
img = cv2.imread(img_path)
img_height, img_width, _ = img.shape
number_plate = cv2.imread(num_path)
num_height, num_width, _ = number_plate.shape

original_image_size = (img_height, img_width)  # Replace with actual size of the original car image
number_plate_size = (num_height, num_width)  # Replace with actual size of the extracted number plate image
first_model_coords = []
second_model_coords = []

img_df = pd.read_csv("E:/Devanagari Number Plate Recognition/Project_All/Project_Host/Streamlit_App/inference/extracted_numberplate/plate1.txt", sep=" ", header=None)
rows, columns = img_df.shape
for i in range(rows):
    for j in range(columns):
        first_model_coords.append(img_df[j][i]) 
        
print(first_model_coords)

# transformed_coords = transform_coordinates(first_model_coords, second_model_coords, original_image_size, number_plate_size)
# print("Transformed Coordinates:", transformed_coords)
