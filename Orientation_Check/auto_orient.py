import cv2
import numpy as np

def correct_and_show_white_text_plate(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert the image to make white text on a dark background
    inverted = cv2.bitwise_not(gray)

    # Apply adaptive thresholding to binarize the inverted image
    _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area (assuming the number plate is a large contour)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

    # Draw the contours on a blank image to obtain a mask
    mask = np.zeros_like(inverted)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Bitwise AND operation to extract the region of interest (ROI)
    roi = cv2.bitwise_and(inverted, inverted, mask=mask)

    # Apply edge detection using Canny on the ROI
    edges = cv2.Canny(roi, 50, 150, apertureSize=3)

    # Perform Hough Transform to detect lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=220)

    # Find the average angle to determine the rotation angle
    angles = [line[0][1] for line in lines]
    rotation_angle = np.mean(angles)

    # Rotate the image to correct the tilt
    rotated = cv2.warpAffine(image, cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), rotation_angle * 25 / np.pi, 1), (image.shape[1], image.shape[0]))

    # Display the original and corrected images
    cv2.imshow("Original Image", image)
    cv2.imshow("Corrected Image", rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = "Orientation_Check\plate3.jpg"
correct_and_show_white_text_plate(image_path)
