import cv2
import numpy as np

# Loading the image
img = cv2.imread('data/img2.jpg')

# preprocess the image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Applying 7x7 Gaussian Blur
blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)

# Applying threshold
threshold = cv2.threshold(
    blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# Apply the Component analysis function
analysis = cv2.connectedComponentsWithStats(threshold, 4, cv2.CV_32S)
(totalLabels, label_ids, values, centroid) = analysis

# Initialize a new image to store the merged components
merged_components = np.zeros(img.shape, dtype="uint8")

# Loop through each component
for i in range(1, totalLabels):
    # Create a new image for the component figure
    component_figure = np.zeros(img.shape, dtype="uint8")
    componentMask = (label_ids == i).astype("uint8") * 255

    # Apply the mask using the bitwise operator
    component_figure = cv2.bitwise_and(img, cv2.merge(
        [componentMask, componentMask, componentMask]))

    # Merge the current component with the merged components image
    merged_components = cv2.bitwise_or(merged_components, cv2.merge(
        [componentMask, componentMask, componentMask]))

    # Display the component figure
    cv2.imshow(f"Component {i}", component_figure)

# Display the merged components on the input image
output = cv2.addWeighted(img, 0.7, merged_components, 0.3, 0)

cv2.imshow("Merged Components", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
