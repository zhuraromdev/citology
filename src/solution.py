import cv2
import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True)
def merge_components(label_ids, merged_components, img):
    # Merge the components using bitwise OR operation
    for i in prange(1, len(label_ids)):
        componentMask = (label_ids == i).astype("uint8") * 255
        for channel in range(3):
            merged_components[..., channel] = np.bitwise_or(
                merged_components[..., channel], componentMask)
    return merged_components


# Loading the image
img = cv2.imread('data/img.jpg')

# Preprocess the image by converting it to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Applying 7x7 Gaussian Blur
blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)

# Applying threshold
_, threshold = cv2.threshold(
    blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# Apply the connected component analysis function
_, label_ids, _, _ = cv2.connectedComponentsWithStats(
    threshold, connectivity=4, ltype=cv2.CV_32S)

# Initialize a new image to store the merged components
merged_components = np.zeros(img.shape, dtype="uint8")

# Merge the components
merged_components = merge_components(label_ids, merged_components, img)

# Display the merged components on the input image
output = cv2.addWeighted(merged_components, 0.7, img, 0.3, 0)

cv2.imshow("Merged Components", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
