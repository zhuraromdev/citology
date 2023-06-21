import cv2
import numpy as np
from numba import cuda

# Loading the image
img = cv2.imread('data/img.jpg')

# preprocess the image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Copy the grayscale image to the GPU
d_gray_img = cuda.to_device(gray_img)

# Applying 7x7 Gaussian Blur on GPU
d_blurred = cuda.device_array_like(d_gray_img)
cv2.cuda.GaussianBlur(d_gray_img, (7, 7), d_blurred)

# Copy the result back to the host
blurred = d_blurred.copy_to_host()

# Applying threshold on CPU
_, threshold = cv2.threshold(
    blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# Copy the threshold image to the GPU
d_threshold = cuda.to_device(threshold)

# Apply the Component analysis function on GPU
d_label_ids = cuda.device_array_like(d_threshold)
d_values = cuda.device_array_like(d_threshold)
d_centroid = cuda.device_array_like(d_threshold)
totalLabels = cv2.cuda.connectedComponentsWithStats(
    d_threshold, d_label_ids, d_values, d_centroid, 4, cv2.CV_32S)

# Copy the results back to the host
label_ids = d_label_ids.copy_to_host()
values = d_values.copy_to_host()
centroid = d_centroid.copy_to_host()

# Initialize a new image to store the merged components on CPU
merged_components = np.zeros(img.shape, dtype="uint8")

# Loop through each component
for i in range(1, totalLabels):
    # Create a mask for the current component on GPU
    d_componentMask = (label_ids == i).astype("uint8") * 255

    # Copy the mask back to the host
    componentMask = d_componentMask.copy_to_host()

    # Merge the current component with the merged components image on CPU
    merged_components = cv2.bitwise_or(merged_components, cv2.merge(
        [componentMask, componentMask, componentMask]))

# Copy the merged components image to the GPU
d_merged_components = cuda.to_device(merged_components)

# Display the merged components on the input image
d_output = cuda.device_array_like(d_merged_components)
cv2.cuda.addWeighted(d_merged_components, 0.7,
                     cuda.to_device(img), 0.3, 0, d_output)

# Copy the result back to the host
output = d_output.copy_to_host()

cv2.imshow("Merged Components", output)
cv2.waitKey(0)
cv2.destroyAllWindows()


# OR
# import cv2
# import numpy as np
# import multiprocessing

# def process_component(threshold, label_ids, index):
#     _, label_map, _, _ = cv2.connectedComponentsWithStats(
#         threshold, connectivity=4, ltype=cv2.CV_32S)
#     label_ids[index] = label_map.tobytes()

# # Loading the image
# img = cv2.imread('data/img.jpg')

# # Preprocess the image by converting it to grayscale
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Applying 7x7 Gaussian Blur
# blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)

# # Applying threshold
# _, threshold = cv2.threshold(
#     blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# # Split the threshold image into multiple segments
# num_segments = multiprocessing.cpu_count()
# segment_height = threshold.shape[0] // num_segments
# segments = [threshold[i * segment_height : (i + 1) * segment_height, :]
#             for i in range(num_segments)]

# # Initialize an array to store the label maps
# label_ids = multiprocessing.Array('b', num_segments * threshold.size)

# # Create a pool of processes
# pool = multiprocessing.Pool()

# # Process each segment in parallel
# for i in range(num_segments):
#     pool.apply_async(process_component, args=(segments[i], label_ids, i))

# # Wait for all processes to finish
# pool.close()
# pool.join()

# # Concatenate the label maps from all segments
# label_map = np.concatenate([np.frombuffer(label_ids, dtype=np.int32)
#                             for label_ids in label_ids])

# # Merge the components
# merged_components = np.zeros(img.shape, dtype="uint8")
# for i in range(1, np.max(label_map) + 1):
#     componentMask = (label_map == i).astype("uint8") * 255
#     merged_components = cv2.bitwise_or(merged_components, cv2.merge([componentMask, componentMask, componentMask]))

# # Display the merged components on the input image
# output = cv2.addWeighted(merged_components, 0.7, img, 0.3, 0)

# cv2.imshow("Merged Components", output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
