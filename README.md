# Image Component Analysis

This repository contains three versions of image component analysis implemented in Python using OpenCV and Numba.

## Files

1. `solution.py`: This is the working version of the image component analysis code. It performs component merging using bitwise operations and displays the merged components on the input image.

2. `run_with_cuda.py`: This version utilizes GPU acceleration through CUDA for some operations. It copies images and arrays to the GPU for processing and then copies the results back to the host. Note that this version requires a CUDA-enabled environment to run properly.

3. `classic.py`: This is a single-threaded implementation of image component analysis. It performs component merging using bitwise operations and displays both individual components and the merged components on the input image.

## Description

1. `solution.py`:
   - The code loads an image and converts it to grayscale.
   - It applies a 7x7 Gaussian blur to the grayscale image.
   - Thresholding is applied using Otsu's method to obtain a binary image.
   - Connected component analysis is performed to label the connected components in the binary image.
   - A new image is initialized to store the merged components.
   - The `merge_components` function merges the components using bitwise OR operation.
   - The merged components are displayed on the input image using `addWeighted` function.
   - Press any key to close the image window.

2. `run_with_cuda.py`:
   - The code loads an image and converts it to grayscale.
   - The grayscale image is copied to the GPU using `cuda.to_device` function.
   - Gaussian blur is applied on the GPU using `cv2.cuda.GaussianBlur`.
   - The blurred image is copied back to the host.
   - Thresholding is applied on the host using Otsu's method to obtain a binary image.
   - The binary image is copied to the GPU using `cuda.to_device`.
   - Connected component analysis is performed on the GPU using `cv2.cuda.connectedComponentsWithStats`.
   - The results are copied back to the host.
   - A new image is initialized to store the merged components on the host.
   - The code loops through each component, creates a mask for the current component on the GPU, and copies it back to the host.
   - The current component is merged with the merged components image using bitwise OR operation on the host.
   - The merged components are displayed on the input image using `addWeighted` function on the GPU.
   - Press any key to close the image window.

3. `classic.py`:
   - The code loads an image and converts it to grayscale.
   - It applies a 7x7 Gaussian blur to the grayscale image.
   - Thresholding is applied using Otsu's method to obtain a binary image.
   - Connected component analysis is performed to label the connected components in the binary image.
   - A new image is initialized to store the merged components.
   - The code loops through each component, creates a mask for the current component, and merges it with the merged components image using bitwise OR operation.
   - Individual component figures are displayed for each component.
   - The merged components are displayed on the input image using `addWeighted` function.
   - Press any key to close the image windows.

Note: The `solution.py` file is the recommended working version for general use. The `run_with_cuda.py` version utilizes GPU acceleration but requires a CUDA-enabled environment to run properly. The `classic.py` version is a single-threaded solution without GPU acceleration.

Please ensure that you have the necessary dependencies installed before running the code.
