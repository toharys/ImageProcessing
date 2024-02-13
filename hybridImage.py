import sys
import cv2
import numpy as np


def generate_laplacian_pyramid(image, levels):
    gaussian_pyramid = generate_gaussian_pyramid(image, levels)
    laplacian_pyramid = []
    for i in range(levels-1):
        expanded_image = cv2.pyrUp(gaussian_pyramid[i+1], dstsize=(gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
        laplacian = cv2.subtract(gaussian_pyramid[i], expanded_image)
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1]) # Last level is the same as the smallest image
    return laplacian_pyramid


def generate_gaussian_pyramid(image, levels):
    pyramid = [image]
    for _ in range(levels - 1):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid


# def generate_laplacian_pyramid(image, levels):
#     pyramid = [image]
#     for _ in range(levels-1):
#         image = cv2.pyrDown(image)
#         expanded_image = cv2.pyrUp(image, dstsize=(pyramid[0].shape[1], pyramid[0].shape[0]))
#         laplacian = cv2.subtract(image, expanded_image)
#         pyramid.append(laplacian)
#     return pyramid

def combine_pyramid_levels(pyramid1, pyramid2):
    combined_pyramid = []
    for level1, level2 in zip(pyramid1, pyramid2):
        combined_level = 0.7 * level1 + 0.3 * level2  # Adjust the weights as needed
        combined_pyramid.append(combined_level)
    return combined_pyramid

def reconstruct_image(pyramid):
    reconstructed_image = pyramid[0]
    for i in range(1, len(pyramid)):
        reconstructed_image = cv2.pyrUp(reconstructed_image)
        # Resize the current level of the Laplacian pyramid to match the dimensions of the reconstructed image
        pyramid_level_resized = cv2.resize(pyramid[i], (reconstructed_image.shape[1], reconstructed_image.shape[0]))
        # Add the resized level of the Laplacian pyramid to the reconstructed image
        reconstructed_image = reconstructed_image.astype(np.float32)
        reconstructed_image += pyramid_level_resized

    return np.clip(reconstructed_image, 0, 255).astype(np.uint8)

# Load the images
image1 = cv2.imread(r"C:\Users\tohar\ImageProcessingExs\ex3\sky1.jpg")
image2 = cv2.imread(r"C:\Users\tohar\ImageProcessingExs\ex3\blackRat.jpg")

# Set the number of pyramid levels
levels = 5

# Generate Laplacian pyramids for both images
pyramid1 = generate_laplacian_pyramid(image1, levels)
pyramid2 = generate_laplacian_pyramid(image2, levels)

# Combine pyramid levels
combined_pyramid = combine_pyramid_levels(pyramid1, pyramid2)

# Reconstruct the hybrid image
hybrid_image = reconstruct_image(combined_pyramid)

# Display the hybrid image
cv2.imshow("Hybrid Image", hybrid_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


