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
    laplacian_pyramid.append(gaussian_pyramid[-1])  # Last level is the same as the smallest image
    return laplacian_pyramid


def generate_gaussian_pyramid(image, levels):
    pyramid = [image]
    for _ in range(levels - 1):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid


def merge_pyramid(laplacian1, laplacian2, cut_threshold):
    merged_pyramid = []
    for i in range(cut_threshold):
        merged_pyramid.append(laplacian1[i])
    for i in range(cut_threshold,len(laplacian2)-1):
        merged_pyramid.append(laplacian2[i])
    return merged_pyramid


def reconstruct_image_from_laplacian(laplacian_pyramid):
    laplacian_pyramid.reverse()
    image = laplacian_pyramid[0]
    for i in range(1, len(laplacian_pyramid)):
        image = cv2.pyrUp(image, (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0]))
        image = cv2.add(image, laplacian_pyramid[i])
    return image


def build_hybrid_image(image1, image2):
    pyramid_levels = 13
    laplacian_im1 = generate_laplacian_pyramid(image1, pyramid_levels)
    laplacian_im2 = generate_laplacian_pyramid(image2, pyramid_levels)
    cut_threshold = 6
    laplacian_hybrid = merge_pyramid(laplacian_im1, laplacian_im2, cut_threshold)
    return reconstruct_image_from_laplacian(laplacian_hybrid)


def load_images():
    # Load the images
    image1_path =r"C:\Users\tohar\ImageProcessingExs\ex3\kingBB.jpg"
    image2_path = r"C:\Users\tohar\ImageProcessingExs\ex3\sara.jpg"
    image1 = cv2.imread(image1_path).astype(np.uint8)
    image2 = cv2.imread(image2_path).astype(np.uint8)
    heigth,width = image1.shape[:2]
    image2 = cv2.resize(image2,(width,heigth))
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    return image1, image2


if __name__ == "__main__":
    # Load the images
    image1, image2 = load_images()
    # Create the hybrid image
    result = build_hybrid_image(image1, image2)
    result = cv2.resize(result, (400,600))
    # Display the hybrid image
    cv2.imshow("Hybrid Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
