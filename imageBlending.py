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


def reconstruct_blended_image(blended_pyramid):
    blended_image = blended_pyramid[0]
    for i in range(1, len(blended_pyramid)):
        blended_image = cv2.pyrUp(blended_image)
        blended_image += blended_pyramid[i]
    return np.clip(blended_image, 0, 255).astype(np.uint8)


def blend_images_laplacian(image1, image2, mask):
    # Convert images to float32 for better precision in calculations
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    # Generate Laplacian pyramids for images A and B
    levels = min(cv2.__version__.startswith('4'), 6)  # Choose appropriate level based on OpenCV version
    laplacian_a = generate_laplacian_pyramid(image1, levels)
    laplacian_b = generate_laplacian_pyramid(image2, levels)

    # Resize the mask to match the dimensions of the Laplacian pyramids
    gaussian_m = generate_gaussian_pyramid(mask,levels)
    for level in range(levels):
        gaussian_m[level] = cv2.resize(mask, (laplacian_a[level].shape[1], laplacian_a[level].shape[0]))
        gaussian_m[level] = gaussian_m[level] / 255 # in order to keep the binary mask vlues 0 and 1 after the expand
        # Expand the dimensions of the mask to match the number of channels in the Laplacian pyramids
        gaussian_m[level] = np.expand_dims(gaussian_m[level], axis=2)

    # Blend Laplacian levels according to the mask
    blended_pyramid = []
    for la, lb, gm in zip(laplacian_a, laplacian_b, gaussian_m[::-1]):
        # Convert the binary mask to an RGB mask
        blended_level = gm * la + (1 - gm) * lb
        blended_pyramid.append(blended_level)

    # Reconstruct the blended image by summing up the blended Laplacian levels
    return reconstruct_blended_image(blended_pyramid)

def load_images():
    # Load the images and the mask
    image1_path =r"chair.jpg"
    image2_path = r"sky.jpg"
    mask_path =r"mask.png"
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # Resize the mask and image2 to match the images if needed
    height, width = image1.shape[:2]
    image2 = cv2.resize(image2, (width, height))
    mask = cv2.resize(mask, (width,height))
    cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    return image1, image2, mask


if __name__ == "__main__":
    # Load the images
    image1, image2, mask = load_images()
    # Blend the images using Laplacian and Gaussian pyramids
    result = blend_images_laplacian(image1, image2, mask)
    # Display the result
    cv2.imshow("Blended Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()