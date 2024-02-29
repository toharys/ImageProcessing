import cv2
import numpy as np


def load_images(low_res_path, high_res_path):
    """
    Load images from the given paths.

    Parameters:
        low_res_path (str): The file path of the low-resolution image.
        high_res_path (str): The file path of the high-resolution image.

    Returns:
        tuple: A tuple containing the low-resolution and high-resolution images, respectively.
    """
    try:
        # Load the images
        low_res = cv2.imread(low_res_path)
        high_res = cv2.imread(high_res_path)
        if low_res is None or high_res is None:
            raise FileNotFoundError("There is an image which could not be loaded")

        return low_res, high_res

    except Exception as e:
        print("Error loading images:", e)
        return None, None


def save_image(image, save_path=None):
    """
    Save the given image to the specified path.

    Parameters:
        image (numpy.ndarray): The image to be saved.
        save_path (str): The file path to save the image.
        window_name (str): The name of the window to display the image.

    Returns:
        None
    """
    try:
        # Save the image if a save path is provided
        if save_path is not None:
            cv2.imwrite(save_path, image)
            print(f"Image saved successfully to: {save_path}")

    except Exception as e:
        print("Error saving image using save_image:", e)


def detect_and_compute_sift(image):
    """
    Detect keypoints and compute descriptors using SIFT algorithm.

    Parameters:
        image (numpy.ndarray): The input image.

    Returns:
        tuple: A tuple containing the keypoints and descriptors, respectively.
    """
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray_img, None)

    return keypoints, descriptors


def save_keypoints_and_descriptors(image1, keypoints1, descriptors1, image2, keypoints2, descriptors2, save_path):
    """
    Save images with keypoints and descriptors drawn on them.

    Parameters:
        image1 (numpy.ndarray): The first image.
        keypoints1 (list): The keypoints detected in the first image.
        descriptors1 (numpy.ndarray): The descriptors computed for the keypoints in the first image.
        image2 (numpy.ndarray): The second image.
        keypoints2 (list): The keypoints detected in the second image.
        descriptors2 (numpy.ndarray): The descriptors computed for the keypoints in the second image.
        save_path (str): The directory path to save the images.

    Returns:
        None
    """
    # Draw keypoints on both images
    image_with_keypoints1 = cv2.drawKeypoints(image1, keypoints1, None)
    image_with_keypoints2 = cv2.drawKeypoints(image2, keypoints2, None)

    # Combine images horizontally
    combined_image = np.hstack((image_with_keypoints1, image_with_keypoints2))

    # Save the combined image of keypoints
    cv2.imwrite(save_path + "_combined_keypoints.jpg", combined_image)

    # Draw visual representations of descriptors
    image1_with_descriptors = image1.copy()
    for kp, desc in zip(keypoints1, descriptors1):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        color = tuple(map(int, desc[:3] * 255))  # Use the descriptor values as color
        cv2.circle(image1_with_descriptors, (x, y), 5, color, -1)

    image2_with_descriptors = image2.copy()
    for kp, desc in zip(keypoints2, descriptors2):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        color = tuple(map(int, desc[:3] * 255))  # Use the descriptor values as color
        cv2.circle(image2_with_descriptors, (x, y), 5, color, -1)

    # Save the images with descriptors
    cv2.imwrite(save_path + "_descriptors_low_res.jpg", image1_with_descriptors)
    cv2.imwrite(save_path + "_descriptors_high_res.jpg", image2_with_descriptors)


def match_descriptors(descriptors1, descriptors2):
    """
    Match descriptors using FLANN-based matcher.

    Parameters:
        descriptors1 (numpy.ndarray): Descriptors from the first image.
        descriptors2 (numpy.ndarray): Descriptors from the second image.

    Returns:
        list: A list containing the indices of matched keypoints.
    """
    # Create FLANN matcher object
    flann = cv2.FlannBasedMatcher()

    # Match descriptors using FLANN
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

    # Extract the indices of keypoints from the matches
    converted_matches = [(match.queryIdx, match.trainIdx) for match in good_matches]

    return converted_matches


def visualize_matches(image1, keypoints1, image2, keypoints2, matches):
    """
    Visualize matches between keypoints in two images.

    Parameters:
        image1 (numpy.ndarray): The first image.
        keypoints1 (list): Keypoints detected in the first image.
        image2 (numpy.ndarray): The second image.
        keypoints2 (list): Keypoints detected in the second image.
        matches (list): Indices of matched keypoints.

    Returns:
        numpy.ndarray: An image showing the matches between keypoints in the two input images.
    """
    # Create copies of the input images
    image1_copy = np.copy(image1)
    image2_copy = np.copy(image2)

    # Define colors for drawing circles around keypoints
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Blue, Red

    # Draw circles around the matched keypoints in both images with different colors
    for i, match in enumerate(matches):
        # Get the keypoints from the matches
        keypoint1 = keypoints1[match[0]].pt
        keypoint2 = keypoints2[match[1]].pt

        # Draw circles around the matched keypoints with different colors
        cv2.circle(image1_copy, (int(keypoint1[0]), int(keypoint1[1])), 5, colors[i % len(colors)], 2)
        cv2.circle(image2_copy, (int(keypoint2[0]), int(keypoint2[1])), 5, colors[i % len(colors)], 2)

    # Concatenate the two images horizontally
    image_matches = np.concatenate((image1_copy, image2_copy), axis=1)

    return image_matches


def estimate_homography_ransac(matches, keypoints_dst, keypoints_src, ransac_threshold=3):
    """
    Estimate the homography matrix using RANSAC algorithm.

    Parameters:
        matches (list): Indices of matched keypoints.
        keypoints_dst (list): Keypoints from the destination image.
        keypoints_src (list): Keypoints from the source image.
        ransac_threshold (float): Threshold value for RANSAC algorithm.

    Returns:
        numpy.ndarray: The estimated homography matrix.
    """
    # Extract matched keypoints
    src_pts = np.float32([keypoints_src[i].pt for _, i in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_dst[j].pt for j, _ in matches]).reshape(-1, 1, 2)

    # Find homography using RANSAC
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)

    # Refine homography using all inliers with Linear Regression (Least Squares)
    src_inliers = src_pts[mask.ravel() == 1]
    dst_inliers = dst_pts[mask.ravel() == 1]
    homography, _ = cv2.findHomography(src_inliers, dst_inliers, 0)
    print(homography)

    return homography


def warp_and_blend(low_res_img, high_res_img, homography):
    """
    Warp and blend the high-resolution image onto the low-resolution image.

    Parameters:
        low_res_img (numpy.ndarray): The low-resolution image.
        high_res_img (numpy.ndarray): The high-resolution image.
        homography (numpy.ndarray): The homography matrix.

    Returns:
        numpy.ndarray: The blended image.
    """
    # Warp the high-resolution image onto the coordinate system of the low-resolution image
    warped_img = cv2.warpPerspective(high_res_img, homography, (low_res_img.shape[1], low_res_img.shape[0]))

    # Mask for the region of interest in the high-resolution image (where transparency is present)
    mask = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    # Invert the mask
    mask_inv = cv2.bitwise_not(mask)

    # Blend the images
    blended_img = cv2.bitwise_and(low_res_img, low_res_img, mask=mask_inv)  # Low-resolution image outside region of interest
    blended_img += warped_img  # Warped high-resolution image within region of interest

    return blended_img

if __name__ == "__main__":
    paths = [[r"assets\desert_low_res.jpg", r"assets\desert_high_res.png"],
             [r"assets\lake_low_res.jpg", r"assets\lake_high_res.png"]]

    saving_descriptors_paths = [r"descriptors\desert", r"descriptors\lake"]

    saving_matches_paths = [r"matches\desert_matches.jpg", r"matches\lake_matches.jpg"]

    saving_blended_paths = [r"blended\desert_blended.jpg", r"blended\lake_blended.jpg"]

    for i in range(2):
        # Load the images
        low_res, high_res = load_images(paths[i][0], paths[i][1])

        # Build descriptors using SIFT
        keypoints_low, descriptors_low = detect_and_compute_sift(low_res)
        keypoints_high, descriptors_high = detect_and_compute_sift(high_res)

        # Save the descriptors
        save_keypoints_and_descriptors(low_res, keypoints_low, descriptors_low,
                                       high_res, keypoints_high, descriptors_high,
                                       saving_descriptors_paths[i])

        # Find matches points
        matches = match_descriptors(descriptors_low, descriptors_high)

        # Save the matches
        image_matches = visualize_matches(low_res, keypoints_low,
                                          high_res, keypoints_high,
                                          matches)
        save_image(image_matches, saving_matches_paths[i])

        # Estimate using RANSAC and Least Squres
        homography = estimate_homography_ransac(matches, keypoints_low, keypoints_high)

        # Wrap and Blend the images
        blended_image = warp_and_blend(low_res, high_res, homography)

        # Save the blended image
        save_image(blended_image, saving_blended_paths[i])

