import os
import matplotlib.pyplot as plt
import numpy as np
import cv2


# Function to plot histogram
def plot_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.title("Histogram")
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.show()


# Function to print histogram values
def unique_pixel_values(image):
    flattened_image = image.ravel()
    unique_values, counts = np.unique(flattened_image, return_counts=True)
    pixel_counts = dict(zip(unique_values, counts))
    return pixel_counts


# Function to fix sin wave pattern
def fix_sin_wave(image):
    f_transformed = np.fft.fft2(image)
    magnitude_spectrum = np.log(np.abs(f_transformed) + 1)

    dc_component = magnitude_spectrum[0][0]
    sorted_magnitude_spectrum = np.sort(magnitude_spectrum)[::-1]
    second_max = np.max(sorted_magnitude_spectrum[1])

    condition = (dc_component > magnitude_spectrum) & (
        magnitude_spectrum > (second_max * 1.05)
    )
    f_transformed[condition] = 0

    img_back_modified = np.abs(np.fft.ifft2(f_transformed))
    return img_back_modified


def fix_inverted(image, hist_values):
    if 0 in hist_values and 255 in hist_values:
        if hist_values[0] > hist_values[255]:
            image_new = 255 - image
            _, image_new = cv2.threshold(image_new, 192, 255, cv2.THRESH_BINARY)
            return image_new
        else:
            return image
    else:
        return image


def fix_low_brightness(image, hist_values):
    if max(hist_values.keys()) < 64:
        increased_brightness = cv2.add(image, 128)
        equalized_image = cv2.equalizeHist(increased_brightness)
        _, image_new = cv2.threshold(equalized_image, 92, 255, cv2.THRESH_BINARY)
        return image_new
    else:
        return image


def fix_high_brightness(image, hist_values):
    if min(hist_values.keys()) > 192:
        subtracted_image = np.clip(image - 128, 0, 255).astype(np.uint8)
        equalized_image = cv2.equalizeHist(subtracted_image)
        _, image_new = cv2.threshold(equalized_image, 127, 255, cv2.THRESH_BINARY)
        return image_new
    else:
        return image


def fix_low_contrast(image, hist_values):
    if 0 not in hist_values and 255 not in hist_values:
        sum_of_keys = sum(key for key in hist_values)
        keys_average = sum_of_keys / len(hist_values)
        equalized_image = cv2.equalizeHist(np.uint8(image))
        _, image_new = cv2.threshold(
            equalized_image, keys_average, 255, cv2.THRESH_BINARY
        )
        return image_new
    else:
        return image


def fix_salt_pepper(img):
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    kernel = np.ones((9, 9), np.uint8)
    denoised_image = cv2.medianBlur(img, ksize=15)
    denoised_image = cv2.morphologyEx(denoised_image, cv2.MORPH_CLOSE, kernel)
    kernel_size = (81, 81)
    denoised_image = cv2.GaussianBlur(denoised_image, kernel_size, sigmaX=0)
    _, denoised_image = cv2.threshold(
        denoised_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    return denoised_image


def detect_locator_boxes(image):
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Detect edges using Canny edge detection
    edges = cv2.Canny((blurred), 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(
        edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Initialize list to store bounding boxes of locator boxes
    locator_boxes = []

    # Iterate through the contours and find the bounding box of each locator box
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Adjust threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            locator_boxes.append((x, y, w, h))

    return locator_boxes


def expand_to_qr_code(image, locator_boxes):
    # Calculate the bounding box that encompasses all three locator patterns
    x_min = min(box[0] for box in locator_boxes)
    y_min = min(box[1] for box in locator_boxes)
    x_max = max(box[0] + box[2] for box in locator_boxes)
    y_max = max(box[1] + box[3] for box in locator_boxes)

    # Calculate the size of the expanded bounding box
    qr_code_width = x_max - x_min
    qr_code_height = y_max - y_min

    # Return the expanded bounding box
    return (x_min, y_min, qr_code_width, qr_code_height)


def crop_to_bounding_box(image, bounding_box):
    # Extract bounding box coordinates
    x, y, w, h = bounding_box

    # Crop the image to the bounding box
    cropped_image = image[y : y + h, x : x + w]

    return cropped_image


def crop_to_bounding_box_margin(image, bounding_box, margin=5):
    # Extract bounding box coordinates
    x, y, w, h = bounding_box

    # Add margin to the bounding box
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(image.shape[1], x + w + margin)
    y2 = min(image.shape[0], y + h + margin)

    # Crop the image to the adjusted bounding box
    cropped_image = image[y1:y2, x1:x2]

    return cropped_image


def fix_tilt(image):
    # Step 1: Edge Detection using Canny edge detector
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Step 2: Apply Hough Transform to detect lines
    lines = cv2.HoughLines(
        edges, rho=1, theta=np.pi / 180, threshold=64
    )  # Adjust threshold for smaller QR codes

    if lines is not None:
        # Calculate angles of detected lines and filter out lines with extreme angles
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta)
            angles.append(angle)

        # Filter out lines based on angle threshold (e.g., exclude lines with extreme angles)
        filtered_angles = [angle for angle in angles if abs(angle) < 90]

        if filtered_angles:
            # Compute the median angle of remaining lines
            rotation_angle = np.median(filtered_angles)

            # Check if rotation angle is significant
            if 5 < abs(90 - rotation_angle) < 45:  # Adjust threshold as needed
                rotation_angle = abs(90 - rotation_angle)
                if LOG:
                    print("Estimated Rotation Angle:", rotation_angle)
                # Rotate the image to make the QR code horizontal
                height, width = image.shape[:2]
                rotation_center = (width // 2, height // 2)  # Center of the image
                rotation_matrix = cv2.getRotationMatrix2D(
                    rotation_center, -rotation_angle, 1
                )

                # Apply rotation to the image (with white background)
                rotated_image = cv2.warpAffine(
                    image,
                    rotation_matrix,
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=255,
                )

                return rotated_image
            else:
                return image  # No significant rotation needed
        else:
            return image  # Return original image
    else:
        return image  # Return original image


# fixes skew on one locator box only??
def fix_locator_box_skew(image):
    # Apply edge detection to find contours
    edges = cv2.Canny(image, 50, 150, apertureSize=5)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assumed to be the QR code)
    max_contour = max(contours, key=cv2.contourArea)

    # Approximate the polygonal curve of the contour
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)

    # Check if the contour is a quadrilateral (4 vertices)
    if len(approx) == 4:
        # Reshape the vertices of the quadrilateral
        rect = np.zeros((4, 2), dtype=np.float32)
        for i in range(4):
            rect[i] = approx[i][0]

        # Calculate angles of the lines forming the sides of the locator box
        angles = []
        for i in range(4):
            p1 = rect[i]
            p2 = rect[(i + 1) % 4]
            p3 = rect[(i + 2) % 4]
            v1 = p1 - p2
            v2 = p3 - p2
            angle = np.degrees(
                np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            )
            angles.append(angle)

        # Check if any angle deviates significantly from 90 degrees
        if any(abs(angle - 90) > 10 for angle in angles):
            # Compute the target square shape
            target_shape = np.array(
                [[0, 0], [1012, 0], [1012, 1012], [0, 1012]], dtype=np.float32
            )

            # Compute the perspective transformation matrix
            matrix = cv2.getPerspectiveTransform(rect, target_shape)

            # Apply the perspective transformation to correct skew
            corrected_image = cv2.warpPerspective(image, matrix, (1012, 1012))
            return corrected_image

    # Return the original image if no skew is detected or contour is not a quadrilateral
    return image


def rotate_image(image):
    # dilate to isolate black squares
    custom_kernel_size = 105
    custom_kernel = np.ones((custom_kernel_size, custom_kernel_size), dtype=np.uint8)
    temp_img = cv2.dilate(image, custom_kernel, iterations=1)

    # Load the image in grayscale and apply thresholding to create a binary image
    _, binary = cv2.threshold(
        np.uint8(temp_img), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Perform connected component analysis (CCA) to label connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    # loop to check on orientation
    rotation_flags = [0, 0, 0, 0]
    for i in range(1, num_labels):  # Skip the background label (0)
        x, y, _, _, _ = stats[i]
        if x < image.shape[0] / 2 and y < image.shape[0] / 2:
            rotation_flags[0] = 1
        elif x > image.shape[0] / 2 and y < image.shape[0] / 2:
            rotation_flags[1] = 1
        elif x < image.shape[0] / 2 and y > image.shape[0] / 2:
            rotation_flags[2] = 1
        elif x > image.shape[0] / 2 and y > image.shape[0] / 2:
            rotation_flags[3] = 1

    # check orientation and do optimal rotation movements
    if rotation_flags == [1, 1, 1, 0]:
        rotated_image = image
    elif rotation_flags == [1, 0, 1, 1]:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_flags == [0, 1, 1, 1]:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_flags == [1, 1, 0, 1]:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        rotated_image = image

    return rotated_image


def repair_image(image):
    # Specify the desired size for the new image
    new_size = (924, 924)

    # Resize the original image to the new size using interpolation
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

    grid_cells_num = 21
    grid_cell_size = 44
    qr_cells = resized_image.reshape(
        (
            grid_cells_num,
            grid_cell_size,
            grid_cells_num,
            grid_cell_size,
            -1,  # Keep all color channels
        )
    ).swapaxes(1, 2)

    qr_cells_numeric = np.ndarray((grid_cells_num, grid_cells_num), dtype=np.uint8)
    for i, row in enumerate(qr_cells):
        for j, cell in enumerate(row):
            # Compute the median value for each color channel
            median_color = np.median(cell) / 255
            # Replace all pixel values in the cell with the median value
            qr_cells[i, j] = median_color

    # Reshape the modified cells back into the original image shape
    repaired_image = qr_cells.swapaxes(1, 2).reshape(new_size)

    return repaired_image

    # reed solomon?


def replace_with_median(image, size):
    # Define the size of each cell
    cell_size = size // 21

    # Create an empty array to store the result
    result = np.empty_like(image)

    # Iterate over the image in strides of cell_size
    for i in range(0, image.shape[0], cell_size):
        for j in range(0, image.shape[1], cell_size):
            # Extract the current cell
            cell = image[i : i + cell_size, j : j + cell_size]

            # Compute the median value of the cell
            median_value = np.median(cell)

            # Fill the entire cell with the median value
            result[i : i + cell_size, j : j + cell_size] = median_value

    return result


def perform_pipeline(folder_path, log=True, plot=True):
    global LOG
    LOG = log

    if plot:
        fig, axes = plt.subplots(3, 6, figsize=(8, 8))

    for i, filename in enumerate(os.listdir(folder_path)):
        if i >= 18:
            break

        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        image_new = np.uint8(fix_sin_wave(image))

        # Initial Thresholding and Histogram Equalization
        hist_values = unique_pixel_values(image_new)
        sum_of_keys = sum(key for key in hist_values)
        keys_average = sum_of_keys / len(hist_values)
        _, image_new = cv2.threshold(
            image_new, keys_average - 24, keys_average + 24, cv2.THRESH_BINARY
        )
        image_new = cv2.equalizeHist(np.uint8(image_new))

        # If new image is corrupted (all black or all white), keep original for now
        hist_values = unique_pixel_values(image_new)
        if len(hist_values) == 1:
            image_new = image

        # By observation, 99% of QR images thresholding will have more white pixels than black pixels
        # So, if black pixels are more than white pixels, invert image
        hist_values = unique_pixel_values(image_new)

        # Needs rework (Broken)
        image_new = fix_inverted(image_new, hist_values)

        # Checking for extreme low/high brightess

        hist_values = unique_pixel_values(image_new)

        # Low Brightness fix (can be more generic by removing small noise using median filter)
        image_new = fix_low_brightness(image_new, hist_values)

        # High Brightness fix (can be more generic by removing small noise using median filter)
        image_new = fix_high_brightness(image_new, hist_values)

        # High Brightness fix (can be more generic by removing small noise using median filter)
        image_new = fix_high_brightness(image_new, hist_values)

        # After all above preprocessing, if an image still doesn't have any black or white pixel values, it is most probably a low contrast image
        # Preprocessing with average of key values as threshold to fix low contrast images
        hist_values = unique_pixel_values(image_new)
        image_new = fix_low_contrast(image_new, hist_values)

        # After all previous preprocessing all QR codes quiet zones should be pure white pixels only
        # If the first 5x5 window in image still contains more than 1 pixel value, then it must contain salt & pepper noise
        small_corner = unique_pixel_values(image_new[0:5][0:5])
        if len(small_corner) > 1:
            image_new = fix_salt_pepper(image)

        # DANGER ZONE
        ################################################################

        # Detect all three locator boxes to detect QR code frame
        locator_boxes = detect_locator_boxes(image_new)

        if locator_boxes:
            # Expand to encompass the entire QR code
            expanded_box = expand_to_qr_code(image_new, locator_boxes)

            # This function simulates a zoom effect to be able to frame smaller qr codes more accurately
            image_new = crop_to_bounding_box_margin(image_new, expanded_box, margin=50)

            # This function fixes tilt in images using Hough Lines, it works only for tilt angles less than 45
            image_new = fix_tilt(image_new)

            # Detect locator boxes again after zooming
            locator_boxes = detect_locator_boxes(image_new)

            # Get QR frame coordinates
            expanded_box = expand_to_qr_code(image_new, locator_boxes)

            # Draw QR frame
            x, y, w, h = expanded_box
            # cv2.rectangle(image_new, (x, y), (x + w, y + h), (0, 0, 0), 3)

            # Crop image to locator frame size
            image_new = crop_to_bounding_box(image_new, expanded_box)

        # FIX 6 HERE
        # image_new = fix_locator_box_skew(image_new)

        # rotate images that has clear 3 locator boxes (to be moved down)
        image_new = rotate_image(image_new)

        # Resizing for decoding
        #########################################

        # if image_new.shape > (924, 924):
        #     image_new = cv2.resize(image_new, (924, 924), interpolation=cv2.INTER_LINEAR)
        #     image_new = replace_with_median(image_new, 924)
        # elif image_new.shape > (672, 672):
        #     image_new = cv2.resize(image_new, (672, 672), interpolation=cv2.INTER_LINEAR)
        #     image_new = replace_with_median(image_new, 672)
        # elif image_new.shape > (420, 420):
        #     image_new = cv2.resize(image_new, (420, 420), interpolation=cv2.INTER_LINEAR)
        #     image_new = replace_with_median(image_new, 420)
        # else:
        #     image_new = cv2.resize(image_new, (168, 168), interpolation=cv2.INTER_LINEAR)
        #     image_new = replace_with_median(image_new, 168)

        # Apply Final thresholding to remove any noise pixels
        _, image_new = cv2.threshold(image_new, 64, 255, cv2.THRESH_BINARY)

        # Plot the image in the corresponding subplot
        if plot:
            row = i // 6
            col = i % 6
            axes[row, col].axis("off")
            axes[row, col].imshow(image_new, cmap="gray")

        if log:
            hist_values = unique_pixel_values(image_new)
            print(f"Case {i+1}", hist_values, image_new.shape)

        # save each image
        cv2.imwrite(f"output_images/{filename}", image_new)

    if plot:
        plt.tight_layout()
        plt.show()


LOG = True