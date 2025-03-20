import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, exposure, morphology
from PIL import Image, ImageDraw
from scipy.spatial.distance import cdist

# List to store clicked points
polygon_points = []


def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse click
        polygon_points.append((x, y))
        print(f"Point selected: {x}, {y}")


def analyze_fibers(image_path, scale_bar_length=500, kernel_size=51, min_pixels_in_actual_fiber_contour=50, roi_hor_start=0.01, roi_hor_end=0.99):
    """
    Analyzes fibers in a microscopic image to determine their diameters and orientations.

    The function processes an image by:
    - Extracting the scale bar and computing the pixel-to-micrometer ratio.
    - Defining a region of interest (ROI) for fiber analysis.
    - Enhancing contrast and reducing noise.
    - Detecting fiber edges using edge detection and morphological operations.
    - Finding and visualizing fiber contours.
    - Fitting straight lines to fiber contours and computing slopes.
    - Estimating fiber diameters and separations.

    Parameters:
    -----------
    image_path : str
        Path to the microscopic image file.
    scale_bar_length : int, optional (default=500)
        Length of the scale bar in micrometers for pixel-to-micrometer conversion.
    roi_vert_start : float, optional (default=0)
        Fraction (0-1) indicating the vertical start position of the ROI.
    roi_vert_end : float, optional (default=0.9)
        Fraction (0-1) indicating the vertical end position of the ROI.
    roi_hor_start : float, optional (default=0.15)
        Fraction (0-1) indicating the horizontal start position of the ROI.
    roi_hor_end : float, optional (default=0.85)
        Fraction (0-1) indicating the horizontal end position of the ROI.

    Returns:
    --------
    None
        The function displays several processed images and prints fiber measurement statistics.
    """

    # Load the image in grayscale mode
    image = io.imread(image_path, as_gray=True)

    # Display the smoothened image
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap="gray")
    plt.title("Smoothened Image")
    plt.axis("off")
    plt.show(block=False)
    plt.pause(0.1)

    # Convert grayscale image to BGR for colored overlays
    image_bgr = cv2.cvtColor((image / np.max(image) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Display the smoothened image
    plt.figure(figsize=(6, 6))
    plt.imshow(image_bgr, cmap="gray")
    plt.title("Smoothened Image")
    plt.axis("off")
    plt.show(block=False)
    plt.pause(0.1)

    height, width = image.shape
    black_region = image[int(0.92 * height):, :]  # Bottom 8% of the image
    binary_black_region = black_region > filters.threshold_otsu(black_region)

    # Find contours in the black region
    contours, _ = cv2.findContours(binary_black_region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Identify the white horizontal bar
    scale_bar_contour = max(contours, key=cv2.contourArea)  # Assume the largest contour is the scale bar
    x, y, w, h = cv2.boundingRect(scale_bar_contour)

    # Compute pixel-to-micrometer ratio
    pixels_per_micrometer = w / scale_bar_length  # Using the user-defined scale bar length in micrometer
    print(f"Pixel-to-micrometer ratio: {pixels_per_micrometer} pixels per micrometer")

    # # Define the region of interest (ROI) for fiber analysis
    # roi = image[int(roi_vert_start * height):int(roi_vert_end * height), int(roi_hor_start * width):int(roi_hor_end * width)]

    # -----------------------------------------------------------------------------

    # Show image and let user click points
    cv2.namedWindow("Select Polygon ROI", cv2.WINDOW_NORMAL)
    cv2.imshow("Select Polygon ROI (Click to add points, press any key to finish)", image)
    cv2.setMouseCallback("Select Polygon ROI (Click to add points, press any key to finish)", select_points)
    cv2.waitKey(0)  # Wait for any key press
    cv2.destroyAllWindows()

    # Convert points to numpy array
    polygon_array = np.array(polygon_points, np.int32).reshape((-1, 1, 2))
    print("Final Polygon Points:", polygon_array)

    # Create a black mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Draw a white-filled polygon on the mask
    cv2.fillPoly(mask, [polygon_array], 255)

    # # Apply the mask
    # roi = cv2.bitwise_and(image, image, mask=mask)
    #
    #
    # # Display the masked ROI
    # plt.figure(figsize=(6, 6))
    # plt.imshow(roi, cmap="gray")
    # plt.title("Region of Interest")
    # plt.axis("off")
    # plt.show(block=False)
    # plt.pause(0.1)
    # ------------------------------------------------------------------------------------------
    image_copy = image.copy()  # Copy original

    # Enhance contrast using adaptive histogram equalization
    image_copy = exposure.equalize_adapthist(image_copy)

    # Further enhance contrast using histogram stretching
    image_copy = exposure.rescale_intensity(image_copy, in_range=(np.percentile(image_copy, 10), np.percentile(image_copy, 90)))

    # Apply Gaussian blur with a larger kernel to reduce noise and smooth the image
    blurred = cv2.GaussianBlur(image_copy, (kernel_size, kernel_size), 0)

    # Display the smoothened image
    plt.figure(figsize=(6, 6))
    plt.imshow(blurred, cmap="gray")
    plt.title("Smoothened Image")
    plt.axis("off")
    plt.show(block=False)
    plt.pause(0.1)

    # Apply Sobel edge detection to highlight fiber borders
    edges = filters.sobel(blurred)
    edges[mask == 0] = 0

    # Determine an optimal threshold using Otsu's method and binarize the image
    thresh = filters.threshold_otsu(edges)
    binary = edges > thresh

    # Perform morphological operations to clean up the detected edges
    binary = morphology.binary_closing(binary, morphology.disk(2))  # Close small gaps
    binary = morphology.binary_opening(binary, morphology.disk(2))  # Remove small noise

    # Apply skeletonization to reduce fiber edges to single-pixel width
    binary = morphology.skeletonize(binary)

    # Convert grayscale image to RGB format for visualization
    # image_pil = Image.fromarray((image * 255).astype(np.uint8)).convert("RGB")
    # draw = ImageDraw.Draw(image_pil)

    # Find contours of the detected fiber edges
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # Offset contours to match full image coordinates
    # x_offset = int(roi_hor_start * width)  # Adjust for the cropped region
    # y_offset = int(roi_vert_start * height)

    # Compute the bounding box of the polygon
    x_offset, y_offset, _, _ = cv2.boundingRect(mask)  # Extract only x and y

    # # Get the exact leftmost (min x) and topmost (min y) points from the polygon
    # x_offset = np.min(polygon_array[:, 0, 0])  # Smallest x-coordinate
    # y_offset = np.min(polygon_array[:, 0, 1])  # Smallest y-coordinate
    print(x_offset, y_offset)

    # Draw cyan lines along the detected fiber edges
    for contour in contours:
        # contour = contour + [x_offset, y_offset]  # Adjust contour position
        # contour = [tuple(pt[0]) for pt in contour]  # Convert contour points to tuples
        # draw.line(contour, fill=(0, 255, 255), width=1)  # Draw lines in cyan
        cv2.drawContours(image_bgr, [contour], -1, (255, 255, 0), 1)  # Cyan color (BGR: 255, 255, 0)

    print("Number of pixels and slopes of fitted lines:")
    valid_slopes = []  # Store slopes of contours with more than 150 pixels

    # Initial and final deviation thresholds
    initial_threshold = 20  # Start with a large value
    final_threshold = 2  # Minimum threshold after iterations
    decay_factor = 0.8  # Reduce threshold by 20% per iteration
    max_iterations = 10  # Maximum number of refinement steps

    filtered_contours = []  # Store final valid fiber contours

    # Fit and draw red lines over detected fiber edges
    # actual_fiber_contours = []
    for contour in contours:
        if len(contour) >= min_pixels_in_actual_fiber_contour:  # Ensure there are enough points to fit a line

            # ------------------------------------

            # # Fit a straight line to the contour
            # [vx, vy, x0, y0] = cv2.fitLine(contour_array, cv2.DIST_L2, 0, 0.01, 0.01)
            # # This is just for plotting the red lines
            # # Get bounding box of the contour.
            # x_min = np.min(contour_array[:, 0, 0])
            # x_max = np.max(contour_array[:, 0, 0])
            # # Compute the corresponding y-values for the fitted line within the bounding box
            # y_min = int(y0 + ((x_min - x0) * vy / vx))
            # y_max = int(y0 + ((x_max - x0) * vy / vx))
            # # Print the number of pixels in the contour and the slope
            # num_pixels = len(contour)
            # slope = vy / vx if abs(vx) > 1e-6 else float('inf')  # Avoid division by zero
            # print(f"Contour with {num_pixels} pixels, Slope: {slope}")
            # # if num_pixels > 150:
            # valid_slopes.append(slope)
            # actual_fiber_contours.append(contour)
            # # Draw the fitted red line within the detected segment
            # # draw.line([(x_min + x_offset, y_min + y_offset), (x_max + x_offset, y_max + y_offset)], fill=(255, 0, 0), width=2)
            # # cv2.line(image_bgr, (int(x_min + x_offset), int(y_min + y_offset)), (int(x_max + x_offset), int(y_max + y_offset)), (0, 0, 255), 2)  # Red color (BGR: 0, 0, 255)
            # cv2.line(image_bgr, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)  # Red color (BGR: 0, 0, 255)
            # ----------------------------------------
            contour_array = np.array(contour, dtype=np.float32)
            iteration = 0
            deviation_threshold = initial_threshold

            while deviation_threshold >= final_threshold and iteration < max_iterations:
                # Fit a straight line to the contour
                [vx, vy, x0, y0] = cv2.fitLine(contour_array, cv2.DIST_L2, 0, 0.01, 0.01)

                # Convert contour points to (x, y) format
                x_points = contour_array[:, 0, 0]
                y_points = contour_array[:, 0, 1]

                # Compute perpendicular distance of each point to the fitted line
                distances = np.abs(vy * (x_points - x0) - vx * (y_points - y0)) / np.sqrt(vx**2 + vy**2)

                # # Compute the expected y-values along the fitted line
                # y_fitted = y0 + ((x_points - x0) * (vy / vx))
                #
                # # Compute deviation (absolute distance from fitted line)
                # deviation = np.abs(y_points - y_fitted)

                # Keep only points that are close to the fitted line
                valid_points = distances < deviation_threshold
                cleaned_contour = contour_array[valid_points]

                # If the contour has too few points, stop refining
                if len(cleaned_contour) < min_pixels_in_actual_fiber_contour:
                    break

                # Update contour and decrease threshold
                contour_array = cleaned_contour
                deviation_threshold *= decay_factor  # Reduce threshold
                iteration += 1  # Increment iteration count

            # Store the final cleaned contour
            if len(contour_array) >= min_pixels_in_actual_fiber_contour:
                filtered_contours.append(contour_array)

    # Use only `filtered_contours` for further calculations
    actual_fiber_contours = filtered_contours

    # Define the number of longest contours to keep
    top_contours_by_len = 10  # Store as a variable for easy adjustments

    # Sort contours by length (descending order)
    actual_fiber_contours = sorted(actual_fiber_contours, key=len, reverse=True)

    # Keep only the top `top_contours_by_len` longest contours
    actual_fiber_contours = actual_fiber_contours[:top_contours_by_len]

    # Iterate through the final cleaned fiber contours
    for contour in actual_fiber_contours:  # We already filtered small contours
        contour_array = np.array(contour, dtype=np.float32)

        # Fit a straight line to the contour
        [vx, vy, x0, y0] = cv2.fitLine(contour_array, cv2.DIST_L2, 0, 0.01, 0.01)

        # Compute the bounding box of the contour for line endpoints
        x_min = np.min(contour_array[:, 0, 0])
        x_max = np.max(contour_array[:, 0, 0])

        # Compute corresponding y-values for the fitted line
        y_min = int(y0 + ((x_min - x0) * vy / vx))
        y_max = int(y0 + ((x_max - x0) * vy / vx))
        num_pixels = len(contour)
        slope = vy / vx if abs(vx) > 1e-6 else float('inf')  # Avoid division by zero
        angle = float(np.nan_to_num(np.degrees(np.arctan(slope)), nan=0.0))
        print(f"Contour with {num_pixels} pixels, Slope: {slope}, angle: {angle}")
        # if num_pixels > 150:
        valid_slopes.append(slope)
        # Draw the fitted red line directly on image_bgr
        cv2.line(image_bgr,
                 (int(x_min), int(y_min)),
                 (int(x_max), int(y_max)),
                 (0, 0, 255), 2)  # Red color (BGR: 0, 0, 255)


    # Compute and print the average slope of valid contours
    fiber_diameters = []
    fiber_separations = []
    if valid_slopes:
        avg_slope = sum(valid_slopes) / len(valid_slopes)
        avg_angle = float(np.nan_to_num(np.degrees(np.arctan(avg_slope)), nan=0.0))
        print(f"Average angle: {avg_angle} deg")
        # # Compute the bounding box of the mask
        # x_mask, y_mask, w_mask, h_mask = cv2.boundingRect(mask)
        # # Compute the center of the mask
        # mask_center = (x_mask + w_mask // 2, y_mask + h_mask // 2)
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, (90-avg_angle), 1.0)
        rotated_contours = [cv2.transform(np.array(contour, dtype=np.float32), rotation_matrix).astype(int) for contour in actual_fiber_contours]
        rotated_image = Image.new("RGB", (width, height), (0, 0, 0))
        rotated_draw = ImageDraw.Draw(rotated_image)

        y_positions = range(0, height, 15)
        for y_pos in y_positions:
            x_positions = []
            for contour in rotated_contours:
                x_vals = contour[:, 0, 0]
                y_vals = contour[:, 0, 1]
                if np.min(y_vals) <= y_pos <= np.max(y_vals):
                    x_at_y = x_vals[np.argmin(np.abs(y_vals - y_pos))]
                    x_positions.append(int(x_at_y))
            if len(x_positions) > 1:
                x_positions.sort()
                for i in range(len(x_positions) - 1):
                    rotated_draw.line([(x_positions[i], y_pos), (x_positions[i + 1], y_pos)], fill=(255, 255, 0), width=1)
                    length_micrometers = (x_positions[i + 1] - x_positions[i]) / pixels_per_micrometer
                    # print(f"Horizontal line at y={y_pos}: Length = {length_micrometers:.2f} micrometers")
                    if 300 < length_micrometers < 500:
                        fiber_diameters.append(length_micrometers)
                    elif 100 < length_micrometers < 300:
                        fiber_separations.append(length_micrometers)

        # Draw rotated contours
        for contour in rotated_contours:
            rotated_draw.line([tuple(pt[0]) for pt in contour], fill=(0, 255, 0), width=1)  # Draw green lines

        # Restore previous plotting and display logic
        plt.figure(figsize=(6, 6))
        plt.hist(fiber_diameters, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel("Fiber Diameter (µm)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Fiber Diameters")
        plt.show(block=False)
        plt.pause(0.1)

        print(f"Average Fiber Diameter: {sum(fiber_diameters) / len(fiber_diameters):.2f} µm")

        plt.figure(figsize=(6, 6))
        plt.imshow(rotated_image)
        plt.title("Rotated Contours to Align Vertically")
        plt.axis("off")
        plt.show(block=False)
        plt.pause(0.1)

    else:
        print("No contours with more than 150 pixels found.")

    # Display the final image with detected fiber borders and fitted lines
    plt.figure(figsize=(6, 6))
    # plt.imshow(image_pil)
    # plt.title("Original Image with Fiber Borders (Cyan) and Fitted Lines (Red)")
    plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct display
    plt.title("Original Image with Fiber Borders (Cyan) and Fitted Lines (Red)")
    plt.axis("off")
    plt.show()


    return


# Example usage
# image_path = "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/Diego SEM/Diego_test_fast.tif"

image_path = "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/Diego SEM/90deg_400um_45perc/botom_edge_dimens.bmp"
analyze_fibers(image_path)
