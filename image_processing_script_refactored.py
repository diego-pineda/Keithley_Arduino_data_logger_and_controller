
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, exposure, morphology
from PIL import Image, ImageDraw
import tkinter as tk
from skan import Skeleton
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize



# Get screen size for display scaling
def get_screen_resolution():
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    return screen_width, screen_height


# Global storage for clicked polygon points
polygon_points = []


def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        scale = param["scale"]
        orig_x = int(x / scale)
        orig_y = int(y / scale)
        polygon_points.append((orig_x, orig_y))
        print(f"Point selected: {orig_x}, {orig_y}")


def get_pixel_to_micron_ratio(image, scale_bar_length):
    height = image.shape[0]
    black_region = image[int(0.92 * height):, :]
    binary = black_region > filters.threshold_otsu(black_region)
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return w / scale_bar_length


def get_polygon_roi(image):
    global polygon_points
    polygon_points = []

    screen_width, screen_height = get_screen_resolution()
    scale = min(screen_width / image.shape[1], screen_height / image.shape[0])
    image_display = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)

    cv2.namedWindow("Select Polygon ROI", cv2.WINDOW_NORMAL)
    cv2.imshow("Select Polygon ROI (Click to add points, press any key to finish)", image_display)
    cv2.setMouseCallback("Select Polygon ROI (Click to add points, press any key to finish)", select_points, param={"scale": scale})
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    polygon_array = np.array(polygon_points, np.int32).reshape((-1, 1, 2))
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_array], 255)
    return mask, polygon_array


def preprocess_image(image, mask, kernel_size):
    image = exposure.equalize_adapthist(image)
    image = exposure.rescale_intensity(image, in_range=(np.percentile(image, 10), np.percentile(image, 90)))
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    edges = filters.sobel(blurred)
    edges[mask == 0] = 0
    binary = edges > filters.threshold_otsu(edges)
    binary = morphology.binary_closing(binary, morphology.disk(2))
    binary = morphology.binary_opening(binary, morphology.disk(2))
    return morphology.skeletonize(binary)

# ------------------------deprecated --------------------


def filter_and_fit_contours(contours, min_pixels, image_bgr):
    filtered = []
    slopes = []
    for contour in contours:
        if len(contour) >= min_pixels:
            contour_array = np.array(contour, dtype=np.float32)
            deviation_threshold, final_threshold, decay_factor = 20, 2, 0.8
            for _ in range(10):
                [vx, vy, x0, y0] = cv2.fitLine(contour_array, cv2.DIST_L2, 0, 0.01, 0.01)
                x_points = contour_array[:, 0, 0]
                y_points = contour_array[:, 0, 1]
                distances = np.abs(vy * (x_points - x0) - vx * (y_points - y0)) / np.sqrt(vx**2 + vy**2)
                valid = distances < deviation_threshold
                cleaned = contour_array[valid]
                if len(cleaned) < min_pixels:
                    break
                contour_array = cleaned
                deviation_threshold *= decay_factor
            if len(contour_array) >= min_pixels:
                filtered.append(contour_array)
                slope = vy / vx if abs(vx) > 1e-6 else float('inf')
                slopes.append(slope)
                x_min = np.min(contour_array[:, 0, 0])
                x_max = np.max(contour_array[:, 0, 0])
                y_min = int(y0 + ((x_min - x0) * vy / vx))
                y_max = int(y0 + ((x_max - x0) * vy / vx))
                cv2.line(image_bgr, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
    return filtered, slopes
# ------------------------------


def filter_and_refine_contours(contours, min_pixels, image_bgr,
                               initial_deviation_threshold=50,
                               final_deviation_threshold=10,
                               decay_factor=0.8,
                               max_iterations=10):
    """
    Filters contours by iteratively refining points that deviate from a fitted line.
    Removes first the contours with less than min_pixels

    Parameters:
    -----------
    contours : list of ndarray
        Input contours to be filtered.
    min_pixels : int
        Minimum number of pixels a contour must have to be kept.
    image_bgr : ndarray
        Image for visualizing fitted lines (optional).
    initial_deviation_threshold : float
        Initial threshold (in pixels) for deviation from the fitted line.
    final_deviation_threshold : float
        Minimum threshold to stop refinement.
    decay_factor : float
        How much the threshold is reduced each iteration.
    max_iterations : int
        Maximum number of refinement iterations per contour.

    Returns:
    --------
    filtered : list of ndarray
        Refined contours compatible with OpenCV.
    """
    filtered = []
    for contour in contours:
        if len(contour) >= min_pixels:
            contour_array = np.array(contour, dtype=np.float32)
            deviation_threshold = initial_deviation_threshold

            for _ in range(max_iterations):
                [vx, vy, x0, y0] = cv2.fitLine(contour_array, cv2.DIST_L2, 0, 0.01, 0.01)
                x = contour_array[:, 0, 0]
                y = contour_array[:, 0, 1]
                distances = np.abs(vy * (x - x0) - vx * (y - y0)) / np.sqrt(vx**2 + vy**2)
                valid = distances < deviation_threshold
                cleaned = contour_array[valid]
                if len(cleaned) < min_pixels or deviation_threshold < final_deviation_threshold:
                    break
                contour_array = cleaned
                deviation_threshold *= decay_factor

            if len(contour_array) >= min_pixels:
                contour_cv = np.round(contour_array).astype(np.int32).reshape((-1, 1, 2))
                filtered.append(contour_cv)

                # Draw fitted line
                x_min = np.min(contour_array[:, 0, 0])
                x_max = np.max(contour_array[:, 0, 0])
                y_min = int(y0 + ((x_min - x0) * vy / vx))
                y_max = int(y0 + ((x_max - x0) * vy / vx))
                cv2.line(image_bgr, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)

    return filtered


def remove_short_branches(binary_image, min_branch_length=10):
    '''
    Extracts main (longest) fiber-like edges from a skeletonized binary image
    using skan's graph-based branch pruning.

    Parameters:
    -----------
    binary_image : 2D ndarray
        A binary image where the fibers are already skeletonized.
    min_branch_length : float
        Minimum length of branch to keep. Shorter side branches are discarded.

    Returns:
    --------
    contours : list of ndarray
        A list of OpenCV-style contours (N, 1, 2) dtype=int32 representing main fiber edges.
    '''
    binary_image = binary_image.astype(bool)
    skeleton = Skeleton(binary_image)

    path_lengths = skeleton.path_lengths()
    keep_mask = path_lengths >= min_branch_length

    main_paths = []
    for idx, keep in enumerate(keep_mask):
        if keep:
            vertex_indices = skeleton.paths_list()[idx]
            main_path = skeleton.coordinates[vertex_indices]
            main_paths.append(main_path)

    contours = [np.round(path[:, ::-1]).astype(np.int32).reshape((-1, 1, 2)) for path in main_paths]
    return contours


def extract_main_fiber_edges(binary_image, avg_angle_deg, angle_threshold=10):
    '''
    Extracts main fiber-like branches from a skeletonized binary image using skan,
    filtering based on angle similarity to a reference fiber orientation.

    Parameters:
    -----------
    binary_image : 2D ndarray
        A binary image where the fibers are already skeletonized.
    avg_angle_deg : float
        The average fiber angle in degrees (from -90 to +90) to compare against.
    angle_threshold : float
        Maximum allowed angular deviation (in degrees) from the average fiber angle.

    Returns:
    --------
    contours : list of ndarray
        A list of OpenCV-style contours (N, 1, 2) dtype=int32 representing main fiber edges.
    '''
    binary_image = binary_image.astype(bool)
    skeleton = Skeleton(binary_image)

    main_paths = []
    for path_indices in skeleton.paths_list():
        coords = skeleton.coordinates[path_indices]
        if len(coords) < 2:
            continue  # not enough points to fit a line

        # Flip (y, x) -> (x, y) to match OpenCV convention
        flipped_coords = coords[:, ::-1].astype(np.float32)
        contour = flipped_coords.reshape((-1, 1, 2))

        # Fit line using OpenCV
        [vx, vy, _, _] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        angle_deg = np.degrees(np.arctan2(vy, vx))[0]

        # Compute the smallest angle between two crossing lines (0–90°)
        angle_diff = abs(angle_deg - avg_angle_deg) % 180
        if angle_diff > 90:
            angle_diff = 180 - angle_diff

        if angle_diff < angle_threshold:
            main_paths.append(flipped_coords)

    # Convert to OpenCV-style contours
    contours = [np.round(path).astype(np.int32).reshape((-1, 1, 2)) for path in main_paths]
    return contours


def rotate_and_measure(contours, avg_slope, width, height, pixels_per_micron, dia_range, sep_range):
    avg_angle = float(np.nan_to_num(np.degrees(np.arctan(avg_slope)), nan=0.0))
    center = (width // 2, height // 2)
    rot_mat = cv2.getRotationMatrix2D(center, 90 + avg_angle, 1.0)
    rotated_contours = [cv2.transform(np.array(c, dtype=np.float32), rot_mat).astype(int) for c in contours]
    rotated_img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(rotated_img)

    fiber_diameters, fiber_separations = [], []
    for y in range(0, height, 15):
        x_vals = []
        for c in rotated_contours:
            x = c[:, 0, 0]
            y_c = c[:, 0, 1]
            if np.min(y_c) <= y <= np.max(y_c):
                x_at_y = x[np.argmin(np.abs(y_c - y))]
                x_vals.append(int(x_at_y))
        x_vals.sort()
        for i in range(len(x_vals) - 1):
            draw.line([(x_vals[i], y), (x_vals[i+1], y)], fill=(255, 255, 0), width=1)
            dist = (x_vals[i+1] - x_vals[i]) / pixels_per_micron
            if dia_range[0] < dist < dia_range[1]:
                fiber_diameters.append(dist)
            elif sep_range[0] < dist < sep_range[1]:
                fiber_separations.append(dist)

    for c in rotated_contours:
        draw.line([tuple(pt[0]) for pt in c], fill=(0, 255, 0), width=1)

    return rotated_img, fiber_diameters, fiber_separations


def filter_contours_by_proximity(candidate_contours, reference_contours, max_distance=20):
    """
    Filters contours based on proximity to any of the reference contours.

    Parameters:
    -----------
    candidate_contours : list of ndarray
        Contours to evaluate (e.g., from skan).
    reference_contours : list of ndarray
        Clean fiber edge contours from step 2.
    max_distance : float
        Maximum allowed distance in pixels from candidate to reference to keep it.

    Returns:
    --------
    filtered : list of ndarray
        Candidate contours that are spatially close to reference contours.
    """

    # Flatten all reference contour points into one array for KD-Tree
    ref_points = np.vstack([c.reshape(-1, 2) for c in reference_contours])
    tree = cKDTree(ref_points)

    filtered = []
    for contour in candidate_contours:
        # Get all points in the candidate contour
        points = contour.reshape(-1, 2)
        distances, _ = tree.query(points)
        if np.min(distances) <= max_distance:
            filtered.append(contour)

    return filtered


def contours_to_skeleton_image(contours, image_shape, line_thickness=1):
    """
    Converts a list of contours into a skeletonized binary image for use with skan.

    Parameters:
    -----------
    contours : list of ndarray
        OpenCV-style contours (N, 1, 2).
    image_shape : tuple
        Shape of the original image (height, width).
    line_thickness : int
        Thickness of lines to draw before skeletonizing.

    Returns:
    --------
    skeleton : 2D ndarray
        Binary skeleton image suitable for skan.
    """
    # Create empty black image
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Draw all contours onto mask
    for c in contours:
        cv2.polylines(mask, [c], isClosed=False, color=255, thickness=line_thickness)

    # Binarize
    binary = (mask > 0).astype(np.uint8)

    # Skeletonize
    skeleton = skeletonize(binary)

    return skeleton


def compute_average_angle_from_contours(contours):
    """
    Computes the average slope and angle (in degrees) from a list of OpenCV contours.

    Returns:
    --------
    avg_slope : float
    avg_angle_deg : float
    """
    slopes = []
    for contour in contours:
        [vx, vy, _, _] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        slope = vy / vx if abs(vx) > 1e-6 else float('inf')
        slopes.append(slope)

    if not slopes:
        return None, None

    avg_slope = sum(slopes) / len(slopes)
    avg_angle_deg = float(np.degrees(np.arctan(avg_slope)))
    return avg_slope, avg_angle_deg


def analyze_fibers(image_path, scale_bar_length=500, initial_kernel_size=91, final_kernel_size=21, min_pixels_in_rough_filtering=100,
                   fiber_diameter_range=(340, 440), fiber_spacing_range=(100, 300), top_n_contours=11):

    image = io.imread(image_path, as_gray=True)
    image_bgr = cv2.cvtColor((image / np.max(image) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    pixels_per_micron = get_pixel_to_micron_ratio(image, scale_bar_length)
    print(f"Pixels per micrometer: {pixels_per_micron}")

    mask, _ = get_polygon_roi(image)
    binary = preprocess_image(image, mask, initial_kernel_size)
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        print(f'Contour length is: {len(contour)}')
        cv2.drawContours(image_bgr, [contour], -1, (0, 255, 0), 1)  # Green lines
    print(f"Number of initial contours: {len(contours)}")

    initial_filtered_contours = filter_and_refine_contours(contours, min_pixels_in_rough_filtering, image_bgr)

    print(f"Number of contours after 1st filtering: {len(initial_filtered_contours)}")
    # filtered_contours = extract_main_fiber_edges(binary, min_branch_length=min_pixels)
    initial_filtered_contours = sorted(initial_filtered_contours, key=len, reverse=True)[:top_n_contours]

    # initial_filtered_contours = sorted(initial_filtered_contours, key=len, reverse=True)[:top_n_contours]

    for contour in initial_filtered_contours:
        print(f'Contour length is: {len(contour)}')
        cv2.drawContours(image_bgr, [contour], -1, (0, 255, 255), 1)  # Yellow lines

    if not initial_filtered_contours:
        print("No valid contours found for estimating average angle.")
        return

    # Recalculate slopes using only top-N filtered contours
    avg_slope, avg_angle_deg = compute_average_angle_from_contours(initial_filtered_contours)
    if avg_slope is None:
        print("No valid contours to compute slope.")
        return

    binary = preprocess_image(image, mask, final_kernel_size)
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        cv2.drawContours(image_bgr, [contour], -1, (255, 0, 255), 1)  # Magenta lines

    # Step 2: Use skan with angle-based filtering to get final contours
    # filtered_contours = extract_main_fiber_edges(binary, avg_angle_deg, angle_threshold=70)
    filtered_contours = filter_contours_by_proximity(contours, initial_filtered_contours, max_distance=20)
    filtered_contours = extract_main_fiber_edges(contours_to_skeleton_image(filtered_contours, image.shape), avg_angle_deg, angle_threshold=30)
    filtered_contours = filter_contours_by_proximity(filtered_contours, initial_filtered_contours, max_distance=10)
    # filtered_contours = filter_and_refine_contours(filtered_contours, 10, image_bgr, initial_deviation_threshold=20, final_deviation_threshold=8)

    filtered_contours = sorted(filtered_contours, key=len, reverse=True)[:max(1, int(0.9 * len(filtered_contours)))]
    # filtered_contours = remove_short_branches(contours_to_skeleton_image(filtered_contours, image.shape), min_branch_length=50)
    # --------------------
    # filtered_contours = sorted(filtered_contours, key=len, reverse=True)[:top_n_contours]
    # Sort contours by length (descending)
    # filtered_contours = sorted(filtered_contours, key=len, reverse=True)
    #
    # # Keep top 80% (at least 1 contour)
    # n_keep = max(1, int(0.8 * len(filtered_contours)))
    # filtered_contours = filtered_contours[:n_keep]
    # ----------------------
    print(f"Number of contours after filtering: {len(filtered_contours)}")

    for contour in filtered_contours:
        cv2.drawContours(image_bgr, [contour], -1, (255, 255, 0), 1)  # Cyan for edges

    if not filtered_contours:
        print("No valid contours found after angle-based filtering.")
        return

    # # Recalculate slopes using only top-N filtered contours
    # slopes = []
    # for contour in initial_filtered_contours:
    #     [vx, vy, _, _] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    #     slope = vy / vx if abs(vx) > 1e-6 else float('inf')
    #     slopes.append(slope)
    #
    # avg_slope = sum(slopes) / len(slopes)

    rotated_img, diameters, separations = rotate_and_measure(filtered_contours, avg_slope, image.shape[1],
                                                              image.shape[0], pixels_per_micron,
                                                              fiber_diameter_range, fiber_spacing_range)

    if diameters:
        plt.figure()
        plt.hist(diameters, bins=30, edgecolor='black')
        plt.xlabel("Fiber Diameter (µm)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Fiber Diameters")
        plt.show(block=False)
        plt.pause(0.1)

    if separations:
        plt.figure()
        plt.hist(separations, bins=30, edgecolor='black')
        plt.xlabel("Fiber separation (µm)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Fiber separations")
        plt.show(block=False)
        plt.pause(0.1)

    print(f"Average Fiber Diameter: {np.mean(diameters):.2f} µm")
    print(f"Average Fiber Separation: {np.mean(separations):.2f} µm")

    plt.figure()
    plt.imshow(rotated_img)
    plt.title("Rotated Contours Aligned Vertically")
    plt.axis("off")
    plt.show(block=False)
    plt.pause(0.1)

    plt.figure()
    plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    plt.title("Original Image with Fiber Borders and Fitted Lines")
    plt.axis("off")
    plt.show()


# Example usage
image_path = "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/90deg_400um_40perc/Image1_no_ruler.tif"
analyze_fibers(image_path,
               scale_bar_length=500,
               initial_kernel_size=101,
               final_kernel_size=21,
               min_pixels_in_rough_filtering=100,
               fiber_diameter_range=(370, 430),
               fiber_spacing_range=(50, 150),
               top_n_contours=11)
