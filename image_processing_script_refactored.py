
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, exposure, morphology
from PIL import Image, ImageDraw
import tkinter as tk
from skan import Skeleton
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from pathlib import Path
import os
import json
from tkinter import ttk, simpledialog, messagebox


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

# -------------------- Not used -----------------------------------

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
# ---------------------------------------------------------------

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



def rotate_and_measure(contours, avg_slope, width, height, pixels_per_micron, dia_range, sep_range, vertical_step=15):
    avg_angle = float(np.nan_to_num(np.degrees(np.arctan(avg_slope)), nan=0.0))
    center = (width // 2, height // 2)
    rot_mat = cv2.getRotationMatrix2D(center, 90 + avg_angle, 1.0)
    rotated_contours = [cv2.transform(np.array(c, dtype=np.float32), rot_mat).astype(int) for c in contours]
    rotated_img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(rotated_img)

    fiber_diameters, fiber_separations = [], []
    for y in range(0, height, vertical_step):
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


def filter_contour_points_by_proximity(candidate_contours, reference_contours, max_distance=10, min_contour_length=5):
    '''
    Removes individual points from contours that are too far from any reference contour point.

    Parameters:
    -----------
    candidate_contours : list of ndarray
        List of OpenCV-style contours to filter (N, 1, 2).
    reference_contours : list of ndarray
        List of reference contours to compare distances against.
    max_distance : float
        Maximum allowed distance (in pixels) to retain a point.
    min_contour_length : int
        Minimum number of points for a contour to be kept after filtering.

    Returns:
    --------
    filtered_contours : list of ndarray
        List of filtered contours, each with only nearby points.
    '''
    # Flatten all reference points for KD-tree
    ref_points = np.vstack([c.reshape(-1, 2) for c in reference_contours])
    tree = cKDTree(ref_points)

    filtered_contours = []

    for contour in candidate_contours:
        points = contour.reshape(-1, 2)
        distances, _ = tree.query(points)
        valid = distances < max_distance
        kept_points = points[valid]

        if len(kept_points) >= min_contour_length:
            cleaned = np.round(kept_points).astype(np.int32).reshape((-1, 1, 2))
            filtered_contours.append(cleaned)

    return filtered_contours


# def manual_select_fiber_edges(image):
#     """
#     Allows user to manually select multiple fiber edges by clicking.
#     Press ENTER to finish one fiber edge, and press 'q' to finish all.
#
#     Parameters:
#     -----------
#     image : 2D ndarray
#         Grayscale image where fiber edges are to be selected.
#
#     Returns:
#     --------
#     contours : list of ndarray
#         List of OpenCV-style contours (N, 1, 2) representing manually selected fiber edges.
#     """
#     import cv2
#
#     fiber_edges = []         # Stores all fiber edge point sets
#     current_edge = []        # Stores current fiber edge being drawn
#     temp_image = image.copy()
#     temp_image = ((temp_image / temp_image.max()) * 255).astype(np.uint8)
#     display_image = cv2.cvtColor(temp_image, cv2.COLOR_GRAY2BGR)
#
#     # Get screen scaling
#     screen_width, screen_height = get_screen_resolution()
#     scale = min(screen_width / display_image.shape[1], screen_height / display_image.shape[0])
#     scaled_image = cv2.resize(display_image, (int(display_image.shape[1] * scale), int(display_image.shape[0] * scale)))
#
#     def click_event(event, x, y, flags, param):
#         nonlocal current_edge
#         if event == cv2.EVENT_LBUTTONDOWN:
#             orig_x = int(x / scale)
#             orig_y = int(y / scale)
#             current_edge.append((orig_x, orig_y))
#             print(f"Point added: {orig_x}, {orig_y}")
#
#     cv2.namedWindow("Select Fiber Edges", cv2.WINDOW_NORMAL)
#
#     # Resize window to fit most of the screen without overflowing
#     cv2.resizeWindow("Select Fiber Edges",
#                      min(int(display_image.shape[1] * scale), screen_width),
#                      min(int(display_image.shape[0] * scale), screen_height))
#
#     cv2.setMouseCallback("Select Fiber Edges", click_event)
#
#
#     while True:
#         temp_display = scaled_image.copy()
#         for pt in current_edge:
#             scaled_pt = (int(pt[0] * scale), int(pt[1] * scale))
#             cv2.circle(temp_display, scaled_pt, 3, (0, 255, 0), -1)
#         cv2.imshow("Select Fiber Edges", temp_display)
#         key = cv2.waitKey(1) & 0xFF
#         if key == 13:  # ENTER
#             if len(current_edge) >= 2:
#                 contour = np.array(current_edge, dtype=np.int32).reshape((-1, 1, 2))
#                 fiber_edges.append(contour)
#                 print(f"Edge saved with {len(current_edge)} points.")
#             else:
#                 print("Need at least 2 points to define a fiber edge.")
#             current_edge = []
#         elif key == ord('q'):
#             print("Finished manual edge selection.")
#             break
#
#     cv2.destroyAllWindows()
#     return fiber_edges


def manual_select_fiber_edges(image, image_path):
    """
    Interactive fiber edge selection with support for saving/loading from JSON,
    and live editing via a Tkinter table before final save.

    Parameters:
    -----------
    image : 2D ndarray
        Grayscale image.
    image_path : str
        Full path to the image file.

    Returns:
    --------
    contours : list of ndarray
        List of OpenCV-style contours (N, 1, 2).
    """
    import cv2

    def to_cv2_contours(raw_edges):
        return [np.array(edge, dtype=np.int32).reshape((-1, 1, 2)) for edge in raw_edges]

    def from_cv2_contours(contours):
        return [[pt[0].tolist() for pt in contour] for contour in contours]

    def edit_fiber_points_gui(raw_edges):
        root = tk.Tk()
        root.title("Edit Fiber Edge Coordinates")
        notebook = ttk.Notebook(root)
        notebook.pack(expand=True, fill="both")
        all_entries = []

        for edge_idx, edge_points in enumerate(raw_edges):
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=f"Edge {edge_idx}")
            canvas = tk.Canvas(frame)
            scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            edge_entries = []
            for pt_idx, (x, y) in enumerate(edge_points):
                x_var = tk.StringVar(value=str(x))
                y_var = tk.StringVar(value=str(y))
                ttk.Label(scrollable_frame, text=f"Point {pt_idx}:").grid(row=pt_idx, column=0, padx=5, pady=2)
                x_entry = ttk.Entry(scrollable_frame, textvariable=x_var, width=6)
                y_entry = ttk.Entry(scrollable_frame, textvariable=y_var, width=6)
                x_entry.grid(row=pt_idx, column=1, padx=5)
                y_entry.grid(row=pt_idx, column=2, padx=5)
                edge_entries.append((x_var, y_var))
            all_entries.append(edge_entries)

        def save_and_close():
            for i, edge in enumerate(all_entries):
                new_edge = []
                for x_var, y_var in edge:
                    try:
                        x = int(x_var.get())
                        y = int(y_var.get())
                        new_edge.append([x, y])
                    except ValueError:
                        pass
                raw_edges[i] = new_edge
            root.destroy()

        save_button = ttk.Button(root, text="Save and Return", command=save_and_close)
        save_button.pack(pady=10)
        root.mainloop()
        return raw_edges

    # JSON storage
    json_path = Path(image_path).with_name(Path(image_path).stem + "_selected_fiber_points.json")

    # Load if exists
    if json_path.exists():
        print(f"Loading saved fiber points from: {json_path}")
        with open(json_path, "r") as f:
            fiber_edges = json.load(f)
    else:
        fiber_edges = []

    # Manual interaction
    current_edge = []
    temp_image = image.copy()
    temp_image = ((temp_image / temp_image.max()) * 255).astype(np.uint8)
    display_image = cv2.cvtColor(temp_image, cv2.COLOR_GRAY2BGR)
    screen_width, screen_height = get_screen_resolution()
    scale = min(screen_width / display_image.shape[1], screen_height / display_image.shape[0])
    scaled_image = cv2.resize(display_image, (int(display_image.shape[1] * scale), int(display_image.shape[0] * scale)))

    def click_event(event, x, y, flags, param):
        nonlocal current_edge
        if event == cv2.EVENT_LBUTTONDOWN:
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            current_edge.append([orig_x, orig_y])
            print(f"Point added: {orig_x}, {orig_y}")

    cv2.namedWindow("Select Fiber Edges", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Fiber Edges", int(display_image.shape[1] * scale), int(display_image.shape[0] * scale))
    cv2.setMouseCallback("Select Fiber Edges", click_event)

    print("Instructions:\n - Left-click to add points\n - ENTER to store edge\n - e to edit\n - q to quit and save")

    while True:
        temp_display = scaled_image.copy()

        # Draw current edge points
        for pt in current_edge:
            cv2.circle(temp_display, (int(pt[0] * scale), int(pt[1] * scale)), 3, (0, 255, 0), -1)

        # Draw saved edges
        for edge in fiber_edges:
            for pt in edge:
                cv2.circle(temp_display, (int(pt[0] * scale), int(pt[1] * scale)), 2, (255, 0, 0), -1)

        cv2.imshow("Select Fiber Edges", temp_display)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # ENTER
            if len(current_edge) >= 2:
                fiber_edges.append(current_edge)
                print(f"Saved edge with {len(current_edge)} points.")
            else:
                print("Edge too short.")
            current_edge = []

        elif key == ord('e'):
            print("Opening editor...")
            fiber_edges = edit_fiber_points_gui(fiber_edges)
            print("Returned from editor.")

        elif key == ord('q'):
            print("Exiting and saving...")
            break

    cv2.destroyAllWindows()

    # Save to JSON
    print(f"Saving to file: {json_path}")
    with open(json_path, "w") as f:
        json.dump(fiber_edges, f)

    return to_cv2_contours(fiber_edges)


def edit_fiber_points_gui(raw_edges):
    """
    Opens a GUI to view and edit fiber edge coordinates.
    """
    root = tk.Tk()
    root.title("Edit Fiber Edge Coordinates")

    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill="both")

    all_entries = []  # Store entries for each edge

    for edge_idx, edge_points in enumerate(raw_edges):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=f"Edge {edge_idx}")

        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        edge_entries = []
        for pt_idx, (x, y) in enumerate(edge_points):
            x_var = tk.StringVar(value=str(x))
            y_var = tk.StringVar(value=str(y))

            ttk.Label(scrollable_frame, text=f"Point {pt_idx}:").grid(row=pt_idx, column=0, padx=5, pady=2)
            x_entry = ttk.Entry(scrollable_frame, textvariable=x_var, width=6)
            y_entry = ttk.Entry(scrollable_frame, textvariable=y_var, width=6)
            x_entry.grid(row=pt_idx, column=1, padx=5)
            y_entry.grid(row=pt_idx, column=2, padx=5)

            edge_entries.append((x_var, y_var))

        all_entries.append(edge_entries)

    def save_and_close():
        for i, edge in enumerate(all_entries):
            new_edge = []
            for x_var, y_var in edge:
                try:
                    x = int(x_var.get())
                    y = int(y_var.get())
                    new_edge.append([x, y])
                except ValueError:
                    pass  # Ignore invalid entries
            raw_edges[i] = new_edge
        root.destroy()

    save_button = ttk.Button(root, text="Save and Close", command=save_and_close)
    save_button.pack(pady=10)

    root.mainloop()
    return raw_edges


def analyze_fibers(image_path, scale_bar_length=500, manual_pixels_per_micron=None, manual_detection=False, initial_kernel_size=91, apply_first_angle_filter='no', expected_angle_deg=88, treshold_angle_init=60, final_kernel_size=21, min_pixels_in_rough_filtering=100, ini_dev_thresh=50, fin_dev_thresh=20,
                   fiber_diameter_range=(340, 440), fiber_spacing_range=(100, 300), top_n_contours=11, vertical_step=15, max_distance_first_filter=100, angle_threshold_second_filter=30,max_distance_third_filter=20):

    image = io.imread(image_path, as_gray=True)
    image_bgr = cv2.cvtColor((image / np.max(image) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    if manual_pixels_per_micron is None:
        pixels_per_micron = get_pixel_to_micron_ratio(image, scale_bar_length)
        print(f"Pixels per micrometer: {pixels_per_micron:.5f}")
    else:
        pixels_per_micron = manual_pixels_per_micron
        print(f"Pixels per micrometer (manual input): {pixels_per_micron:.5f}")

    mask, _ = get_polygon_roi(image)

    if not manual_detection:
        # First detection of fiber edge with a roughly blurred image
        binary = preprocess_image(image, mask, initial_kernel_size)
        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # for contour in contours:
        #     print(f'Contour length is: {len(contour)}')
        #     cv2.drawContours(image_bgr, [contour], -1, (0, 255, 0), 1)  # Green lines
        print(f"Number of initial contours: {len(contours)}")

        if apply_first_angle_filter == "yes":
            initial_filtered_contours = extract_main_fiber_edges(contours_to_skeleton_image(contours, image.shape), expected_angle_deg, treshold_angle_init)
            initial_filtered_contours = filter_and_refine_contours(initial_filtered_contours, min_pixels_in_rough_filtering, image_bgr,
                                                               initial_deviation_threshold=ini_dev_thresh,
                                                               final_deviation_threshold=fin_dev_thresh,
                                                               decay_factor=0.8,
                                                               max_iterations=15)
        elif apply_first_angle_filter == "no":
            initial_filtered_contours = filter_and_refine_contours(contours, min_pixels_in_rough_filtering, image_bgr,
                                                                   initial_deviation_threshold=ini_dev_thresh,
                                                                   final_deviation_threshold=fin_dev_thresh,
                                                                   decay_factor=0.8,
                                                                   max_iterations=15)

        for contour in initial_filtered_contours:
            print(f'Contour length is: {len(contour)}')
            cv2.drawContours(image_bgr, [contour], -1, (0, 255, 0), 1)  # Green lines

        print(f"Number of contours after 1st filtering: {len(initial_filtered_contours)}")
        initial_filtered_contours = sorted(initial_filtered_contours, key=len, reverse=True)[:top_n_contours]

        for contour in initial_filtered_contours:
            print(f'Contour length is: {len(contour)}')
            cv2.drawContours(image_bgr, [contour], -1, (0, 255, 255), 1)  # Yellow lines

    else:

        initial_filtered_contours = manual_select_fiber_edges(image, image_path)
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

    # Filtering steps with finer details from the fiber edges

    binary = preprocess_image(image, mask, final_kernel_size)
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        cv2.drawContours(image_bgr, [contour], -1, (255, 0, 255), 1)  # Magenta lines

    # Step 2: Use skan with angle-based filtering to get final contours
    # filtered_contours = extract_main_fiber_edges(binary, avg_angle_deg, angle_threshold=70)
    filtered_contours = filter_contours_by_proximity(contours, initial_filtered_contours, max_distance=max_distance_first_filter)
    filtered_contours = extract_main_fiber_edges(contours_to_skeleton_image(filtered_contours, image.shape), avg_angle_deg, angle_threshold=angle_threshold_second_filter)
    filtered_contours = filter_contours_by_proximity(filtered_contours, initial_filtered_contours, max_distance=max_distance_third_filter)
    # filtered_contours = filter_contour_points_by_proximity(filtered_contours, initial_filtered_contours, max_distance=10, min_contour_length=20)
    # filtered_contours = filter_and_refine_contours(filtered_contours, 10, image_bgr, initial_deviation_threshold=20, final_deviation_threshold=10)
    filtered_contours = sorted(filtered_contours, key=len, reverse=True)[:max(1, int(0.9 * len(filtered_contours)))]
    # filtered_contours = remove_short_branches(contours_to_skeleton_image(filtered_contours, image.shape), min_branch_length=50)

    print(f"Number of contours after filtering: {len(filtered_contours)}")

    for contour in filtered_contours:
        cv2.drawContours(image_bgr, [contour], -1, (255, 255, 0), 1)  # Cyan for edges

    if not filtered_contours:
        print("No valid contours found after angle-based filtering.")
        return

    rotated_img, diameters, separations = rotate_and_measure(filtered_contours, avg_slope, image.shape[1],
                                                              image.shape[0], pixels_per_micron,
                                                              fiber_diameter_range, fiber_spacing_range, vertical_step)

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

    mean_d = np.mean(diameters)
    mean_s = np.mean(separations)
    pitch = mean_d + mean_s

    sem_d = np.std(diameters, ddof=1) / np.sqrt(len(diameters))
    sem_s = np.std(separations, ddof=1) / np.sqrt(len(separations))
    pitch_error = np.sqrt(sem_d**2 + sem_s**2)

    print(f"Average Fiber Diameter: {mean_d:.3f} µm")
    print(f"Standard deviation Fiber Diameter: {np.std(diameters, ddof=1):.3f} µm")
    print(f"Standard error of the mean fiber Diameter: {sem_d:.3f} µm")
    print(f"Average Fiber Separation: {mean_s:.3f} µm")
    print(f"Standard deviation Fiber Separation: {np.std(separations, ddof=1):.3f} µm")
    print(f"Standard error of the mean fiber Separation: {sem_s:.3f} µm")
    print(f'Average pitch: {np.mean(diameters) + np.mean(separations):.3f} µm')
    print(f"Pitch = {pitch:.2f} ± {pitch_error:.2f} µm")


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

    return diameters, separations

# # Example usage

# image1_path = "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/90deg_400um_40perc/Image1_no_ruler.tif"
# image_path = "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/90deg_400um_40perc/Figure4_rigth_edge.tif"
# image3_path = "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/90deg_400um_40perc/Figure4_rigth_edge.tif"
# image4_path = "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/90deg_400um_40perc/Figure6_top_rigth_corner.tif"
#
# image_paths = [image1_path, image2_path, image3_path, image4_path]
#
#
#
# image_path = "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/90deg_400um_45perc/bottom_edge.tif"
# image_path = "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/90deg_400um_45perc/channels.tif"
# image_path = "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/90deg_400um_45perc/DownSideUp_bottom_rigth_corner.tif"
# image_path = "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/90deg_400um_45perc/DownSideUp_bottom_rigth_corner.tif"
# image_path = "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/90deg_400um_45perc/DSU_x70.tif"
# image_path = "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/90deg_400um_45perc/Figure1_with_dimens_x70.bmp"
# image_path = "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/90deg_400um_45perc/Figure2_up_rigth_corner_no_dimens.tif"

# image_path = "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/90deg_400um_45perc/Up_left_no_dimens.tif"

# path = Path("C:\Users\dfpinedaquijan\surfdrive\PhD Project\Data\SEM\45deg_250um_45p_b\x60_center2.jpg")
# image_path = path.as_posix()
# image_path = "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/400um_50p_a/img1_35X_10kV_PC44.jpg"
# image_path = "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/400um_50p_a/img4.tif"

# image_path = "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/600um_50p_a/img10_10kV_X35_PC44.jpg"

# image_path = "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/400um_50p_a/img6.tif"

# image_path = "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/90deg_400um_45perc/Figure2_up_rigth_corner_no_dimens.tif"

# image_path = "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/600um_45p_a/img22.tif"
# analyze_fibers(image_path,
#                scale_bar_length=508,
#                initial_kernel_size=61,
#                manual_pixels_per_micron=None,
#                apply_first_angle_filter='no',
#                manual_detection=True,
#                expected_angle_deg=90,
#                treshold_angle_init=50,
#                final_kernel_size=21,
#                min_pixels_in_rough_filtering=100,
#                ini_dev_thresh=60,
#                fin_dev_thresh=20,
#                fiber_diameter_range=(520, 620),
#                fiber_spacing_range=(150, 400),
#                top_n_contours=10,
#                vertical_step=5,
#                max_distance_first_filter=50,
#                angle_threshold_second_filter=30,
#                max_distance_third_filter=25)


# --------------------------------- Configs 400um_50p_a --------------------------------------

# configs = {}
# configs["C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/400um_50p_a/img4.tif"] = {
#     "scale_bar_length": 508,
#     "initial_kernel_size": 61,
#     "manual_pixels_per_micron": None,
#     "apply_first_angle_filter": "no",
#     "manual_detection": False,
#     "expected_angle_deg": 90,
#     "treshold_angle_init": 50,
#     "final_kernel_size": 31,
#     "min_pixels_in_rough_filtering": 100,
#     "ini_dev_thresh": 45,
#     "fin_dev_thresh": 10,
#     "fiber_diameter_range": (300, 500),
#     "fiber_spacing_range": (100, 300),
#     "top_n_contours": 10,
#     "vertical_step": 5,
#     "max_distance_first_filter": 40,
#     "angle_threshold_second_filter": 30,
#     "max_distance_third_filter": 10
# }

# --------------------------------- Configs 400um_50p_a --------------------------------------
# configs = {}
# configs["C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/400um_50p_a/img2.tif"] = {
#     "scale_bar_length": 508,
#     "initial_kernel_size": 61,
#     "manual_pixels_per_micron": None,
#     "apply_first_angle_filter": "no",
#     "manual_detection": False,
#     "expected_angle_deg": 89,
#     "treshold_angle_init": 50,
#     "final_kernel_size": 31,
#     "min_pixels_in_rough_filtering": 100,
#     "ini_dev_thresh": 45,
#     "fin_dev_thresh": 10,
#     "fiber_diameter_range": (300, 500),
#     "fiber_spacing_range": (100, 300),
#     "top_n_contours": 10,
#     "vertical_step": 5,
#     "max_distance_first_filter": 40,
#     "angle_threshold_second_filter": 30,
#     "max_distance_third_filter": 5
# }
#
# configs["C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/400um_50p_a/img3.tif"] = {
#     "scale_bar_length": 508,
#     "initial_kernel_size": 61,
#     "manual_pixels_per_micron": None,
#     "apply_first_angle_filter": "no",
#     "manual_detection": False,
#     "expected_angle_deg": 89,
#     "treshold_angle_init": 50,
#     "final_kernel_size": 31,
#     "min_pixels_in_rough_filtering": 100,
#     "ini_dev_thresh": 45,
#     "fin_dev_thresh": 10,
#     "fiber_diameter_range": (300, 500),
#     "fiber_spacing_range": (100, 300),
#     "top_n_contours": 12,
#     "vertical_step": 5,
#     "max_distance_first_filter": 40,
#     "angle_threshold_second_filter": 30,
#     "max_distance_third_filter": 5
# }
#
# configs["C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/400um_50p_a/img4.tif"] = {
#     "scale_bar_length": 508,
#     "initial_kernel_size": 61,
#     "manual_pixels_per_micron": None,
#     "apply_first_angle_filter": "no",
#     "manual_detection": False,
#     "expected_angle_deg": 89,
#     "treshold_angle_init": 50,
#     "final_kernel_size": 31,
#     "min_pixels_in_rough_filtering": 100,
#     "ini_dev_thresh": 45,
#     "fin_dev_thresh": 10,
#     "fiber_diameter_range": (300, 500),
#     "fiber_spacing_range": (100, 300),
#     "top_n_contours": 13,
#     "vertical_step": 5,
#     "max_distance_first_filter": 40,
#     "angle_threshold_second_filter": 30,
#     "max_distance_third_filter": 5
# }
#
# configs["C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/400um_50p_a/img5.tif"] = {
#     "scale_bar_length": 508,
#     "initial_kernel_size": 61,
#     "manual_pixels_per_micron": None,
#     "apply_first_angle_filter": "no",
#     "manual_detection": False,
#     "expected_angle_deg": 89,
#     "treshold_angle_init": 50,
#     "final_kernel_size": 31,
#     "min_pixels_in_rough_filtering": 100,
#     "ini_dev_thresh": 45,
#     "fin_dev_thresh": 10,
#     "fiber_diameter_range": (300, 500),
#     "fiber_spacing_range": (100, 300),
#     "top_n_contours": 13,
#     "vertical_step": 5,
#     "max_distance_first_filter": 40,
#     "angle_threshold_second_filter": 30,
#     "max_distance_third_filter": 5
# }
#
# configs["C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/400um_50p_a/img6.tif"] = {
#     "scale_bar_length": 508,
#     "initial_kernel_size": 61,
#     "manual_pixels_per_micron": None,
#     "apply_first_angle_filter": "no",
#     "manual_detection": False,
#     "expected_angle_deg": 89,
#     "treshold_angle_init": 50,
#     "final_kernel_size": 31,
#     "min_pixels_in_rough_filtering": 100,
#     "ini_dev_thresh": 45,
#     "fin_dev_thresh": 10,
#     "fiber_diameter_range": (300, 500),
#     "fiber_spacing_range": (100, 300),
#     "top_n_contours": 5,
#     "vertical_step": 5,
#     "max_distance_first_filter": 40,
#     "angle_threshold_second_filter": 30,
#     "max_distance_third_filter": 5
# }


# --------------------------------- SUYE's image -----------------------------------
# image_path = "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/45deg_250um_45p_a/FIBERS/fibers2.tif"
#
# analyze_fibers(image_path,
#                scale_bar_length=500,
#                initial_kernel_size=21,
#                manual_pixels_per_micron=1.25,
#                apply_first_angle_filter='yes',
#                manual_detection=True,
#                expected_angle_deg=45,
#                treshold_angle_init=50,
#                final_kernel_size=11,
#                min_pixels_in_rough_filtering=50,
#                ini_dev_thresh=45,
#                fin_dev_thresh=15,
#                fiber_diameter_range=(400, 650),
#                fiber_spacing_range=(100, 400),
#                top_n_contours=32,
#                vertical_step=5,
#                max_distance_first_filter=40,
#                angle_threshold_second_filter=30,
#                max_distance_third_filter=10)
# ------------------------------------------------------------------------------------

# configs = {}
# configs["C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/45deg_250um_45p_b/x60_center2.jpg"] = {
#         "scale_bar_length": 200,
#         "initial_kernel_size": 171,
#         "apply_first_angle_filter": 'yes',
#         "expected_angle_deg": -45,
#         "treshold_angle_init": 70,
#         "final_kernel_size": 21,
#         "min_pixels_in_rough_filtering": 100,
#         "ini_dev_thresh": 50,
#         "fin_dev_thresh": 10,
#         "fiber_diameter_range": (200, 300),
#         "fiber_spacing_range": (100, 200),
#         "top_n_contours": 20,
#         "vertical_step": 5,
#         "max_distance_first_filter": 40,
#         "angle_threshold_second_filter": 30,
#         "max_distance_third_filter": 10
#     }

# configs = {
#     "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/90deg_400um_40perc/Image1_no_ruler.tif": {
#         "initial_kernel_size": 91,
#         "final_kernel_size": 31,
#         "min_pixels_in_rough_filtering": 50,
#         "fiber_diameter_range": (340, 460),
#         "fiber_spacing_range": (20, 150),
#         "top_n_contours": 11,
#         "vertical_step": 5,
#         "max_distance_first_filter": 20,
#         "angle_threshold_second_filter": 30,
#         "max_distance_third_filter": 10,
#         "scale_bar_length": 500
#     },
#     "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/90deg_400um_40perc/Figure4_rigth_edge_Copy.tif": {
#         "initial_kernel_size": 121,
#         "final_kernel_size": 31,
#         "min_pixels_in_rough_filtering": 50,
#         "fiber_diameter_range": (340, 460),
#         "fiber_spacing_range": (20, 150),
#         "top_n_contours": 6,
#         "vertical_step": 5,
#         "max_distance_first_filter": 30,
#         "angle_threshold_second_filter": 30,
#         "max_distance_third_filter": 10,
#         "scale_bar_length": 500
#     },
#     "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/90deg_400um_40perc/Figure4_rigth_edge.tif": {
#         "initial_kernel_size": 121,
#         "final_kernel_size": 31,
#         "min_pixels_in_rough_filtering": 50,
#         "fiber_diameter_range": (340, 460),
#         "fiber_spacing_range": (20, 150),
#         "top_n_contours": 10,
#         "vertical_step": 5,
#         "max_distance_first_filter": 30,
#         "angle_threshold_second_filter": 30,
#         "max_distance_third_filter": 10,
#         "scale_bar_length": 500
#     },
#     "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/90deg_400um_40perc/Figure6_top_rigth_corner.tif": {
#         "initial_kernel_size": 201,
#         "final_kernel_size": 31,
#         "min_pixels_in_rough_filtering": 50,
#         "fiber_diameter_range": (340, 460),
#         "fiber_spacing_range": (20, 150),
#         "top_n_contours": 9,
#         "vertical_step": 5,
#         "max_distance_first_filter": 100,
#         "angle_threshold_second_filter": 30,
#         "max_distance_third_filter": 50,
#         "scale_bar_length": 500
#     }
# }
#
# --------------------------------------------------- 400um_45p_a ----------------------------------------------
# configs = {
#     "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/90deg_400um_45perc/bottom_edge.tif":
#         {
#             "scale_bar_length": 500,
#             "initial_kernel_size": 121,
#             "apply_first_angle_filter": "no",
#             "expected_angle_deg": 88,
#             "treshold_angle_init": 60,
#             "final_kernel_size": 31,
#             "min_pixels_in_rough_filtering": 50,
#             "ini_dev_thresh": 50,
#             "fin_dev_thresh": 10,
#             "fiber_diameter_range": (340, 430),
#             "fiber_spacing_range": (50, 200),
#             "top_n_contours": 16,
#             "vertical_step": 5,
#             "max_distance_first_filter": 30,
#             "angle_threshold_second_filter": 30,
#             "max_distance_third_filter": 10
#         },
#     "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/90deg_400um_45perc/channels.tif":
#         {
#             "scale_bar_length": 100,
#             "initial_kernel_size": 101,
#             "apply_first_angle_filter": "no",
#             "expected_angle_deg": 88,
#             "treshold_angle_init": 60,
#             "final_kernel_size": 31,
#             "min_pixels_in_rough_filtering": 100,
#             "ini_dev_thresh": 40,
#             "fin_dev_thresh": 15,
#             "fiber_diameter_range": (340, 420),
#             "fiber_spacing_range": (50, 200),
#             "top_n_contours": 6,
#             "vertical_step": 5,
#             "max_distance_first_filter": 30,
#             "angle_threshold_second_filter": 25,
#             "max_distance_third_filter": 8
#         },
#     "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/90deg_400um_45perc/DownSideUp_bottom_rigth_corner.tif":
#         {
#             "scale_bar_length": 500,
#             "initial_kernel_size": 71,
#             "apply_first_angle_filter": "yes",
#             "expected_angle_deg": 88,
#             "treshold_angle_init": 60,
#             "final_kernel_size": 21,
#             "min_pixels_in_rough_filtering": 100,
#             "ini_dev_thresh": 60,
#             "fin_dev_thresh": 30,
#             "fiber_diameter_range": (300, 400),
#             "fiber_spacing_range": (50, 200),
#             "top_n_contours": 20,
#             "vertical_step": 5,
#             "max_distance_first_filter": 40,
#             "angle_threshold_second_filter": 20,
#             "max_distance_third_filter": 10
#         },
#     "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/90deg_400um_45perc/DSU_x70.tif":
#         {
#             "scale_bar_length": 200,
#             "initial_kernel_size": 211,
#             "apply_first_angle_filter": "yes",
#             "expected_angle_deg": 88,
#             "treshold_angle_init": 60,
#             "final_kernel_size": 21,
#             "min_pixels_in_rough_filtering": 100,
#             "ini_dev_thresh": 80,
#             "fin_dev_thresh": 50,
#             "fiber_diameter_range": (300, 420),
#             "fiber_spacing_range": (50, 220),
#             "top_n_contours": 10,
#             "vertical_step": 5,
#             "max_distance_first_filter": 40,
#             "angle_threshold_second_filter": 15,
#             "max_distance_third_filter": 15
#         },
#     "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/90deg_400um_45perc/Figure1_with_dimens_x70.bmp":
#         {
#             "scale_bar_length": 200,
#             "initial_kernel_size": 171,
#             "apply_first_angle_filter": "yes",
#             "expected_angle_deg": 88,
#             "treshold_angle_init": 60,
#             "final_kernel_size": 21,
#             "min_pixels_in_rough_filtering": 100,
#             "ini_dev_thresh": 80,
#             "fin_dev_thresh": 50,
#             "fiber_diameter_range": (300, 420),
#             "fiber_spacing_range": (50, 220),
#             "top_n_contours": 10,
#             "vertical_step": 5,
#             "max_distance_first_filter": 40,
#             "angle_threshold_second_filter": 25,
#             "max_distance_third_filter": 10
#         },
#     "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/90deg_400um_45perc/Figure2_up_rigth_corner_no_dimens.tif":
#         {
#             "scale_bar_length": 500,
#             "initial_kernel_size": 171,
#             "apply_first_angle_filter": "yes",
#             "expected_angle_deg": 88,
#             "treshold_angle_init": 70,
#             "final_kernel_size": 21,
#             "min_pixels_in_rough_filtering": 100,
#             "ini_dev_thresh": 80,
#             "fin_dev_thresh": 50,
#             "fiber_diameter_range": (340, 440),
#             "fiber_spacing_range": (50, 220),
#             "top_n_contours": 10,
#             "vertical_step": 5,
#             "max_distance_first_filter": 40,
#             "angle_threshold_second_filter": 30,
#             "max_distance_third_filter": 10
#         },
#     "C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/90deg_400um_45perc/Up_left_no_dimens.tif":
#         {
#             "scale_bar_length": 500,
#             "initial_kernel_size": 161,
#             "apply_first_angle_filter": "yes",
#             "expected_angle_deg": 88,
#             "treshold_angle_init": 60,
#             "final_kernel_size": 21,
#             "min_pixels_in_rough_filtering": 100,
#             "ini_dev_thresh": 60,
#             "fin_dev_thresh": 20,
#             "fiber_diameter_range": (340, 440),
#             "fiber_spacing_range": (50, 220),
#             "top_n_contours": 14,
#             "vertical_step": 5,
#             "max_distance_first_filter": 40,
#             "angle_threshold_second_filter": 25,
#             "max_distance_third_filter": 8
#         }
# }

# configs["C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/90deg_400um_45perc/img13.tif"] = {
#     "scale_bar_length": 508,
#     "initial_kernel_size": 61,
#     "manual_pixels_per_micron": None,
#     "apply_first_angle_filter": "no",
#     "manual_detection": True,
#     "expected_angle_deg": 90,
#     "treshold_angle_init": 50,
#     "final_kernel_size": 31,
#     "min_pixels_in_rough_filtering": 100,
#     "ini_dev_thresh": 45,
#     "fin_dev_thresh": 10,
#     "fiber_diameter_range": (300, 500),
#     "fiber_spacing_range": (100, 300),
#     "top_n_contours": 14,
#     "vertical_step": 5,
#     "max_distance_first_filter": 40,
#     "angle_threshold_second_filter": 30,
#     "max_distance_third_filter": 15
# }

# ----------------------------------------- 600um_40p_a -----------------------------------------------------------
# configs = {}
#
# configs["C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/600um_40p_a/img36.tif"] = {
#     "scale_bar_length": 508,
#     "initial_kernel_size": 61,
#     "manual_pixels_per_micron": None,
#     "apply_first_angle_filter": "no",
#     "manual_detection": False,
#     "expected_angle_deg": 90,
#     "treshold_angle_init": 50,
#     "final_kernel_size": 21,
#     "min_pixels_in_rough_filtering": 100,
#     "ini_dev_thresh": 60,
#     "fin_dev_thresh": 20,
#     "fiber_diameter_range": (400, 620),
#     "fiber_spacing_range": (150, 320),
#     "top_n_contours": 10,
#     "vertical_step": 5,
#     "max_distance_first_filter": 40,
#     "angle_threshold_second_filter": 20,
#     "max_distance_third_filter": 10
# }
#
# configs["C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/600um_40p_a/img35.tif"] = {
#     "scale_bar_length": 508,
#     "initial_kernel_size": 61,
#     "manual_pixels_per_micron": None,
#     "apply_first_angle_filter": "no",
#     "manual_detection": False,
#     "expected_angle_deg": 90,
#     "treshold_angle_init": 50,
#     "final_kernel_size": 21,
#     "min_pixels_in_rough_filtering": 100,
#     "ini_dev_thresh": 60,
#     "fin_dev_thresh": 20,
#     "fiber_diameter_range": (400, 620),
#     "fiber_spacing_range": (150, 320),
#     "top_n_contours": 10,
#     "vertical_step": 5,
#     "max_distance_first_filter": 40,
#     "angle_threshold_second_filter": 20,
#     "max_distance_third_filter": 10
# }
#
# configs["C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/600um_40p_a/img30.tif"] = {
#     "scale_bar_length": 508,
#     "initial_kernel_size": 61,
#     "manual_pixels_per_micron": None,
#     "apply_first_angle_filter": "no",
#     "manual_detection": True,
#     "expected_angle_deg": 90,
#     "treshold_angle_init": 50,
#     "final_kernel_size": 21,
#     "min_pixels_in_rough_filtering": 100,
#     "ini_dev_thresh": 60,
#     "fin_dev_thresh": 20,
#     "fiber_diameter_range": (460, 620),
#     "fiber_spacing_range": (100, 400),
#     "top_n_contours": 11,
#     "vertical_step": 5,
#     "max_distance_first_filter": 50,
#     "angle_threshold_second_filter": 30,
#     "max_distance_third_filter": 20
# }
#
# configs["C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/600um_40p_a/img31.tif"] = {
#     "scale_bar_length": 508,
#     "initial_kernel_size": 61,
#     "manual_pixels_per_micron": None,
#     "apply_first_angle_filter": "no",
#     "manual_detection": True,
#     "expected_angle_deg": 90,
#     "treshold_angle_init": 50,
#     "final_kernel_size": 21,
#     "min_pixels_in_rough_filtering": 100,
#     "ini_dev_thresh": 60,
#     "fin_dev_thresh": 20,
#     "fiber_diameter_range": (460, 620),
#     "fiber_spacing_range": (100, 400),
#     "top_n_contours": 11,
#     "vertical_step": 5,
#     "max_distance_first_filter": 50,
#     "angle_threshold_second_filter": 30,
#     "max_distance_third_filter": 20
# }
# --------------------------------------------------------------------------------------------------
configs = {}

configs["C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/600um_50p_a/img28.tif"] = {
    "scale_bar_length": 508,
    "initial_kernel_size": 61,
    "manual_pixels_per_micron": None,
    "apply_first_angle_filter": "no",
    "manual_detection": False,
    "expected_angle_deg": 90,
    "treshold_angle_init": 50,
    "final_kernel_size": 21,
    "min_pixels_in_rough_filtering": 100,
    "ini_dev_thresh": 60,
    "fin_dev_thresh": 20,
    "fiber_diameter_range": (400, 650),
    "fiber_spacing_range": (50, 400),
    "top_n_contours": 4,
    "vertical_step": 5,
    "max_distance_first_filter": 40,
    "angle_threshold_second_filter": 20,
    "max_distance_third_filter": 10
}

configs["C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/600um_50p_a/img29.tif"] = {
    "scale_bar_length": 508,
    "initial_kernel_size": 61,
    "manual_pixels_per_micron": None,
    "apply_first_angle_filter": "no",
    "manual_detection": False,
    "expected_angle_deg": 90,
    "treshold_angle_init": 50,
    "final_kernel_size": 21,
    "min_pixels_in_rough_filtering": 100,
    "ini_dev_thresh": 60,
    "fin_dev_thresh": 20,
    "fiber_diameter_range": (400, 650),
    "fiber_spacing_range": (50, 400),
    "top_n_contours": 8,
    "vertical_step": 5,
    "max_distance_first_filter": 40,
    "angle_threshold_second_filter": 20,
    "max_distance_third_filter": 10
}

configs["C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/600um_50p_a/img30.tif"] = {
    "scale_bar_length": 508,
    "initial_kernel_size": 61,
    "manual_pixels_per_micron": None,
    "apply_first_angle_filter": "no",
    "manual_detection": False,
    "expected_angle_deg": 90,
    "treshold_angle_init": 50,
    "final_kernel_size": 21,
    "min_pixels_in_rough_filtering": 100,
    "ini_dev_thresh": 60,
    "fin_dev_thresh": 20,
    "fiber_diameter_range": (400, 650),
    "fiber_spacing_range": (50, 400),
    "top_n_contours": 8,
    "vertical_step": 5,
    "max_distance_first_filter": 40,
    "angle_threshold_second_filter": 20,
    "max_distance_third_filter": 10
}

configs["C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/600um_50p_a/img31.tif"] = {
    "scale_bar_length": 508,
    "initial_kernel_size": 61,
    "manual_pixels_per_micron": None,
    "apply_first_angle_filter": "no",
    "manual_detection": False,
    "expected_angle_deg": 90,
    "treshold_angle_init": 50,
    "final_kernel_size": 21,
    "min_pixels_in_rough_filtering": 100,
    "ini_dev_thresh": 60,
    "fin_dev_thresh": 20,
    "fiber_diameter_range": (400, 650),
    "fiber_spacing_range": (50, 400),
    "top_n_contours": 8,
    "vertical_step": 5,
    "max_distance_first_filter": 40,
    "angle_threshold_second_filter": 20,
    "max_distance_third_filter": 10
}

configs["C:/Users/dfpinedaquijan/surfdrive/PhD Project/Data/SEM/600um_50p_a/img32.tif"] = {
    "scale_bar_length": 508,
    "initial_kernel_size": 61,
    "manual_pixels_per_micron": None,
    "apply_first_angle_filter": "no",
    "manual_detection": False,
    "expected_angle_deg": 90,
    "treshold_angle_init": 50,
    "final_kernel_size": 21,
    "min_pixels_in_rough_filtering": 100,
    "ini_dev_thresh": 60,
    "fin_dev_thresh": 20,
    "fiber_diameter_range": (400, 650),
    "fiber_spacing_range": (50, 400),
    "top_n_contours": 4,
    "vertical_step": 5,
    "max_distance_first_filter": 40,
    "angle_threshold_second_filter": 20,
    "max_distance_third_filter": 10
}

# ------------------------------- Batch processing ---------------------------------------

# NOTE: Comment all lines below when testing single images

all_diameters = []
all_separations = []

for image_path, params in configs.items():
    print(f"\nProcessing {image_path}")
    diameters, separations = analyze_fibers(
        image_path=image_path,
        **params
    )
    all_diameters.extend(diameters)
    all_separations.extend(separations)


mean_all_d = np.mean(all_diameters)
mean_all_s = np.mean(all_separations)
pitch_all = mean_all_d + mean_all_s

sem_all_d = np.std(all_diameters, ddof=1) / np.sqrt(len(all_diameters))
sem_all_s = np.std(all_separations, ddof=1) / np.sqrt(len(all_separations))
pitch_all_error = np.sqrt(sem_all_d**2 + sem_all_s**2)

# Randomly sample pairs (min number of samples from both lists)
n_samples = min(len(all_diameters), len(all_separations))
sampled_d = np.random.choice(all_diameters, size=n_samples, replace=False)
sampled_s = np.random.choice(all_separations, size=n_samples, replace=False)

# Compute pitch = diameter + separation for each pair
pitches = sampled_d + sampled_s

# Get statistics
mean_pitch = np.mean(pitches)
std_pitch = np.std(pitches, ddof=1)

print(f"Combined average Fiber Diameter: {mean_all_d:.3f} µm")
print(f"Standard deviation Fiber Diameter: {np.std(all_diameters, ddof=1):.3f} µm")
print(f"Standard error of the mean fiber Diameter: {sem_all_d:.3f} µm")
print(f"Combined average Fiber Separation: {mean_all_s:.3f} µm")
print(f"Standard deviation Fiber Separation: {np.std(all_separations, ddof=1):.3f} µm")
print(f"Standard error of the mean fiber Separation: {sem_all_s:.3f} µm")
print(f'Combined pitch: {np.mean(all_diameters) + np.mean(all_separations):.3f} µm')
print(f"Pitch = {pitch_all:.2f} ± {pitch_all_error:.2f} µm")
print(f"Pitch = {mean_pitch:.2f} ± {std_pitch:.2f} µm")

# print(f"\nCombined average diameter: {np.mean(all_diameters):.3f} µm")
# print(f'Combined standard deviation diameter: {np.std(all_diameters, ddof=1):.3f} µm')
# print(f"Combined average separation: {np.mean(all_separations):.3f} µm")
# print(f'Combined standard deviation separation: {np.std(all_separations, ddof=1):.3f} µm')

plt.figure()
plt.hist(all_diameters, bins=30, edgecolor='black')
plt.title("Combined Fiber Diameters")
plt.xlabel("Diameter (µm)")
plt.ylabel("Frequency")
plt.show(block=False)
plt.pause(0.1)

plt.figure()
plt.hist(all_separations, bins=30, edgecolor='black')
plt.title("Combined Fiber Separations")
plt.xlabel("Separation (µm)")
plt.ylabel("Frequency")
plt.show()


