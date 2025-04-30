#! pip install git+https://github.com/facebookresearch/segment-anything.git
from skimage.metrics import structural_similarity as ssim
import torch
import os
import requests
from tqdm import tqdm
from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import segment_anything
print("SAM is installed successfully!")

class RGBDepthProcessor:
    def __init__(self):
        """Initialize the RGBDepthProcessor class."""
        pass
        
    def check_rgb_depth_alignment(self, rgb_image, depth_image, max_depth, threshold=0.75):
        """
        Checks the alignment of objects within an RGB image and a depth map using edge detection and SSIM.
        
        Parameters:
            rgb_image (numpy array): The RGB image.
            depth_image (numpy array): The depth map.
            max_depth (float): Maximum depth value (e.g., 3m or 1.5m).
            threshold (float): The SSIM threshold to determine alignment.
            - Currently set to 0.75 as the default.
        
        Returns:
            tuple: (Rounded SSIM score, 'Aligned' or 'Misaligned')
        """

        # Resize RGB to match Depth Map size
        depth_h, depth_w = depth_image.shape[:2]
        rgb_image = cv2.resize(rgb_image, (depth_w, depth_h), interpolation=cv2.INTER_LINEAR)

        # Convert RGB image to grayscale
        rgb_gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        # Ensure depth values are within the specified max_depth range
        depth_image = np.clip(depth_image, 0, max_depth)

        # Convert depth map to 8-bit without normalization
        depth_8bit = (depth_image / max_depth * 255).astype(np.uint8)  # Scale based on max_depth

        # Apply Canny edge detection
        edges_rgb = cv2.Canny(rgb_gray, 50, 150)
        edges_depth = cv2.Canny(depth_8bit, 50, 150)

        # Compute SSIM between edge images
        similarity_index = ssim(edges_rgb, edges_depth)
        similarity_index = round(similarity_index, 2)  # Round SSIM to 2 decimal places

        # Determine alignment status
        alignment_status = "Aligned" if similarity_index > threshold else "Misaligned"

        return similarity_index, alignment_status

    def download_yolov8_checkpoint(self, model_name="yolov8x-seg.pt"):
        """
        Downloads the YOLOv8 instance segmentation checkpoint if it does not exist.
        """
        url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}"
        
        if not os.path.exists(model_name):
            print(f"Downloading {model_name}...")
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(model_name, 'wb') as f, tqdm(
                desc=model_name, total=total_size, unit='B', unit_scale=True, unit_divisor=1024
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    f.write(data)
                    bar.update(len(data))
            
            print(f"{model_name} downloaded successfully!")
        else:
            print(f"{model_name} already exists in the current directory.")

    def load_sam_model(self, sam_checkpoint="sam_vit_h_finetuned.pth", model_type="vit_h"):
        """Loads the SAM model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        return predictor

    def detect_person_bboxes(self, image, model_name="yolov8x-seg.pt"):
        """Detects persons in the image using YOLOv8 and returns bounding boxes."""
        model = YOLO(model_name)
        results = model(image)
        
        person_bboxes = []
        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                label = model.names[int(cls)]
                if "person" in label.lower():
                    person_bboxes.append(box.cpu().numpy())
        
        return np.array(person_bboxes) if person_bboxes else None

    def segment_using_sam(self, image, predictor, bbox):
        """Segments an object using SAM given an image and bounding box."""
        predictor.set_image(image)
        masks, _, _ = predictor.predict(box=bbox.reshape(1, 4), multimask_output=False)
        return masks[0]

    def classify_bboxes(self, bboxes):
        """Classifies bounding boxes into child and foot based on area size."""
        areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in bboxes]
        sorted_indices = np.argsort(areas)
        child_bbox = bboxes[sorted_indices[-1]]  # Largest bbox (child)
        foot_bbox = bboxes[sorted_indices[0]] if len(bboxes) > 1 else None  # Smallest bbox (foot)
        return child_bbox, foot_bbox

    def detect_and_segment(self, rgb, depth, model_path="yolov8x-seg.pt", max_depth=1.5):
        """
        Detects persons, segments the child and foot regions, and calculates the remaining region mask.
        """
        person_bboxes = self.detect_person_bboxes(rgb, model_path)
        
        if person_bboxes is None:
            rgb = np.flip(rgb, axis=1)
            depth = np.flip(depth, axis=1)
            person_bboxes = self.detect_person_bboxes(rgb, model_path)
        
        child_mask, foot_mask, remaining_mask = None, None, None
        
        if person_bboxes is not None and len(person_bboxes) > 0:
            predictor = self.load_sam_model()
            
            if len(person_bboxes) >= 2:
                child_bbox, foot_bbox = self.classify_bboxes(person_bboxes)
            else:
                child_bbox, foot_bbox = person_bboxes[0], None

            child_mask = self.segment_using_sam(rgb, predictor, child_bbox)
            foot_mask = self.segment_using_sam(rgb, predictor, foot_bbox) if foot_bbox is not None else None

            if foot_mask is not None:
                kernel = np.ones((3,3), np.uint8)  # Increased kernel size for better coverage
                foot_mask = cv2.dilate(foot_mask.astype(np.uint8), kernel, iterations=8)  # Increased iterations
                foot_mask = foot_mask.astype(bool)

            remaining_mask = 1 - child_mask
            if foot_mask is not None:
                remaining_mask -= foot_mask
                remaining_mask = np.clip(remaining_mask, 0, 1)
        
        return child_mask, foot_mask, remaining_mask

    def display_segmented_images(
        self, rgb, depth, inpainted_depth,
        child_mask=None, second_mask=None, third_mask=None,
        second_label="Second", third_label="Third",
        max_depth=3.0
    ):
        """
        Display RGB, original and inpainted depth maps, and segmentations for up to 3 masks.

        Args:
            rgb (np.ndarray): Original RGB image.
            depth (np.ndarray): Original depth map.
            inpainted_depth (np.ndarray): Inpainted depth map.
            child_mask (np.ndarray, optional): Mask for the child.
            second_mask (np.ndarray, optional): Mask for second entity (e.g. wall or remaining).
            third_mask (np.ndarray, optional): Mask for third entity (e.g. floor or foot).
            second_label (str): Label for second mask.
            third_label (str): Label for third mask.
            max_depth (float): Maximum depth value for visualization scaling.
        """

        def safe_rgb_mask(mask):
            return rgb * np.expand_dims(mask, -1) if mask is not None else np.zeros_like(rgb)

        def safe_depth_mask(depth_map, mask):
            return depth_map * mask if mask is not None else np.zeros_like(depth_map)

        def safe_show(ax, image, title, cmap=None, vmin=None, vmax=None):
            if image is not None:
                ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12)
            ax.set_title(title)
            ax.axis("off")

        gray_rgb = np.mean(rgb, axis=-1)

        fig, axes = plt.subplots(4, 5, figsize=(18, 16))

        # Row 0 – Overview
        safe_show(axes[0, 0], rgb, "Original RGB")
        safe_show(axes[0, 1], depth, "Original Depth", cmap="jet", vmin=0, vmax=max_depth)
        safe_show(axes[0, 2], gray_rgb, "Gray RGB", cmap="gray")
        axes[0, 3].axis("off")
        safe_show(axes[0, 4], inpainted_depth, "Inpainted Depth", cmap="jet", vmin=0, vmax=max_depth)

        # Row 1 – Child
        safe_show(axes[1, 0], safe_rgb_mask(child_mask), "Child RGB")
        safe_show(axes[1, 1], safe_depth_mask(depth, child_mask), "Child Depth", cmap="jet", vmin=0, vmax=max_depth)
        safe_show(axes[1, 2], child_mask, "Child Mask", cmap="gray")
        axes[1, 3].axis("off")
        safe_show(axes[1, 4], safe_depth_mask(inpainted_depth, child_mask), "Child Inpainted", cmap="jet", vmin=0, vmax=max_depth)

        # Row 2 – Second mask
        safe_show(axes[2, 0], safe_rgb_mask(second_mask), f"{second_label} RGB")
        safe_show(axes[2, 1], safe_depth_mask(depth, second_mask), f"{second_label} Depth", cmap="jet", vmin=0, vmax=max_depth)
        safe_show(axes[2, 2], second_mask, f"{second_label} Mask", cmap="gray")
        axes[2, 3].axis("off")
        safe_show(axes[2, 4], safe_depth_mask(inpainted_depth, second_mask), f"{second_label} Inpainted", cmap="jet", vmin=0, vmax=max_depth)

        # Row 3 – Third mask
        safe_show(axes[3, 0], safe_rgb_mask(third_mask), f"{third_label} RGB")
        safe_show(axes[3, 1], safe_depth_mask(depth, third_mask), f"{third_label} Depth", cmap="jet", vmin=0, vmax=max_depth)
        safe_show(axes[3, 2], third_mask, f"{third_label} Mask", cmap="gray")
        axes[3, 3].axis("off")
        safe_show(axes[3, 4], safe_depth_mask(inpainted_depth, third_mask), f"{third_label} Inpainted", cmap="jet", vmin=0, vmax=max_depth)

        plt.tight_layout()
        plt.show()

    def visualize_rgb_depth_overlay(self, rgb, depth, max_depth=1.5):
        """
        Visualize the original RGB, depth map, and overlay of both images.
        
        Parameters:
        - rgb: Original RGB image (H, W, 3)
        - depth: Depth map (H, W)
        - max_depth: Maximum depth value (for colormap normalization)
        """

        original_rgb = rgb  # Original RGB
        rgb_image = rgb     # Preprocessed RGB (can apply resizing or filters if needed)
        depth_image = depth # Depth map

        # Resize RGB to match Depth Map size
        depth_h, depth_w = depth_image.shape[:2]
        rgb_image_resized = cv2.resize(rgb_image, (depth_w, depth_h), interpolation=cv2.INTER_LINEAR)

        # Apply Jet colormap to Depth Map
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=255/max_depth), cv2.COLORMAP_JET)

        # Create overlay by blending RGB and Depth Map
        overlay = cv2.addWeighted(rgb_image_resized, 0.6, depth_colormap, 0.4, 0)

        # Display images in a grid
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Original RGB
        axes[0].imshow(original_rgb)
        axes[0].set_title("RGB")
        axes[0].axis("off")

        # Depth Map
        depth_plot = axes[1].imshow(depth_image, cmap="jet", vmin=0, vmax=max_depth)
        axes[1].set_title("Depth Map")
        axes[1].axis("off")
        
        cbar = plt.colorbar(depth_plot, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_ticks(np.linspace(0, max_depth, num=15))
        cbar.ax.set_yticklabels([f"{tick:.2f}" for tick in np.linspace(0, max_depth, num=15)])

        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay: RGB + Depth")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

    def separate_wall_and_floor_exclude_child(self, rgb, child_mask):
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor((rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb, cv2.COLOR_RGB2GRAY)

        # Step 2: Mask out child region (only process background)
        background_gray = gray.copy()
        background_gray[child_mask == 1] = 0  # Zero out child pixels

        # Step 3: Canny edge detection on background
        edges = cv2.Canny(background_gray, 50, 150)

        # Step 4: Hough Line Transform to find horizontal lines
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

        # Step 5: Find the strongest (longest) horizontal line
        strongest_line = None
        max_length = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 5:  # Approx horizontal
                    length = np.hypot(x2 - x1, y2 - y1)
                    if length > max_length:
                        max_length = length
                        strongest_line = (x1, y1, x2, y2)

        # Step 6: Determine wall-floor boundary based on strongest line
        if strongest_line is not None:
            _, y1, _, y2 = strongest_line
            boundary_row = (y1 + y2) // 2  # Average y-coordinates
        else:
            # Fallback to previous method if no horizontal line is found
            white_pixel_counts = np.sum(edges == 255, axis=1)
            boundary_row = np.argmax(white_pixel_counts)

        # Step 7: Generate wall and floor masks
        wall_mask = np.zeros_like(gray, dtype=np.uint8)
        floor_mask = np.zeros_like(gray, dtype=np.uint8)
        wall_mask[:boundary_row, :] = 1
        floor_mask[boundary_row:, :] = 1

        # Step 8: Exclude child from both wall and floor
        wall_mask[child_mask == 1] = 0
        floor_mask[child_mask == 1] = 0

        return wall_mask, floor_mask, boundary_row

    def inpaint_depth_by_rowwise_mean(self, depth, mask, max_depth=3.0):
        """
        Inpaint missing values in depth using row-wise IQR mean.
        - If >10% of row is missing, sample randomly in (mean-0.1, mean+0.1).
        - Otherwise, use normal IQR mean.
        - If the entire row is missing, check adjacent rows (left or right) for valid depth and adjust.
        """
        inpainted = depth.copy()
        height, width = depth.shape

        mask_valid = (mask > 0)
        depth_valid = (depth > 0) & (depth <= max_depth)
        overall_valid_mask = mask_valid & depth_valid
        overall_mask_mean = depth[overall_valid_mask].mean() if np.any(overall_valid_mask) else 0

        rng = np.random.default_rng()

        for row in range(height):
            row_mask = mask_valid[row, :]
            row_depth = depth[row, :]
            valid_row = row_mask & (row_depth > 0) & (row_depth <= max_depth)
            
            missing_row = row_mask & ((row_depth == 0) | (row_depth > max_depth))
            missing_ratio = np.sum(missing_row) / np.sum(row_mask) if np.sum(row_mask) > 0 else 1.0

            if np.any(valid_row):
                row_valid_depths = row_depth[valid_row]
                q1 = np.percentile(row_valid_depths, 25)
                q3 = np.percentile(row_valid_depths, 75)
                iqr_values = row_valid_depths[(row_valid_depths >= q1) & (row_valid_depths <= q3)]
                row_mean = iqr_values.mean() if len(iqr_values) > 0 else row_valid_depths.mean()

                if missing_ratio > 0.1:
                    # Sample randomly around the overall_mask_mean
                    low = overall_mask_mean - 0.1
                    high = overall_mask_mean + 0.1
                    sampled_values = rng.uniform(low, high, size=np.sum(missing_row))
                else:
                    sampled_values = np.full(np.sum(missing_row), row_mean)
            else:
                # Entire row invalid — check adjacent rows (left or right)
                above = row - 1 if row > 0 else None
                below = row + 1 if row < height - 1 else None
                means = []
                if above is not None:
                    valid_above = mask_valid[above, :] & (depth[above, :] > 0) & (depth[above, :] < max_depth)
                    if np.any(valid_above):
                        row_valid_above = depth[above, :][valid_above]
                        q1_above = np.percentile(row_valid_above, 25)
                        q3_above = np.percentile(row_valid_above, 75)
                        iqr_above = row_valid_above[(row_valid_above >= q1_above) & (row_valid_above <= q3_above)]
                        means.append(iqr_above.mean() if len(iqr_above) > 0 else row_valid_above.mean())
                if below is not None:
                    valid_below = mask_valid[below, :] & (depth[below, :] > 0) & (depth[below, :] < max_depth)
                    if np.any(valid_below):
                        row_valid_below = depth[below, :][valid_below]
                        q1_below = np.percentile(row_valid_below, 25)
                        q3_below = np.percentile(row_valid_below, 75)
                        iqr_below = row_valid_below[(row_valid_below >= q1_below) & (row_valid_below <= q3_below)]
                        means.append(iqr_below.mean() if len(iqr_below) > 0 else row_valid_below.mean())

                if means:
                    row_mean = np.mean(means)
                else:
                    row_mean = overall_mask_mean

                sampled_values = rng.uniform(row_mean - 0.1, row_mean + 0.1, size=np.sum(missing_row))

            missing_indices = np.where(missing_row)[0]
            inpainted[row, missing_indices] = sampled_values

        return inpainted

    def inpaint_depth_by_columnwise_mean(self, depth, mask, max_depth=1.5):
        """
        Inpaint missing values in depth using column-wise IQR mean for lying children.
        - If >10% of column is missing, sample randomly in (mean-0.05, mean+0.05).
        - Otherwise, use normal IQR mean.
        - If the entire column is missing, check adjacent columns (left or right) for valid depth and adjust.
        """
        inpainted = depth.copy()
        height, width = depth.shape

        mask_valid = (mask > 0)
        depth_valid = (depth > 0) & (depth <= max_depth)
        overall_valid_mask = mask_valid & depth_valid
        overall_mask_mean = depth[overall_valid_mask].mean() if np.any(overall_valid_mask) else 0

        rng = np.random.default_rng()

        for col in range(width):
            col_mask = mask_valid[:, col]
            col_depth = depth[:, col]
            valid_col = col_mask & (col_depth > 0) & (col_depth <= max_depth)

            missing_col = col_mask & ((col_depth == 0) | (col_depth > max_depth))
            missing_ratio = np.sum(missing_col) / np.sum(col_mask) if np.sum(col_mask) > 0 else 1.0

            if np.any(valid_col):
                col_valid_depths = col_depth[valid_col]
                q1 = np.percentile(col_valid_depths, 25)
                q3 = np.percentile(col_valid_depths, 75)
                iqr_values = col_valid_depths[(col_valid_depths >= q1) & (col_valid_depths <= q3)]
                col_mean = iqr_values.mean() if len(iqr_values) > 0 else col_valid_depths.mean()

                if missing_ratio > 0.1:
                    # Sample randomly around the overall_mask_mean
                    low = overall_mask_mean - 0.05
                    high = overall_mask_mean + 0.05
                    sampled_values = rng.uniform(low, high, size=np.sum(missing_col))
                else:
                    sampled_values = np.full(np.sum(missing_col), col_mean)
            else:
                # Entire column invalid — check adjacent columns (left or right)
                left = col - 1 if col > 0 else None
                right = col + 1 if col < width - 1 else None
                means = []
                if left is not None:
                    valid_left = mask_valid[:, left] & (depth[:, left] > 0) & (depth[:, left] < max_depth)
                    if np.any(valid_left):
                        left_depths = depth[:, left][valid_left]
                        q1_left = np.percentile(left_depths, 25)
                        q3_left = np.percentile(left_depths, 75)
                        iqr_left = left_depths[(left_depths >= q1_left) & (left_depths <= q3_left)]
                        means.append(iqr_left.mean() if len(iqr_left) > 0 else left_depths.mean())
                if right is not None:
                    valid_right = mask_valid[:, right] & (depth[:, right] > 0) & (depth[:, right] < max_depth)
                    if np.any(valid_right):
                        right_depths = depth[:, right][valid_right]
                        q1_right = np.percentile(right_depths, 25)
                        q3_right = np.percentile(right_depths, 75)
                        iqr_right = right_depths[(right_depths >= q1_right) & (right_depths <= q3_right)]
                        means.append(iqr_right.mean() if len(iqr_right) > 0 else right_depths.mean())

                if means:
                    col_mean = np.mean(means)
                else:
                    col_mean = overall_mask_mean

                sampled_values = rng.uniform(col_mean - 0.05, col_mean + 0.05, size=np.sum(missing_col))

            missing_indices = np.where(missing_col)[0]
            inpainted[missing_indices, col] = sampled_values

        return inpainted

    def inpaint_depth_by_interpolation(self, depth, child_mask, max_depth):
        """
        Inpaint depth values using interpolation within the child mask area.

        Parameters:
            depth (np.ndarray): The depth map to inpaint.
            child_mask (np.ndarray): A binary mask where non-zero values represent the region to inpaint.
            max_depth (float): Maximum depth value to constrain inpainting.

        Returns:
            np.ndarray: Inpainted depth map.
        """
        # Ensure the mask is a binary mask (0 or 255)
        child_mask = np.uint8(child_mask)
        
        # Create the inverse mask (non-child areas)
        inverse_mask = cv2.bitwise_not(child_mask)

        # Inpainting using OpenCV's inpainting function (method = 1 for inpainting using telea)
        inpainted_depth = cv2.inpaint(depth, child_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        
        # Apply max depth constraint (ensuring the depth values do not exceed max_depth)
        inpainted_depth = np.minimum(inpainted_depth, max_depth)
        
        return inpainted_depth

    def inpaint_depth_all_masks(self, depth, pose_type, child_mask, floor_mask=None, wall_mask=None, foot_mask=None, max_depth=3.0, foot_area_threshold=0.3):
        """
        Generalized depth inpainting function for both lying and standing children.
        """
        # Calculate percentage of missing values in child mask
        child_mask_missing_percentage = np.sum(child_mask == 0) / child_mask.size * 100

        if pose_type == "standing":
            # If child mask has more than 10% missing values, perform row-wise inpainting; otherwise, use interpolation
            if child_mask_missing_percentage > 10:
                depth = self.inpaint_depth_by_rowwise_mean(depth, child_mask, max_depth)
            else:
                depth = self.inpaint_depth_by_interpolation(depth, child_mask, max_depth)
            # Inpainting for standing children: Use row-wise mean for floor_mask, and wall_mask
            if floor_mask is not None:
                depth = self.inpaint_depth_by_rowwise_mean(depth, floor_mask, max_depth)
            if wall_mask is not None:
                depth = self.inpaint_depth_by_rowwise_mean(depth, wall_mask, max_depth)

        elif pose_type == "lying":
            # If child mask has more than 10% missing values, perform column-wise inpainting; otherwise, use interpolation
            if child_mask_missing_percentage > 10:
                depth = self.inpaint_depth_by_columnwise_mean(depth, child_mask, max_depth)
            else:
                depth = self.inpaint_depth_by_interpolation(depth, child_mask, max_depth)
            # Inpainting for lying children: Use column-wise mean for child_mask and floor_mask
            if floor_mask is not None:
                depth = self.inpaint_depth_by_columnwise_mean(depth, floor_mask, max_depth=max_depth)
            # Foot region inpainting for lying children using floor_mask column mean
            if foot_mask is not None:
                foot_mask = (foot_mask == 1).astype(np.uint8)
                foot_area = np.sum(foot_mask)
                # Check if foot area is smaller than a threshold
                child_area = np.sum(child_mask)
                if foot_area < foot_area_threshold * child_area:
                    # Inpaint using the floor mask (foot region is too small)
                    depth[foot_mask == 1] = 0
                    remaining_mask = 1 - child_mask
                    depth = self.inpaint_depth_by_columnwise_mean(depth, remaining_mask, max_depth=max_depth)
                    print("foot inpaint")
                else:
                    # If foot area is too large, return None (image not usable)
                    print("foot inpaint not possible")
                    return None  # Image not usable due to too much overlap.
        return depth

    def overlay_mask(self, image, mask, color, alpha=0.3):
        # Convert grayscale/depth to 3-channel if needed
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        overlay = image.copy()
        for c in range(3):
            overlay[..., c] = np.where(mask,
                                       (1 - alpha) * overlay[..., c] + alpha * color[c] * 255,
                                       overlay[..., c])
        return np.clip(overlay, 0, 255).astype(np.uint8)

    def plot_with_masks_on_image(self, image, wall_mask, floor_mask, child_mask, foot_mask=None,
                                 is_depth=False, is_standing=True, alpha=0.3):
        # Normalize depth to [0, 255] if it's depthmap
        if is_depth and image.ndim == 2:
            image = 255 * (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-5)
            image = image.astype(np.uint8)

        # Define light colors
        wall_color = (0.4, 0.7, 1.0)
        floor_color = (0.6, 1.0, 0.6)
        child_color = (1.0, 0.7, 0.8)
        foot_color = (0.9, 0.9, 0.1)

        # Start with the child and floor masks
        combined = self.overlay_mask(image, floor_mask, floor_color, alpha)
        combined = self.overlay_mask(combined, child_mask, child_color, alpha)

        # For standing child, overlay wall_mask, foot_mask is ignored
        if is_standing:
            combined = self.overlay_mask(combined, wall_mask, wall_color, alpha)
        elif foot_mask is not None:
            combined = self.overlay_mask(combined, foot_mask, foot_color, alpha)

        # Add labels conditionally
        def add_label(mask, label, color):
            pos = np.argwhere(mask)
            if len(pos) > 0:
                y, x = pos[len(pos)//2]
                ax.text(x, y, label, color=color, fontsize=12, weight='bold',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

        # Display the result
        ax = plt.gca()
        add_label(child_mask, "Child", "red")
        add_label(floor_mask, "Floor", "green")

        if is_standing:
            add_label(wall_mask, "Wall", "blue")
        if foot_mask is not None:
            add_label(foot_mask, "Foot", "yellow")

        plt.axis('off')
        plt.title("Overlay of Masks on " + ("Depthmap" if is_depth else "RGB"))
        plt.tight_layout()

        # Return the overlaid image
        return combined
'''
# Sample Usage

# Instantiate
proc = RGBDepthProcessor()

# Alignment check
score, status = proc.check_rgb_depth_alignment(rgb, depth, max_depth=max_depth)
print(f"Alignment: {status} (SSIM={score})")

# Full detect + segment pipeline
child_mask, foot_mask, floor_mask = proc.detect_and_segment(rgb, depth)

# Wall/floor separation for standing child
if child_position == "standing":
    wall_mask, floor_mask, boundary_row = proc.separate_wall_and_floor_exclude_child(rgb, child_mask)

# Inpainting
if child_position == "lying":
    depth_inpainted = proc.inpaint_depth_all_masks(depth, pose_type=child_position, child_mask=child_mask, floor_mask=floor_mask, foot_mask=foot_mask, max_depth=max_depth)
else:
    depth_inpainted = proc.inpaint_depth_all_masks(depth, pose_type=child_position, child_mask=child_mask, floor_mask=floor_mask, wall_mask=wall_mask, max_depth=max_depth)

# Display segmented and inpainted images
if child_position == "lying":
    if depth_inpainted is not None:
        proc.display_segmented_images(
            rgb, depth, depth_inpainted,
            child_mask=child_mask,
            second_mask=floor_mask,
            third_mask=foot_mask,
            second_label="Floor",
            third_label="Foot",
            max_depth=max_depth
        )
    else:
        print("Another person holding child")
else:
    proc.display_segmented_images(
        rgb, depth, depth_inpainted,
        child_mask=child_mask,
        second_mask=floor_mask,
        third_mask=wall_mask, 
        second_label="Floor",
        third_label="Wall",
        max_depth=max_depth
    )


# RGB overlay visualization
if child_position == "lying":
    overlaid_image_rgb = proc.plot_with_masks_on_image(rgb, None, floor_mask, child_mask, foot_mask=foot_mask, is_depth=False, is_standing=False)
else:
    overlaid_image_rgb = proc.plot_with_masks_on_image(rgb, wall_mask, floor_mask, child_mask, is_depth=False, is_standing=True)
plt.imshow(overlaid_image_rgb)
plt.title("Overlay RGB")
plt.axis('off')
plt.show()
'''