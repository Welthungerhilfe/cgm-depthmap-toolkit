import numpy as np
from skimage.restoration import inpaint


def replace_values_above_threshold(depth_map, threshold, expected_shape=(240, 180, 1)):
    """
    Replace depth values in a depth map above a given threshold with the average
    of their four non-zero neighbors.

    Args:
    - depth_map (numpy.ndarray): The input depth map.
    - threshold (float): The threshold value.
    - expected_shape (tuple): The expected shape of the input depth map.

    Returns:
    - numpy.ndarray: The depth map with values above the threshold replaced by
                    the average of their four non-zero neighbors.
    """
    # Ensure depth_map has the expected shape
    if depth_map.shape != expected_shape:
        raise ValueError(f"Input depth_map should have a shape of {expected_shape}")

    # Create a binary mask for values above the threshold
    above_threshold_mask = depth_map > threshold

    # Get the indices of the values above the threshold
    above_threshold_indices = np.argwhere(above_threshold_mask)

    # Iterate over the indices and replace values with the average of neighbors
    for i, j, k in above_threshold_indices:
        neighbors = [
            depth_map[i - 1, j, k],  # Top neighbor
            depth_map[i + 1, j, k],  # Bottom neighbor
            depth_map[i, j - 1, k],  # Left neighbor
            depth_map[i, j + 1, k],  # Right neighbor
        ]
        
        # Filter out zero values
        non_zero_neighbors = [neighbor for neighbor in neighbors if neighbor != 0]

        if non_zero_neighbors:
            # Calculate the average of non-zero neighbors
            avg_neighbor_value = np.mean(non_zero_neighbors)
            depth_map[i, j, k] = avg_neighbor_value
        else:
            # If all neighbors are zero, set the value to zero
            depth_map[i, j, k] = 0

    return depth_map


def fill_zeros_inpainting(depth_map):
    """
    Fill zero values in a depth map using inpainting.

    Uses biharmonic interpolation to fill in missing or undefined values in a depth map by generating a smooth and continuous surface that adheres to the available depth values. Biharmonic interpolation is a mathematical technique that extends the concept of harmonic functions to interpolate values over a surface or within a region.

    Args:
        depth_map (numpy.ndarray): Input depth map with zero values.

    Returns:
        numpy.ndarray: Depth map with zero values filled using inpainting.
    """
    # Apply inpainting to fill in zero values using biharmonic interpolation
    depth_map_filled = inpaint.inpaint_biharmonic(depth_map, mask=(depth_map == 0))
    
    return depth_map_filled
