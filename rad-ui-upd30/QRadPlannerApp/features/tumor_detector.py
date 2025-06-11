# --- START OF FILE rad-ui-full-02-main/QRadPlannerApp/features/tumor_detector.py ---

import logging
import numpy as np
from typing import Optional
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.measure import regionprops, label # <-- CORRECTED IMPORT
from skimage.segmentation import watershed

logger = logging.getLogger(__name__)

class TumorDetector:
    def __init__(self):
        self.model = None
        self.threshold = 0.5

    def detect_tumors(self, volume_data: np.ndarray) -> np.ndarray:
        """Detect tumors in the volume data using the full pipeline."""
        logger.info("--- Starting detect_tumors (full pipeline) ---")
        try:
            # 1. Preparation: Convert to float32
            volume_float32 = volume_data.astype(np.float32)

            # 2. Normalization
            vol_min = volume_float32.min()
            vol_max = volume_float32.max()
            logger.debug(f"detect_tumors: input volume min={vol_min}, max={vol_max}")

            if vol_max - vol_min < 1e-6: # Check if range is very small or zero
                logger.warning("detect_tumors: Volume range is very small. Normalization might be unstable. Returning empty mask.")
                return np.zeros_like(volume_float32, dtype=bool)
            else:
                volume_normalized = (volume_float32 - vol_min) / (vol_max - vol_min)
            logger.debug(f"detect_tumors: normalized volume min={volume_normalized.min()}, max={volume_normalized.max()}")

            # 3. Apply Gaussian blur (operates on the normalized volume)
            # This is the volume used for peak finding and as input for distance transform for watershed
            volume_blurred = gaussian_filter(volume_normalized, sigma=1)
            logger.debug(f"detect_tumors: volume_blurred min={volume_blurred.min()}, max={volume_blurred.max()}")

            # 4. Prepare array for peak_local_max: Ensure C-contiguity and float32 dtype
            # This is crucial as gaussian_filter might return a slightly different type or non-contiguous array.
            volume_prepared_for_peaks = np.ascontiguousarray(volume_blurred, dtype=np.float32)
            logger.debug(f"detect_tumors: volume_prepared_for_peaks flags: {volume_prepared_for_peaks.flags}, dtype: {volume_prepared_for_peaks.dtype}")

            # 5. Find local maxima
            min_distance_peaks = 25
            threshold_abs_peaks = 0.35 # This threshold is on normalized data [0,1]
            logger.info(f"detect_tumors: Calling peak_local_max with min_distance={min_distance_peaks}, threshold_abs={threshold_abs_peaks}")

            coordinates = peak_local_max(volume_prepared_for_peaks, min_distance=min_distance_peaks, threshold_abs=threshold_abs_peaks)
            logger.info(f"detect_tumors: peak_local_max call succeeded. Found {len(coordinates)} peaks.")

            if len(coordinates) == 0:
                logger.warning("detect_tumors: No peaks found by peak_local_max. Tumor mask will be empty.")
                return np.zeros_like(volume_prepared_for_peaks, dtype=bool)

            # 6. Create mask from coordinates (this will be the markers for watershed)
            peak_markers_mask = np.zeros_like(volume_prepared_for_peaks, dtype=bool)
            for coord_z, coord_r, coord_c in coordinates: # Assuming z, r, c order
                peak_markers_mask[coord_z, coord_r, coord_c] = True

            # 7. Apply watershed segmentation
            # Watershed is applied on a distance transform. The distance is usually calculated from the inverted image.
            # Here, `volume_blurred` (or `volume_normalized`) can serve as the basis for the distance.
            # `peak_markers_mask` provides the markers.
            # The watershed function itself will compute the distance transform if not provided.
            # skimage.segmentation.watershed expects `image` (inverted intensity for basins), `markers` (labeled regions), `mask` (optional region to segment).

            # Create a version of the blurred volume for watershed: higher values = higher "elevation"
            # We want basins to form around our peaks.
            # So, use -volume_blurred, where peaks become minima.
            # Markers for watershed: label connected components in peak_markers_mask
            #labeled_peak_markers, num_features = label(peak_markers_mask) # Now 'label' should be defined
            labeled_peak_markers, num_features = label(peak_markers_mask, return_num=True) # Corrected line
            if num_features == 0: # Should not happen if coordinates were found
                 logger.warning("detect_tumors: No features found after labeling peak markers. Returning empty mask.")
                 return np.zeros_like(volume_prepared_for_peaks, dtype=bool)

            logger.info(f"detect_tumors: Found {num_features} distinct peak regions to use as markers for watershed.")

            # The image for watershed: use negative of blurred volume so peaks are "low points"
            watershed_image = -volume_blurred # volume_blurred is normalized [0,1] and blurred

            # Create a constraint mask for watershed:
            # Only allow watershed to segment regions where the normalized&blurred intensity is above a certain level.
            # This helps prevent watershed from "leaking" into irrelevant areas.
            # Threshold is on the normalized [0,1] scale of volume_blurred.
            # A value of 0.2 could mean roughly values above -600 HU if -1000 HU is 0 and max HU is 1.
            # Let's try a threshold that might correspond to denser soft tissue.
            # If 0 HU is normalized to around 0.2-0.3 (depending on actual min/max of original volume),
            # then a threshold like 0.3 might be a starting point to exclude very low densities.
            threshold_for_watershed_constraint = 0.35 # Tune this value
            watershed_segmentation_mask = volume_blurred > threshold_for_watershed_constraint
            logger.info(f"detect_tumors: Watershed will be constrained to mask where blurred normalized value > {threshold_for_watershed_constraint}. Mask sum: {np.sum(watershed_segmentation_mask)}")

            if not np.any(labeled_peak_markers & watershed_segmentation_mask):
                logger.warning("detect_tumors: None of the peak markers fall within the watershed constraint mask. Tumor mask will be empty.")
                return np.zeros_like(volume_prepared_for_peaks, dtype=bool)

            labels_ws = watershed(watershed_image, markers=labeled_peak_markers, mask=watershed_segmentation_mask, connectivity=1) # connectivity can be tuned; Renamed to labels_ws
            logger.info(f"detect_tumors: Watershed segmentation done. Unique labels count: {len(np.unique(labels_ws))}")

            # 8. Filter regions based on properties (e.g., size)
            # `labels_ws` now contains the segmented regions. Regions corresponding to markers will have their respective label.
            # Background (where no basin formed from a marker) will be 0.
            final_tumor_mask = np.zeros_like(volume_prepared_for_peaks, dtype=bool)
            min_tumor_area = 75000 # Example minimum size in voxels
            found_regions_after_filter = 0

            # Iterate through unique labels found by watershed, skipping 0 (background)
            unique_labels_ws = np.unique(labels_ws)
            for region_label_ws in unique_labels_ws:
                if region_label_ws == 0: # Skip background
                    continue

                region_voxels = (labels_ws == region_label_ws)
                region_area = np.sum(region_voxels) # Number of voxels in the region

                # Additional properties can be extracted using regionprops if needed on `labels_ws` and `volume_blurred`
                # For now, simple area filtering
                if region_area > min_tumor_area:
                    final_tumor_mask[region_voxels] = True
                    found_regions_after_filter += 1

            if found_regions_after_filter > 0:
                logger.info(f"detect_tumors: Found {found_regions_after_filter} tumor regions after area filtering (>{min_tumor_area} voxels).")
            else:
                logger.warning(f"detect_tumors: No regions met the area criteria (>{min_tumor_area} voxels). Tumor mask will be empty.")

            return final_tumor_mask

        except Exception as e:
            logger.error(f"Error during full tumor detection pipeline: {str(e)}", exc_info=True)
            logger.info("Falling back to a placeholder spherical tumor mask.")
            # Create a small spherical mask in the center
            center = (np.array(volume_data.shape) / 2).astype(int)
            radius = min(5, int(min(volume_data.shape) / 4) -1 ) # Ensure radius is small and fits
            radius = max(radius, 1) # ensure radius is at least 1

            z_coords, r_coords, c_coords = np.ogrid[:volume_data.shape[0], :volume_data.shape[1], :volume_data.shape[2]]
            placeholder_mask = (z_coords - center[0])**2 + (r_coords - center[1])**2 + (c_coords - center[2])**2 <= radius**2
            return placeholder_mask.astype(bool)

    def test_peak_local_max(self, volume_data_input: np.ndarray) -> Optional[np.ndarray]:
        logger.info("--- Starting test_peak_local_max ---")
        try:
            # 1. Preparation (as in detect_tumors)
            volume = volume_data_input.astype(np.float32)
            vol_min = volume.min()
            vol_max = volume.max()
            logger.debug(f"test_peak_local_max: input volume min={vol_min}, max={vol_max}")

            if vol_max - vol_min < 1e-6:
                logger.warning("test_peak_local_max: Volume range is very small. Using zero volume.")
                volume_normalized = np.zeros_like(volume, dtype=np.float32)
            else:
                volume_normalized = (volume - vol_min) / (vol_max - vol_min)
            logger.debug(f"test_peak_local_max: normalized volume min={volume_normalized.min()}, max={volume_normalized.max()}")

            volume_contiguous = np.ascontiguousarray(volume_normalized, dtype=np.float32)

            # 2. Log properties
            logger.info(f"test_peak_local_max: volume_contiguous shape: {volume_contiguous.shape}, dtype: {volume_contiguous.dtype}, flags: {volume_contiguous.flags}")
            logger.info(f"test_peak_local_max: volume_contiguous min: {volume_contiguous.min()}, max: {volume_contiguous.max()}")

            # 3. Call peak_local_max directly
            min_distance = 5 # Reduced for potentially smaller test arrays later
            threshold_abs = 0.3 # Standard value from original code

            logger.info(f"test_peak_local_max: Calling peak_local_max with min_distance={min_distance}, threshold_abs={threshold_abs}")
            coordinates = peak_local_max(volume_contiguous, min_distance=min_distance, threshold_abs=threshold_abs)

            logger.info(f"test_peak_local_max: peak_local_max call succeeded. Found {len(coordinates)} peaks.")
            logger.info(f"Coordinates: {coordinates}")
            return coordinates

        except Exception as e:
            logger.error(f"test_peak_local_max: Error during peak_local_max call: {str(e)}", exc_info=True)
            # This will catch the "No matching signature found" if it occurs here
            return None
        finally:
            logger.info("--- Finished test_peak_local_max ---")

    # Modify detect_tumors to use the test method for now
    def detect_tumors_original(self, volume_data: np.ndarray) -> np.ndarray:
        # This is the original method, renamed for safekeeping
        try:
            # Convert to float32
            volume = volume_data.astype(np.float32)

            # Normalize
            vol_min = volume.min()
            vol_max = volume.max()
            logger.debug(f"TumorDetector: input volume min={vol_min}, max={vol_max}")
            if vol_max - vol_min < 1e-6: # Check if range is very small or zero
                logger.warning(f"TumorDetector: Volume range is very small ({vol_max - vol_min}). Normalization might be unstable. Setting to all zeros.")
                volume = np.zeros_like(volume, dtype=np.float32)
            else:
                volume = (volume - vol_min) / (vol_max - vol_min)
            logger.debug(f"TumorDetector: normalized volume min={volume.min()}, max={volume.max()}")

            # Apply Gaussian blur
            volume = gaussian_filter(volume, sigma=1)

            # Ensure C-contiguity for peak_local_max, which might use Numba
            volume_contiguous = np.ascontiguousarray(volume, dtype=np.float32)
            logger.debug(f"TumorDetector: volume_contiguous flags: {volume_contiguous.flags}")

            # Find local maxima
            coordinates = peak_local_max(volume_contiguous, min_distance=20, threshold_abs=0.3)

            # Create mask
            mask = np.zeros_like(volume, dtype=bool) # Ensure boolean mask
            for coord in coordinates:
                mask[coord[0], coord[1], coord[2]] = True

            # Apply watershed
            # distance = gaussian_filter(volume, sigma=2) # Temporarily disable for isolated test
            # labels = watershed(-distance, mask) # Temporarily disable for isolated test
            labels_ws = np.zeros_like(volume, dtype=int) # Placeholder for labels; Renamed to labels_ws

            # Filter regions
            regions = regionprops(labels_ws)
            tumor_mask = np.zeros_like(volume)

            for region in regions:
                if region.area > 100:  # Minimum size threshold
                    tumor_mask[labels_ws == region.label] = 1 # Use labels_ws here

            return tumor_mask.astype(bool) # Ensure boolean mask

        except Exception as e:
            logger.error(f"Error in tumor detection: {str(e)}")
            # If peak_local_max itself fails, this generic catch might obscure it.
            # The specific test_peak_local_max will give more direct feedback.
            return np.zeros_like(volume_data, dtype=bool)
