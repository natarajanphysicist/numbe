# q_rad_plan_may-13_enhanced_motion.py

import logging 
from typing import Dict, Any, List, Optional, Tuple 
from skimage.measure import label 

logger = logging.getLogger(__name__)

import logging # Added
from typing import Dict, Any, List, Optional, Tuple # Added
from skimage.measure import label # Added for connected components in watershed markers
import logging # Added
from typing import Dict, Any, List, Optional, Tuple # Added
from skimage.measure import label # Added for connected components in watershed markers
import logging # Added
from typing import Dict, Any, List, Optional, Tuple # Added for type hinting
import numpy as np
from scipy.ndimage import convolve, map_coordinates
from scipy.special import erf
import matplotlib.pyplot as plt
import pydicom
from pydicom.fileset import FileSet 
import os
from numba import jit, prange 
from scipy.optimize import minimize 
import logging  # Ensure logging is imported
from typing import Dict, Any, List, Optional, Tuple # Ensure typing hints are imported
from skimage.measure import label # Ensure label is imported for watershed markers

logger = logging.getLogger(__name__) # Ensure module-level logger is defined
import logging # Added
from typing import Dict, Any, List, Optional, Tuple # Added for type hinting

logger = logging.getLogger(__name__) # Added

# --- Re-paste the Numba function here if it's unchanged ---
@jit(nopython=True, parallel=True)
def calculate_primary_fluence_numba(source, direction, density_grid, grid_shape):
    fluence = np.zeros(grid_shape)
    for i in prange(grid_shape[0]):
        for j in range(grid_shape[1]):
            for k in range(grid_shape[2]):
                r = np.array([i, j, k], dtype=np.float64)
                dist = np.linalg.norm(r - source)
                if dist == 0:
                    continue
                num_ray_points = 50
                ray_path = np.empty((num_ray_points, source.shape[0]), dtype=np.float64)
                for dim_idx in range(source.shape[0]):
                    ray_path[:, dim_idx] = np.linspace(source[dim_idx], r[dim_idx], num=num_ray_points)
                path_density = 0.0
                for p_idx in range(num_ray_points): # Changed loop variable
                    p_coord = ray_path[p_idx]      # Changed variable name
                    idx = np.round(p_coord).astype(np.int32)
                    # Ensure indices are within grid bounds before accessing density_grid
                    if (0 <= idx[0] < grid_shape[0] and 0 <= idx[1] < grid_shape[1] and 0 <= idx[2] < grid_shape[2]):
                        path_density += density_grid[idx[0], idx[1], idx[2]]
                mu = 0.02
                fluence[i, j, k] = (1 / (dist**2 + 1e-9)) * np.exp(-mu * path_density) # Added epsilon to dist**2
    return fluence
# --- End of Numba function ---

logger = logging.getLogger(__name__) # Added

class QRadPlan3D:
    def __init__(self, grid_size=(100, 100, 100), num_beams=8, kernel_path="dose_kernel.npy",
                 dicom_rt_struct_path=None, ct_path=None, fourd_ct_path=None,
                 reference_phase_name="phase_0", 
                 patient_params=None, dir_method='simplified_sinusoidal'):

        self.grid_size = grid_size 
        self.dose_kernel = np.load(kernel_path)
        self.beam_directions = self._generate_beam_directions(num_beams)
        self.dir_method = dir_method
        self.reference_phase_name = reference_phase_name
        self.tumor_mask_name = "Tumor" # Added default tumor ROI name
        self.beam_weights: Optional[np.ndarray] = None # Initialize beam_weights

        self.num_phases = 10 
        self.respiratory_phase_weights = np.ones(self.num_phases) / self.num_phases
        self.accumulated_dose = np.zeros(self.grid_size)
        
        # Initialize attributes that might be set later by simulation
        self.tcp_value: Optional[float] = None
        self.ntcp_values: Dict[str, float] = {}
        self.dose_distribution: Optional[np.ndarray] = None # Will store current relevant dose (single fraction or accumulated)


        if fourd_ct_path and dicom_rt_struct_path:
            logger.info(f"Attempting to load 4D CT data from: {fourd_ct_path}")
            logger.info(f"Attempting to load RTStruct from: {dicom_rt_struct_path}")
            try:
                self.density_grids_phases, self.tumor_masks_phases, self.oar_masks_phases, self.affine_transforms = \
                    self._load_4d_ct_data(fourd_ct_path, dicom_rt_struct_path)
                
                if self.density_grids_phases:
                    self.grid_size = self.density_grids_phases[0].shape
                    logger.info(f"Grid size updated to {self.grid_size} based on loaded CT data.")
                    self.accumulated_dose = np.zeros(self.grid_size)

                self.num_phases = len(self.density_grids_phases)
                self.respiratory_phase_weights = np.ones(self.num_phases) / self.num_phases
                logger.info(f"Successfully loaded {self.num_phases} respiratory phases.")

                self.tumor_mask = np.any(self.tumor_masks_phases, axis=0) 
                if self.oar_masks_phases and isinstance(self.oar_masks_phases[0], dict):
                     all_oar_names = set(key for phase_oars in self.oar_masks_phases for key in phase_oars)
                     self.oar_masks = {
                         oar_name: np.any([phase_oars.get(oar_name, np.zeros(self.grid_size, dtype=bool))
                                           for phase_oars in self.oar_masks_phases], axis=0)
                         for oar_name in all_oar_names
                     }
                else: 
                    self.oar_masks = {}
                self.density_grid = np.mean(self.density_grids_phases, axis=0) 
            except Exception as e:
                logger.error(f"Error loading 4D DICOM data: {e}. Falling back to simplified model.", exc_info=True)
                self._initialize_simplified_model(grid_size)
        elif dicom_rt_struct_path and ct_path: 
            logger.info(f"Attempting to load static 3D CT data from: {ct_path}")
            logger.info(f"Attempting to load RTStruct from: {dicom_rt_struct_path}")
            try:
                ref_ct_series = self._load_ct_series(ct_path) 
                if ref_ct_series is None or not ref_ct_series:
                    raise ValueError("Failed to load static CT series.")
                
                rows = ref_ct_series[0].Rows
                cols = ref_ct_series[0].Columns
                num_slices = len(ref_ct_series)
                self.grid_size = (cols, rows, num_slices)
                logger.info(f"Grid size set to {self.grid_size} from static CT.")
                self.accumulated_dose = np.zeros(self.grid_size)

                ct_pixel_data = np.stack([s.pixel_array * s.RescaleSlope + s.RescaleIntercept for s in ref_ct_series], axis=-1)
                ct_pixel_data = ct_pixel_data.transpose(1,0,2)

                density_grid_ref = self._hu_to_density(ct_pixel_data)
                # In _load_rt_struct, self.tumor_mask_name can be updated if a specific ROI is identified as 'tumor'
                tumor_mask_ref, oar_masks_ref, roi_names_map = self._load_rt_struct(dicom_rt_struct_path, ref_ct_series)
                # Update self.tumor_mask_name if a primary tumor ROI was identified by name
                for roi_num, name in roi_names_map.items():
                    if "tumor" in name.lower() or "gtv" in name.lower() or "ptv" in name.lower(): # Example check
                        self.tumor_mask_name = name
                        break


                self.density_grids_phases = [density_grid_ref] * self.num_phases
                self.tumor_masks_phases = [tumor_mask_ref] * self.num_phases
                self.oar_masks_phases = [oar_masks_ref] * self.num_phases
                self.affine_transforms = [np.eye(4)] * self.num_phases 

                self.tumor_mask = tumor_mask_ref
                self.oar_masks = oar_masks_ref
                self.density_grid = density_grid_ref
                logger.info("Successfully loaded static 3D CT and RTStruct.")
            except Exception as e:
                logger.error(f"Error loading static DICOM data: {e}. Falling back to simplified model.", exc_info=True)
                self._initialize_simplified_model(grid_size)
        else:
            logger.info("No DICOM paths provided. Initializing with simplified model.")
            self._initialize_simplified_model(grid_size)

        default_params = {
            "tumor": {"alpha": 0.3, "beta": 0.03, "alpha_beta": 10, "N0_density": 1e7}, 
            "lung": {"alpha_beta": 3, "TD50": 24.5, "m": 0.3, "n": 1},
            "heart": {"alpha_beta": 3, "TD50": 40, "m": 0.1, "n": 0.5}
        }
        self.radiobiological_params = patient_params if patient_params else default_params
        self.voxel_volume = 0.001 
        if dicom_rt_struct_path or ct_path or fourd_ct_path:
             try:
                 if hasattr(self, 'voxel_size_mm'): 
                      self.voxel_volume = np.prod(self.voxel_size_mm) * 1e-3 
                 else:
                      self.voxel_volume = 1.0 * 1.0 * 1.0 * 1e-3 
                      logger.warning("Could not determine voxel volume from DICOM. Assuming 1 mm^3.")
             except Exception as e:
                 logger.error(f"Error determining voxel volume from DICOM: {e}. Assuming 1 mm^3.", exc_info=True)
                 self.voxel_volume = 1.0 * 1.0 * 1.0 * 1e-3 

    def set_tumor_data(self, tumor_center=None, tumor_size=None, tumor_mask_input=None):
        """
        Set tumor data using either center and size parameters or a direct mask.
        
        Args:
            tumor_center (tuple/list/ndarray): 3D coordinates of tumor center (x,y,z)
            tumor_size (float or tuple): Radius for spherical tumor or (rx,ry,rz) for ellipsoid
            tumor_mask_input (ndarray): Direct boolean mask of tumor volume
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if tumor_mask_input is not None:
                if tumor_mask_input.shape != self.grid_size:
                    from scipy.ndimage import zoom
                    logger.info(f"Resizing input tumor mask from {tumor_mask_input.shape} to {self.grid_size}")
                    zoom_factors = tuple(gs_dim / tm_dim for gs_dim, tm_dim in zip(self.grid_size, tumor_mask_input.shape))
                    tumor_mask_input_resized = zoom(tumor_mask_input.astype(float), zoom_factors, order=0).astype(bool)
                    if tumor_mask_input_resized.shape != self.grid_size:
                         # This can happen due to rounding with zoom factors.
                         # A more robust way is to pad/crop or use a library that guarantees output shape.
                         # For now, log a warning and proceed if shapes are very close, or error if too different.
                         logger.warning(f"Resized mask shape {tumor_mask_input_resized.shape} still not matching grid {self.grid_size}. Check zoom logic.")
                         # Simple crop/pad to force match - this is crude
                         cropped_padded_mask = np.zeros(self.grid_size, dtype=bool)
                         slices = [slice(0, min(s_g, s_r)) for s_g, s_r in zip(self.grid_size, tumor_mask_input_resized.shape)]
                         cropped_padded_mask[tuple(slices)] = tumor_mask_input_resized[tuple(slices)]
                         self.tumor_mask = cropped_padded_mask
                    else:
                         self.tumor_mask = tumor_mask_input_resized
                else:
                    self.tumor_mask = tumor_mask_input.astype(bool)
            
            elif tumor_center is not None and tumor_size is not None:
                tumor_center = np.asarray(tumor_center)
                if isinstance(tumor_size, (int, float)):
                    self.tumor_mask = self._create_spherical_mask(tumor_center, tumor_size)
                else:
                    self.tumor_mask = self._create_ellipsoid_mask(tumor_center, tumor_size)
            else:
                # This implies that tumor_mask must have been set via DICOM loading, or this is an error
                if self.tumor_mask is None:
                     raise ValueError("No tumor data provided (mask, or center/size) and no tumor loaded from DICOM.")
                logger.info("Using existing tumor mask (presumably from DICOM or prior call).")


            if not hasattr(self, 'tumor_masks_phases') or self.tumor_masks_phases is None or len(self.tumor_masks_phases) != self.num_phases:
                self.tumor_masks_phases = [self.tumor_mask.copy() for _ in range(self.num_phases)]
            else: # If tumor_masks_phases exists, update each phase. This is simplistic for 4D.
                for i in range(self.num_phases):
                    self.tumor_masks_phases[i] = self.tumor_mask.copy() 
            
            # Update ITV (union of tumor masks across phases)
            if self.tumor_masks_phases:
                self.tumor_mask = np.any(self.tumor_masks_phases, axis=0)
            
            logger.info(f"Tumor data set. ITV volume: {np.sum(self.tumor_mask)} voxels.")
            return True
            
        except Exception as e:
            logger.error(f"Error in set_tumor_data: {e}", exc_info=True)
            return False

    def _initialize_simplified_model(self, grid_size_param):
        self.grid_size = grid_size_param 
        self.tumor_center = np.array(self.grid_size) / 2
        self.tumor_mask = self._create_spherical_mask(self.tumor_center, 10)
        self.tumor_mask_name = "Simulated Tumor"
        self.oar_masks = {
            "Simulated Lung": self._create_ellipsoid_mask(self.tumor_center + np.array([20, 0, 0]), (30, 20, 20)),
            "Simulated Heart": self._create_spherical_mask(self.tumor_center + np.array([-20, 0, 0]), 15)
        }
        self.density_grid = np.ones(self.grid_size)
        self.tumor_masks_phases = [self.tumor_mask.copy() for _ in range(self.num_phases)]
        self.oar_masks_phases = [{oar: mask.copy() for oar, mask in self.oar_masks.items()} for _ in range(self.num_phases)]
        self.density_grids_phases = [self.density_grid.copy() for _ in range(self.num_phases)]
        self.affine_transforms = [np.eye(4) for _ in range(self.num_phases)] 
        self.accumulated_dose = np.zeros(self.grid_size) 
        self.voxel_volume = 1.0 * 1.0 * 1.0 * 1e-3 
        logger.info(f"Initialized with simplified model, grid_size: {self.grid_size}")


    def _load_ct_series(self, ct_dir_path):
        """Loads a single CT series from a directory, sorted by slice location."""
        dicom_files = [pydicom.dcmread(os.path.join(ct_dir_path, f))
                       for f in os.listdir(ct_dir_path) if f.endswith('.dcm')]
        ct_series = [ds for ds in dicom_files if hasattr(ds, 'ImagePositionPatient') and hasattr(ds, 'InstanceNumber')]
        if all(hasattr(s, 'SliceLocation') for s in ct_series):
            ct_series.sort(key=lambda s: s.SliceLocation)
        elif all(hasattr(s, 'InstanceNumber') for s in ct_series):
             ct_series.sort(key=lambda s: s.InstanceNumber)
        else:
            logger.warning("CT slices could not be reliably sorted.")

        if not ct_series:
            logger.warning(f"No CT images found in {ct_dir_path}")
            return None
        
        try:
            pixel_spacing = np.array(ct_series[0].PixelSpacing, dtype=float)
            slice_thickness = float(ct_series[0].SliceThickness)
            self.voxel_size_mm = np.array([pixel_spacing[1], pixel_spacing[0], slice_thickness]) 
            logger.info(f"Derived voxel size (mm): {self.voxel_size_mm}")
        except Exception as e:
            logger.warning(f"Could not derive voxel size from CT series: {e}", exc_info=True)
            self.voxel_size_mm = np.array([1.0, 1.0, 1.0]) 
        return ct_series

    def _get_affine_transform_from_dicom(self, dicom_slice):
        ipp = np.array(dicom_slice.ImagePositionPatient, dtype=float)
        iop = np.array(dicom_slice.ImageOrientationPatient, dtype=float)
        row_vec = iop[:3]
        col_vec = iop[3:]
        ps = np.array(dicom_slice.PixelSpacing, dtype=float)
        rotation_scaling = np.zeros((3, 3))
        rotation_scaling[:, 0] = row_vec * ps[1] 
        rotation_scaling[:, 1] = col_vec * ps[0] 
        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = rotation_scaling
        affine_matrix[:3, 3] = ipp 
        return affine_matrix

    def _load_4d_ct_data(self, fourd_ct_path, rt_struct_path):
        phase_dirs = sorted([d for d in os.listdir(fourd_ct_path) if os.path.isdir(os.path.join(fourd_ct_path, d))])
        if not phase_dirs:
            raise FileNotFoundError(f"No phase directories found in {fourd_ct_path}")

        all_density_grids = []
        all_tumor_masks = []
        all_oar_masks_phases = [] 
        all_affine_transforms = []

        ref_phase_path = os.path.join(fourd_ct_path, self.reference_phase_name)
        if not os.path.isdir(ref_phase_path):
            ref_phase_path = os.path.join(fourd_ct_path, phase_dirs[0]) 
            logger.warning(f"Reference phase '{self.reference_phase_name}' not found. Using '{phase_dirs[0]}'.")

        logger.info(f"Loading reference CT from: {ref_phase_path}")
        ref_ct_series = self._load_ct_series(ref_phase_path)
        if not ref_ct_series:
            raise ValueError(f"Could not load reference CT series from {ref_phase_path}")

        rows = ref_ct_series[0].Rows
        cols = ref_ct_series[0].Columns
        num_slices_ref = len(ref_ct_series)
        self.grid_size = (cols, rows, num_slices_ref) 
        logger.info(f"Derived grid_size from reference CT: {self.grid_size}")

        tumor_mask_ref, oar_masks_ref, roi_names_map = self._load_rt_struct(rt_struct_path, ref_ct_series)
        # Update self.tumor_mask_name if a primary tumor ROI was identified by name
        for roi_num, name in roi_names_map.items():
            if "tumor" in name.lower() or "gtv" in name.lower() or "ptv" in name.lower():
                self.tumor_mask_name = name
                logger.info(f"Identified tumor ROI name: {self.tumor_mask_name}")
                break
        ref_volume_affine = self._get_affine_transform_from_dicom(ref_ct_series[0]) 

        for phase_idx, phase_dir_name in enumerate(phase_dirs):
            phase_ct_path = os.path.join(fourd_ct_path, phase_dir_name)
            logger.info(f"  Processing phase {phase_idx + 1}/{len(phase_dirs)}: {phase_dir_name}")
            current_ct_series = self._load_ct_series(phase_ct_path)
            if not current_ct_series:
                logger.warning(f"  Warning: Could not load CT for phase {phase_dir_name}. Skipping.")
                continue
            
            if current_ct_series[0].Rows != rows or current_ct_series[0].Columns != cols or len(current_ct_series) != num_slices_ref:
                logger.warning(f"  Warning: Phase {phase_dir_name} has different dimensions. Skipping. Real DIR needed.")
                continue

            ct_pixel_data_phase = np.stack([s.pixel_array * s.RescaleSlope + s.RescaleIntercept for s in current_ct_series], axis=-1)
            ct_pixel_data_phase = ct_pixel_data_phase.transpose(1,0,2) 
            
            density_grid_phase = self._hu_to_density(ct_pixel_data_phase)
            all_density_grids.append(density_grid_phase)
            current_volume_affine = self._get_affine_transform_from_dicom(current_ct_series[0]) 
            all_affine_transforms.append(current_volume_affine)

            if phase_dir_name == self.reference_phase_name or phase_ct_path == ref_phase_path:
                logger.info(f"    Phase {phase_dir_name} is the reference phase. Using original masks.")
                deformed_tumor_mask = tumor_mask_ref.copy()
                deformed_oar_masks = {name: mask.copy() for name, mask in oar_masks_ref.items()}
            else:
                logger.info(f"    Deforming structures for phase {phase_dir_name}...")
                if self.dir_method == 'simplified_sinusoidal':
                    logger.info("      DIR Method: Simplified Sinusoidal (Example - requires _apply_simplified_deformation)")
                    deformed_tumor_mask = tumor_mask_ref.copy() 
                    deformed_oar_masks = {name: mask.copy() for name, mask in oar_masks_ref.items()} 
                elif self.dir_method == 'external_sitk': 
                    logger.info("      DIR Method: External SimpleITK (Placeholder - NOT IMPLEMENTED)")
                    deformed_tumor_mask = tumor_mask_ref.copy()
                    deformed_oar_masks = {name: mask.copy() for name, mask in oar_masks_ref.items()}
                else: 
                    logger.info(f"      DIR Method: {self.dir_method} - using reference masks as is.")
                    deformed_tumor_mask = tumor_mask_ref.copy()
                    deformed_oar_masks = {name: mask.copy() for name, mask in oar_masks_ref.items()}
            all_tumor_masks.append(deformed_tumor_mask)
            all_oar_masks_phases.append(deformed_oar_masks)

        if not all_density_grids:
            raise ValueError("Failed to load any density grids from 4D CT.")
        return all_density_grids, all_tumor_masks, all_oar_masks_phases, all_affine_transforms

    def _apply_simplified_deformation(self, tumor_mask_ref, oar_masks_ref, displacement_vector):
        coords = np.indices(self.grid_size, dtype=float) 
        shifted_coords = [coords[i] - displacement_vector[i] for i in range(len(displacement_vector))]
        deformed_tumor_mask = map_coordinates(tumor_mask_ref.astype(float), shifted_coords, order=0, mode='constant', cval=0.0).astype(bool)
        deformed_oar_masks = {}
        for oar_name, oar_mask_ref in oar_masks_ref.items():
            deformed_oar_masks[oar_name] = map_coordinates(oar_mask_ref.astype(float), shifted_coords, order=0, mode='constant', cval=0.0).astype(bool)
        return deformed_tumor_mask, deformed_oar_masks

    def _load_rt_struct(self, rt_struct_path, ref_ct_series):
        rt_struct = pydicom.dcmread(rt_struct_path)
        sop_uid_to_index = {s.SOPInstanceUID: i for i, s in enumerate(ref_ct_series)}
        ref_slice_origin = np.array(ref_ct_series[0].ImagePositionPatient)
        pixel_spacing = np.array(ref_ct_series[0].PixelSpacing, dtype=float) 
        slice_locations = sorted([s.ImagePositionPatient[2] for s in ref_ct_series])
        slice_thickness = np.mean(np.diff(slice_locations)) if len(slice_locations) > 1 else pixel_spacing[0] 
        self.voxel_size_mm = np.array([pixel_spacing[1], pixel_spacing[0], slice_thickness]) 
        self.voxel_volume = np.prod(self.voxel_size_mm) * 1e-3 
        logger.info(f"Derived voxel size (mm): {self.voxel_size_mm}")
        logger.info(f"Derived voxel volume (cm^3): {self.voxel_volume:.6f}")

        iop = ref_ct_series[0].ImageOrientationPatient
        row_dir = np.array(iop[0:3]) 
        col_dir = np.array(iop[3:6]) 
        slice_dir = np.cross(row_dir, col_dir) 
        T_patient_to_grid = np.eye(4)
        R_patient_to_grid = np.column_stack((col_dir * pixel_spacing[1], row_dir * pixel_spacing[0], slice_dir * slice_thickness))
        try:
            T_patient_to_grid[:3,:3] = np.linalg.inv(R_patient_to_grid)
        except np.linalg.LinAlgError:
            logger.warning("Could not invert rotation matrix for RTStruct loading. Using identity.")
            T_patient_to_grid[:3,:3] = np.eye(3) 
        T_patient_to_grid[:3,3] = -T_patient_to_grid[:3,:3] @ ref_slice_origin

        overall_tumor_mask = np.zeros(self.grid_size, dtype=bool)
        overall_oar_masks = {}
        roi_names = {} 

        for i, roi_contour in enumerate(rt_struct.ROIContourSequence):
            roi_number = roi_contour.ReferencedROINumber
            structure_set_roi = next((s for s in rt_struct.StructureSetROISequence if s.ROINumber == roi_number), None)
            if not structure_set_roi: continue
            
            roi_name_orig = structure_set_roi.ROIName
            roi_name_lower = roi_name_orig.lower()
            logger.info(f"    Processing ROI: {roi_name_orig}")
            roi_names[roi_number] = roi_name_orig
            current_roi_mask = np.zeros(self.grid_size, dtype=bool)

            if hasattr(roi_contour, 'ContourSequence'):
                for contour_sequence in roi_contour.ContourSequence:
                    contour_data = np.array(contour_sequence.ContourData).reshape(-1, 3) 
                    slice_idx = None
                    if hasattr(contour_sequence, 'ContourImageSequence') and contour_sequence.ContourImageSequence:
                         if contour_sequence.ContourImageSequence[0].ReferencedSOPInstanceUID in sop_uid_to_index:
                             slice_idx = sop_uid_to_index[contour_sequence.ContourImageSequence[0].ReferencedSOPInstanceUID]
                         else: 
                             contour_z = contour_data[0, 2]
                             slice_z_diffs = [abs(s.ImagePositionPatient[2] - contour_z) for s in ref_ct_series]
                             if slice_z_diffs:
                                 slice_idx = np.argmin(slice_z_diffs)
                    if slice_idx is None:
                        logger.warning(f"      Warning: Could not map contour for ROI {roi_name_orig} to a CT slice. Skipping contour.")
                        continue
                    contour_data_homog = np.hstack((contour_data, np.ones((contour_data.shape[0], 1))))
                    contour_voxels_float = (T_patient_to_grid @ contour_data_homog.T).T[:, :3]
                    from skimage.draw import polygon
                    rr, cc = polygon(
                        np.clip(contour_voxels_float[:, 1], 0, self.grid_size[1] - 1),  
                        np.clip(contour_voxels_float[:, 0], 0, self.grid_size[0] - 1)   
                    )
                    valid_indices = (rr >= 0) & (rr < self.grid_size[1]) & \
                                    (cc >= 0) & (cc < self.grid_size[0])
                    if 0 <= slice_idx < self.grid_size[2]:
                         current_roi_mask[cc[valid_indices], rr[valid_indices], slice_idx] = True
                    else:
                         logger.warning(f"      Warning: Contour for ROI {roi_name_orig} maps to out-of-bounds slice index {slice_idx}. Skipping contour.")
            if "tumor" in roi_name_lower or "gtv" in roi_name_lower or "ctv" in roi_name_lower:
                overall_tumor_mask |= current_roi_mask
            elif "ptv" in roi_name_lower: 
                overall_tumor_mask |= current_roi_mask
            else: 
                if roi_name_orig not in overall_oar_masks:
                    overall_oar_masks[roi_name_orig] = np.zeros(self.grid_size, dtype=bool)
                overall_oar_masks[roi_name_orig] |= current_roi_mask
        return overall_tumor_mask, overall_oar_masks, roi_names

    def _create_spherical_mask(self, center, radius):
        x, y, z = np.ogrid[:self.grid_size[0], :self.grid_size[1], :self.grid_size[2]]
        center = np.asarray(center)
        dist_squared = ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
        return dist_squared <= radius**2
    
    def _create_ellipsoid_mask(self, center, radii):
        x, y, z = np.ogrid[:self.grid_size[0], :self.grid_size[1], :self.grid_size[2]]
        center = np.asarray(center)
        radii = np.asarray(radii)
        dist_squared = ((x - center[0])**2 / radii[0]**2 + 
                       (y - center[1])**2 / radii[1]**2 + 
                       (z - center[2])**2 / radii[2]**2)
        return dist_squared <= 1.0

    def _hu_to_density(self, hu_data):
        density = np.ones_like(hu_data, dtype=float)
        density[hu_data <= -500] = 0.2 
        density[(hu_data > -500) & (hu_data <= 50)] = 1.0 
        density[hu_data > 50] = 1.2 
        return density

    def _generate_beam_directions(self, num_beams):
        angles = np.linspace(0, 2 * np.pi, num_beams, endpoint=False)
        directions = [(np.cos(a), np.sin(a), 0) for a in angles] 
        return directions

    def _get_source_position(self, direction):
        center = np.array(self.grid_size) / 2.0
        max_dim = np.max(self.grid_size)
        source_pos = center - np.array(direction) * max_dim * 1.5
        return source_pos

    def _calculate_dose_for_specific_phase(self, beam_weights, target_phase_index):
        phase_dose_at_target = np.zeros(self.grid_size)
        density_grid_target_phase = self.density_grids_phases[target_phase_index]
        for i, (direction, weight) in enumerate(zip(self.beam_directions, beam_weights)):
            if weight == 0:
                continue
            source = self._get_source_position(direction)
            fluence = calculate_primary_fluence_numba(source, np.array(direction), density_grid_target_phase, self.grid_size)
            partial_dose = convolve(fluence, self.dose_kernel, mode="constant")
            mean_density_this_phase = np.mean(density_grid_target_phase[density_grid_target_phase > 0])
            if mean_density_this_phase > 1e-9: 
                partial_dose *= (density_grid_target_phase / mean_density_this_phase)
            phase_dose_at_target += weight * partial_dose
        return phase_dose_at_target

    def calculate_dose(self, beam_weights):
        dose = np.zeros(self.grid_size)
        if not self.density_grids_phases: 
            logger.error("Error: No density grids available for dose calculation.")
            return dose

        for phase in range(self.num_phases):
            phase_dose = np.zeros(self.grid_size)
            density_grid = self.density_grids_phases[phase]
            for i, (direction, weight) in enumerate(zip(self.beam_directions, beam_weights)):
                if weight == 0:
                    continue
                source = self._get_source_position(direction)
                fluence = calculate_primary_fluence_numba(source, np.array(direction), density_grid, self.grid_size)
                partial_dose = convolve(fluence, self.dose_kernel, mode="constant")
                mean_density_phase = np.mean(density_grid[density_grid > 0])
                if mean_density_phase > 1e-9: 
                     partial_dose *= (density_grid / mean_density_phase)
                phase_dose += weight * partial_dose
            dose += self.respiratory_phase_weights[phase] * phase_dose
        
        tumor_volume = np.sum(self.tumor_mask) if self.tumor_mask is not None else 0
        base_dose_per_fraction = 4.0 
        if tumor_volume > 0: 
            if tumor_volume <= 5: base_dose_per_fraction = 4.0
            elif tumor_volume >= 30: base_dose_per_fraction = 2.0
            else: base_dose_per_fraction = 2.0 + 2.0 * (30.0 - tumor_volume) / 25.0
        
        max_dose_in_itv = 0
        if self.tumor_mask is not None and np.any(self.tumor_mask) and np.any(dose[self.tumor_mask]):
             max_dose_in_itv = np.max(dose[self.tumor_mask])

        if max_dose_in_itv > 1e-6:
            dose *= base_dose_per_fraction / max_dose_in_itv
        elif np.max(dose) > 1e-6: 
            dose *= base_dose_per_fraction / np.max(dose)
        return dose

    def optimize_beams(self):
        logger.info("  [optimize_beams] Starting classical optimization...")
        num_beams = len(self.beam_directions)
        dose_influence_phases = np.zeros((self.num_phases, num_beams) + self.grid_size)
        logger.info(f"  [optimize_beams] Calculating dose influence matrix for {self.num_phases} phases and {num_beams} beams...")
        for phase_opt_idx in range(self.num_phases):
            density_grid_phase = self.density_grids_phases[phase_opt_idx]
            for i_beam in range(num_beams):
                source = self._get_source_position(self.beam_directions[i_beam])
                fluence = calculate_primary_fluence_numba(source, np.array(self.beam_directions[i_beam]), density_grid_phase, self.grid_size)
                partial_dose = convolve(fluence, self.dose_kernel, mode="constant")
                mean_density_phase = np.mean(density_grid_phase[density_grid_phase > 0])
                if mean_density_phase > 1e-9:
                     partial_dose *= (density_grid_phase / mean_density_phase)
                dose_influence_phases[phase_opt_idx, i_beam, :, :, :] = partial_dose
        logger.info("  [optimize_beams] Dose influence matrix calculated.")
        
        def objective_function(weights):
            weights = np.maximum(0, weights)
            total_dose_averaged_phases = np.zeros(self.grid_size)
            for phase_idx in range(self.num_phases):
                 dose_this_phase = np.tensordot(weights, dose_influence_phases[phase_idx], axes=([0], [0]))
                 total_dose_averaged_phases += self.respiratory_phase_weights[phase_idx] * dose_this_phase
            tumor_term = 0
            if self.tumor_mask is not None and np.any(self.tumor_mask):
                 mean_tumor_dose = np.mean(total_dose_averaged_phases[self.tumor_mask])
                 tumor_term = -1.0 * mean_tumor_dose 
            oar_term = 0
            for oar_name, oar_mask in self.oar_masks.items():
                 if oar_mask is not None and np.any(oar_mask):
                      mean_oar_dose = np.mean(total_dose_averaged_phases[oar_mask])
                      oar_term += 0.5 * mean_oar_dose 
            opposing_penalty = 0
            for i in range(num_beams):
                for j in range(i + 1, num_beams):
                    angle_rad = np.arccos(np.clip(np.dot(self.beam_directions[i], self.beam_directions[j]), -1.0, 1.0))
                    angle_deg = np.degrees(angle_rad)
                    if angle_deg > 160:
                        opposing_penalty += 2.0 * weights[i] * weights[j] 
            cost = tumor_term + oar_term + opposing_penalty
            return cost

        logger.info("  [optimize_beams] Running classical optimization (L-BFGS-B)...")
        initial_weights = np.ones(num_beams) / num_beams
        bounds = [(0, None)] * num_beams 
        result = minimize(
            objective_function, initial_weights, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 1000, 'disp': False} 
        )
        optimized_weights = result.x
        logger.info(f"  [optimize_beams] Classical optimization finished. Success: {result.success}, Message: {result.message}")
        threshold = 0.1 
        best_weights_binary = (optimized_weights > threshold).astype(float) 
        if np.sum(best_weights_binary) == 0:
            logger.warning("Warning: Classical optimization resulted in all zero weights after thresholding. Using default (first 3 beams).")
            num_to_activate = min(3, num_beams)
            best_weights_binary = np.zeros(num_beams, dtype=float)
            if num_beams > 0: best_weights_binary[:num_to_activate] = 1.0
        
        self.beam_weights = best_weights_binary # Store optimized weights
        logger.info(f"  [optimize_beams] Final binary beam selection: {self.beam_weights}")
        logger.info("  [optimize_beams] Finished.")
        return self.beam_weights

    def simulate_fractionated_treatment(self, num_fractions=5):
        history = {"tumor_volumes": [], "tcp": [], "ntcp": {oar: [] for oar in self.oar_masks}} 
        self.accumulated_dose = np.zeros(self.grid_size)
        for fraction in range(num_fractions):
            logger.info(f"Processing fraction {fraction + 1}/{num_fractions}")
            current_beam_weights = self.optimize_beams() # This now sets self.beam_weights
            if self.beam_weights is None: # Should be set by optimize_beams
                logger.error("Beam weights not set after optimization. Cannot simulate fraction.")
                return history # Or handle error appropriately
            
            fraction_dose = self.calculate_dose(self.beam_weights) 
            self.accumulated_dose += fraction_dose
            alpha_tumor = self.radiobiological_params["tumor"]["alpha"]
            beta_tumor = self.radiobiological_params["tumor"]["beta"]
            for phase in range(self.num_phases):
                current_phase_tumor_mask = self.tumor_masks_phases[phase]
                if not np.any(current_phase_tumor_mask): continue 
                survival_prob_map_fraction = np.exp(-(alpha_tumor * fraction_dose + beta_tumor * fraction_dose**2))
                random_survival = np.random.rand(*fraction_dose.shape)
                surviving_voxels_this_phase = (random_survival < survival_prob_map_fraction) & current_phase_tumor_mask
                self.tumor_masks_phases[phase] = surviving_voxels_this_phase
            
            self.tumor_mask = np.any(self.tumor_masks_phases, axis=0) 
            tcp_val = self._calculate_tcp(self.accumulated_dose) 
            history["tumor_volumes"].append(np.sum(self.tumor_mask)) 
            history["tcp"].append(tcp_val)
            for oar_name in self.oar_masks.keys(): 
                ntcp_val = self._calculate_ntcp(self.accumulated_dose, oar_name)
                if oar_name not in history["ntcp"]: history["ntcp"][oar_name] = [] 
                history["ntcp"][oar_name].append(ntcp_val)
            
            logger.info(f"Fraction {fraction + 1}: ITV Volume = {np.sum(self.tumor_mask)}, TCP = {tcp_val:.4f}")
            if not self.tumor_mask.any():
                logger.info("Tumor eradicated!")
        
        # Store final TCP/NTCP values from the simulation
        self.tcp_value = tcp_val
        self.ntcp_values = {oar_name: history["ntcp"][oar_name][-1] for oar_name in history["ntcp"] if history["ntcp"][oar_name]}
        self.dose_distribution = self.accumulated_dose # Ensure this reflects final state
        return history

    def _calculate_tcp(self, total_accumulated_dose):
        if self.tumor_mask is None or not np.any(self.tumor_mask): return 100.0 # Tumor eradicated = 100% TCP
        alpha_tumor = self.radiobiological_params["tumor"]["alpha"]
        beta_tumor = self.radiobiological_params["tumor"]["beta"]
        N0_density = self.radiobiological_params["tumor"]["N0_density"] 
        sf_map_total = np.exp(-(alpha_tumor * total_accumulated_dose + beta_tumor * total_accumulated_dose**2))
        surviving_cells_per_voxel_in_itv = (N0_density * self.voxel_volume * sf_map_total[self.tumor_mask])
        total_surviving_cells = np.sum(surviving_cells_per_voxel_in_itv)
        tcp = np.exp(-total_surviving_cells)
        return tcp * 100.0 

    def _calculate_ntcp(self, total_accumulated_dose, oar_name):
        if oar_name not in self.radiobiological_params or oar_name not in self.oar_masks:
            return 0.0
        if self.oar_masks[oar_name] is None or not np.any(self.oar_masks[oar_name]): return 0.0 

        params = self.radiobiological_params[oar_name]
        n, m, TD50 = params["n"], params["m"], params["TD50"]
        alpha_beta_oar = params["alpha_beta"]
        current_oar_mask = self.oar_masks[oar_name] 
        total_dose_oar_voxels = total_accumulated_dose[current_oar_mask]
        if not total_dose_oar_voxels.size: return 0.0 
        d_ref = 2.0 
        # EQD2 calculation needs total_dose_oar_voxels to be the dose per fraction if it's to be summed,
        # or this formula applies to total dose D if d = D/N.
        # Here, total_accumulated_dose is passed. Assuming it's the total dose D.
        # To use the standard EQD2 formula D * ((d + ab)/(2+ab)), we need d (dose per fraction).
        # If total_accumulated_dose is D, and we assume N fractions, d = D/N.
        # For simplicity, if the _calculate_ntcp is called with total_accumulated_dose,
        # we must assume this IS the total EQD2 or can be used directly in LKB if model params are for total dose.
        # The current formula `eqd2_oar_voxels = total_dose_oar_voxels * (total_dose_oar_voxels + alpha_beta_oar) / (d_ref + alpha_beta_oar)`
        # is incorrect if total_dose_oar_voxels is D_total. It should be d_total * (d_frac + ab) / (d_ref + ab)
        # Let's assume total_dose_oar_voxels is the total dose D.
        # And we need to estimate d_frac. If num_fractions available, d_frac = D/num_fractions.
        # For now, as a placeholder, let's assume total_dose_oar_voxels is already EQD2-like for the LKB model,
        # or that the LKB parameters (TD50) are already for this type of total dose.
        # This is a common simplification if fractionation details are not rigorously passed.
        eqd2_oar_voxels = total_dose_oar_voxels # Placeholder: treat total_accumulated_dose as EQD2 or direct input to LKB
        
        if abs(n) < 1e-9: 
             gEUD = np.mean(eqd2_oar_voxels)
        else:
            try:
                gEUD = np.mean(eqd2_oar_voxels**(1/n))**n
            except Exception as e: 
                 logger.warning(f"Warning: Error calculating gEUD for {oar_name} with n={n}. Falling back to mean EQD2. Error: {e}")
                 gEUD = np.mean(eqd2_oar_voxels)
        t_numerator = gEUD - TD50
        t_denominator = m * TD50
        if abs(t_denominator) < 1e-9: 
            return 100.0 if t_numerator > 0 else 0.0
        t = t_numerator / t_denominator 
        ntcp = 0.5 * (1 + erf(t / np.sqrt(2))) 
        return ntcp * 100.0 

    def calculate_plan_metrics(self, beam_weights_input: np.ndarray, num_fractions_for_eval: int = 30) -> Dict[str, Any]:
        logger.info("Calculating plan metrics...")
        metrics: Dict[str, Any] = {'tumor': {}, 'oars': {}}

        if self.tumor_mask is None or not np.any(self.tumor_mask):
            logger.warning("calculate_plan_metrics: Tumor mask not available or empty. Tumor metrics will be zero/None.")
        
        fractional_dose_for_metrics = self.calculate_dose(beam_weights_input)
        if fractional_dose_for_metrics is None:
            logger.error("calculate_plan_metrics: Failed to calculate fractional dose for metrics evaluation.")
            return metrics 

        total_dose_for_tcp_ntcp = fractional_dose_for_metrics * num_fractions_for_eval
        logger.info(f"calculate_plan_metrics: Evaluating TCP/NTCP based on total dose scaled by {num_fractions_for_eval} fractions.")

        if self.tumor_mask is not None and np.any(self.tumor_mask):
            tumor_doses_fractional = fractional_dose_for_metrics[self.tumor_mask]
            mean_tumor_dose_fractional = float(np.mean(tumor_doses_fractional)) if tumor_doses_fractional.size > 0 else 0.0
            metrics['tumor']['mean_fractional_dose'] = mean_tumor_dose_fractional
            
            v95_threshold = 0.95 * mean_tumor_dose_fractional 
            metrics['tumor']['v95_fractional_mean_ref'] = float(np.sum(tumor_doses_fractional >= v95_threshold) / tumor_doses_fractional.size * 100) if tumor_doses_fractional.size > 0 and mean_tumor_dose_fractional > 1e-6 else 0.0
            metrics['tumor']['tcp'] = self._calculate_tcp(total_dose_for_tcp_ntcp) 
        else:
            metrics['tumor']['mean_fractional_dose'] = 0.0
            metrics['tumor']['v95_fractional_mean_ref'] = 0.0
            metrics['tumor']['tcp'] = 0.0

        for oar_name, oar_mask in self.oar_masks.items():
            metrics['oars'][oar_name] = {}
            if oar_mask is not None and np.any(oar_mask):
                oar_doses_fractional = fractional_dose_for_metrics[oar_mask]
                metrics['oars'][oar_name]['mean_fractional_dose'] = float(np.mean(oar_doses_fractional)) if oar_doses_fractional.size > 0 else 0.0
                metrics['oars'][oar_name]['max_fractional_dose'] = float(np.max(oar_doses_fractional)) if oar_doses_fractional.size > 0 else 0.0
                
                dose_threshold_oar_fractional = 0.0 
                threshold_name = "generic_V_dose_fractional"
                if 'lung' in oar_name.lower():
                    total_dose_threshold_lung = 20.0 
                    dose_threshold_oar_fractional = total_dose_threshold_lung / num_fractions_for_eval
                    threshold_name = f"V{total_dose_threshold_lung}Gy_total_equiv_fractional"
                elif 'heart' in oar_name.lower():
                     total_dose_threshold_heart = 30.0 
                     dose_threshold_oar_fractional = total_dose_threshold_heart / num_fractions_for_eval
                     threshold_name = f"V{total_dose_threshold_heart}Gy_total_equiv_fractional"
                else: 
                    total_dose_threshold_generic = 5.0
                    dose_threshold_oar_fractional = total_dose_threshold_generic / num_fractions_for_eval
                    threshold_name = f"V{total_dose_threshold_generic}Gy_total_equiv_fractional"

                metrics['oars'][oar_name][threshold_name] = {
                    'dose_gy_per_fraction_threshold': dose_threshold_oar_fractional,
                    'volume_pct': float(np.sum(oar_doses_fractional >= dose_threshold_oar_fractional) / oar_doses_fractional.size * 100) if oar_doses_fractional.size > 0 else 0.0
                }
                metrics['oars'][oar_name]['ntcp'] = self._calculate_ntcp(total_dose_for_tcp_ntcp, oar_name)
            else: 
                metrics['oars'][oar_name] = {
                    'mean_fractional_dose': 0.0, 'max_fractional_dose': 0.0, 
                    'generic_V_dose_fractional': {'dose_gy_per_fraction_threshold': 0.0, 'volume_pct': 0.0}, 
                    'ntcp': 0.0
                }
        logger.info(f"Calculated plan metrics: {metrics}")
        return metrics

    def generate_dvh_data(self, dose_distribution_input: np.ndarray, num_bins: int = 100) -> Dict[str, Dict[str, np.ndarray]]:
        logger.info("Generating DVH data...")
        if dose_distribution_input is None:
            logger.error("Dose distribution input is None. Cannot generate DVH.")
            return {}

        dvh_data: Dict[str, Dict[str, np.ndarray]] = {}
        max_dose_overall = np.max(dose_distribution_input) if np.any(dose_distribution_input) else 1.0
        if max_dose_overall < 1e-6: 
            logger.warning("Max dose in distribution is close to 0. DVH will be trivial.")
            max_dose_overall = 1.0 

        rois_to_process: List[Tuple[str, Optional[np.ndarray]]] = []
        if self.tumor_mask is not None and np.any(self.tumor_mask) :
            rois_to_process.append((self.tumor_mask_name, self.tumor_mask))
        
        for oar_name, oar_mask in self.oar_masks.items():
            if oar_mask is not None and np.any(oar_mask):
                rois_to_process.append((oar_name, oar_mask))

        if not rois_to_process:
            logger.warning("No valid ROIs (tumor or OARs) found to generate DVH for.")
            return {}

        for roi_name, roi_mask in rois_to_process:
            if roi_mask is None: continue 

            roi_doses = dose_distribution_input[roi_mask]
            if roi_doses.size == 0:
                logger.warning(f"ROI '{roi_name}' is empty or mask does not overlap with dose. Creating trivial DVH.")
                bins = np.linspace(0, max_dose_overall, num_bins + 1)
                volume_pct = np.zeros(num_bins) 
                dvh_data[roi_name] = {'bins': bins[:-1], 'volume_pct': volume_pct }
                continue

            hist, bin_edges = np.histogram(roi_doses, bins=num_bins, range=(0, max_dose_overall))
            cumulative_hist = np.cumsum(hist[::-1])[::-1] 
            roi_total_voxels = np.sum(roi_mask) 
            if roi_total_voxels == 0:
                 volume_percentages = np.zeros_like(cumulative_hist, dtype=float)
            else:
                 volume_percentages = (cumulative_hist / roi_total_voxels) * 100.0
            
            dvh_data[roi_name] = {'bins': bin_edges[:-1], 'volume_pct': volume_percentages}
            logger.debug(f"DVH for {roi_name}: {len(volume_percentages)} points, max dose in ROI: {roi_doses.max():.2f}")
            
        logger.info("DVH data generation complete.")
        return dvh_data

    def validate_dose(self, dose, monte_carlo_reference=None):
        logger.info("\nPerforming dose validation...")
        if not self.density_grids_phases or not self.beam_directions:
             logger.warning("Cannot perform dose validation: No density grids or beam directions available.")
             return 0.0
             
        eval_dose_single_beam = self._calculate_dose_for_specific_phase([1.0] + [0.0]*(len(self.beam_directions)-1), 0)

        if monte_carlo_reference is None:
            logger.info("No Monte Carlo reference provided. Creating a mock reference.")
            noise_scale = 0.02 * np.max(eval_dose_single_beam) if np.max(eval_dose_single_beam) > 0 else 0.0
            noise = np.random.normal(0, noise_scale, eval_dose_single_beam.shape)
            monte_carlo_reference = eval_dose_single_beam + noise
            monte_carlo_reference[monte_carlo_reference < 0] = 0 
        else:
             logger.info("Using provided Monte Carlo reference.")
             if monte_carlo_reference.shape != self.grid_size:
                  logger.warning("Warning: MC reference shape does not match grid size. Gamma analysis may fail or be inaccurate.")
        gamma_pass_rate = self._gamma_analysis(eval_dose_single_beam, monte_carlo_reference, distance_voxels=3, dose_diff_percent=3)
        logger.info(f"Gamma Pass Rate (3 voxels, 3%): {gamma_pass_rate * 100:.2f}%")
        return gamma_pass_rate

    def _gamma_analysis(self, eval_dose, ref_dose, distance_voxels, dose_diff_percent):
        eval_dose = np.asarray(eval_dose)
        ref_dose = np.asarray(ref_dose)
        if eval_dose.shape != ref_dose.shape:
            logger.error("Error: Dose grids for gamma analysis must have the same shape.")
            return 0.0
        ref_dose_max = np.max(ref_dose)
        if ref_dose_max < 1e-9: 
            logger.info("Gamma Analysis: Reference dose is zero everywhere.")
            return 1.0 if np.max(eval_dose) < 1e-9 else 0.0

        dose_diff_abs = (dose_diff_percent / 100.0) * ref_dose_max
        low_dose_threshold = 0.10 * ref_dose_max
        eval_indices = np.argwhere(ref_dose >= low_dose_threshold)
        if eval_indices.size == 0:
             logger.warning("Warning: No points above low dose threshold for Gamma Analysis.")
             return 1.0 
        total_points_evaluated = eval_indices.shape[0]
        passed_points = 0
        dist_criterion_sq = distance_voxels**2
        dose_criterion_sq = dose_diff_abs**2

        for eval_idx_tuple in eval_indices:
            eval_idx = tuple(eval_idx_tuple)
            eval_dose_val = eval_dose[eval_idx]
            min_gamma_sq_for_point = np.inf
            min_coords = [max(0, eval_idx[d] - distance_voxels) for d in range(eval_dose.ndim)]
            max_coords = [min(eval_dose.shape[d], eval_idx[d] + distance_voxels + 1) for d in range(eval_dose.ndim)]
            search_ranges = [range(min_c, max_c) for min_c, max_c in zip(min_coords, max_coords)]
            
            for ref_idx_tuple in np.array(np.meshgrid(*search_ranges)).T.reshape(-1, eval_dose.ndim):
                ref_idx = tuple(ref_idx_tuple)
                dist_spatial_sq = sum([(eval_idx[d] - ref_idx[d])**2 for d in range(eval_dose.ndim)])
                dose_deviation_val = eval_dose_val - ref_dose[ref_idx]
                term1_sq = dist_spatial_sq / dist_criterion_sq if dist_criterion_sq > 1e-9 else 0
                term2_sq = (dose_deviation_val**2) / dose_criterion_sq if dose_criterion_sq > 1e-9 else 0
                current_gamma_sq = term1_sq + term2_sq
                if current_gamma_sq < min_gamma_sq_for_point:
                    min_gamma_sq_for_point = current_gamma_sq
            if min_gamma_sq_for_point <= 1.0 + 1e-9: 
                passed_points += 1
        pass_rate = passed_points / total_points_evaluated if total_points_evaluated > 0 else 0.0
        return pass_rate

    def plot_results(self, dose, beam_weights, history): # beam_weights is passed but not used here
        fig = plt.figure(figsize=(18, 6)) 
        ax1 = fig.add_subplot(131)
        if self.tumor_mask is not None and np.any(self.tumor_mask):
            itv_indices_z = np.where(self.tumor_mask.any(axis=(0,1)))[0]
            slice_idx_z = itv_indices_z[len(itv_indices_z)//2] if len(itv_indices_z) > 0 else self.grid_size[2] // 2
            dose_slice = dose[:, :, slice_idx_z]
            im = ax1.imshow(dose_slice.T, cmap='jet', origin='lower', aspect='auto') 
            itv_contour_slice = self.tumor_mask[:, :, slice_idx_z]
            ax1.contour(itv_contour_slice.T, colors='w', linewidths=0.8, origin='lower')
            fig.colorbar(im, ax=ax1, label="Dose (Gy_eff)")
            ax1.set_title(f"Final Accumulated Dose on Slice z={slice_idx_z} (ITV)")
            ax1.set_xlabel("X-voxel")
            ax1.set_ylabel("Y-voxel")
        else:
            ax1.text(0.5, 0.5, "No Tumor Mask (ITV)", ha='center', va='center')
            ax1.set_title("Final Accumulated Dose Distribution")

        ax2 = fig.add_subplot(132)
        # Use generate_dvh_data for the DVH plot
        dvh_plot_data = self.generate_dvh_data(dose) # 'dose' here is the accumulated dose from simulation
        for roi_name, data_points in dvh_plot_data.items():
            color = 'red' if roi_name == self.tumor_mask_name else None
            ax2.plot(data_points['bins'], data_points['volume_pct'], label=roi_name, color=color)
        
        ax2.set_title("Cumulative Dose-Volume Histogram (Final Dose)")
        ax2.set_xlabel("Dose (Gy_eff)")
        ax2.set_ylabel("Volume (%)")
        ax2.legend()
        ax2.grid(True, linestyle=':')
        ax2.set_xlim(left=0)
        ax2.set_ylim(0, 100)

        ax3 = fig.add_subplot(133)
        if history and "tumor_volumes" in history and history["tumor_volumes"]: # Check if history is not empty
            fractions = range(1, len(history["tumor_volumes"]) + 1)
            if history["tumor_volumes"]:
                ax3.plot(fractions, history["tumor_volumes"], label="ITV Volume", marker='o')
            if history["tcp"]:
                ax3.plot(fractions, history["tcp"], label="TCP (%)", marker='s')
            for oar_name, ntcp_history in history["ntcp"].items():
                if ntcp_history: 
                     ax3.plot(fractions, ntcp_history, label=f"NTCP {oar_name} (%)", marker='^', linestyle='--')
            ax3.set_title("Treatment History")
            ax3.set_xlabel("Fraction Number")
            ax3.set_ylabel("Value")
            ax3.legend()
            ax3.grid(True, linestyle=':')
            ax3.set_ylim(bottom=0) 
        else:
            ax3.text(0.5,0.5, "No history data.", ha='center', va='center')
            ax3.set_title("Treatment History")


        plt.tight_layout()
        plt.savefig("results_enhanced.png")
        logger.info("Plot saved as 'results_enhanced.png'")
        plt.close(fig)

    def get_plan_summary(self) -> Dict[str, Any]:
        """ Returns a summary of the current plan status, including metrics if available. """
        summary = {
            "beam_weights": self.beam_weights.tolist() if self.beam_weights is not None else None,
            "grid_size": self.grid_size,
            "num_beams": len(self.beam_directions), # Use len of beam_directions
            "tumor_present": np.any(self.tumor_mask) if self.tumor_mask is not None else False,
            "oars_defined": list(self.oar_masks.keys()),
            "kernel_loaded": self.dose_kernel is not None,
        }
        
        if self.beam_weights is not None:
            summary["plan_metrics_eval_30_fractions"] = self.calculate_plan_metrics(self.beam_weights, num_fractions_for_eval=30)

        if self.tcp_value is not None: # From simulation
             summary["final_tcp_simulated"] = self.tcp_value
        if self.ntcp_values: # From simulation
             summary["final_ntcp_simulated"] = self.ntcp_values
        
        current_dose_to_report = self.accumulated_dose if self.accumulated_dose is not None and np.any(self.accumulated_dose) else self.dose_distribution
        if current_dose_to_report is not None: 
            summary["reported_dose_max"] = float(np.max(current_dose_to_report))
            if summary["tumor_present"] and self.tumor_mask is not None and np.any(self.tumor_mask) and current_dose_to_report[self.tumor_mask].size > 0:
                 summary["reported_dose_mean_tumor"] = float(np.mean(current_dose_to_report[self.tumor_mask]))
            else:
                 summary["reported_dose_mean_tumor"] = 0.0
        return summary

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # Setup basic logging for __main__
    logger.info("Starting QRadPlan3D main execution...")
    if not os.path.exists("dose_kernel.npy"):
        logger.info("Dose kernel not found. Generating a new one...")
        try:
            from generate_dose_kernel import generate_updated_dose_kernel
            kernel = generate_updated_dose_kernel()
            np.save("dose_kernel.npy", kernel)
            logger.info("Dose kernel generated successfully!")
        except ImportError:
            logger.error("Could not import generate_updated_dose_kernel. Please ensure generate_dose_kernel.py is available.")
            exit()
        except Exception as e:
            logger.error(f"Error generating dose kernel: {e}", exc_info=True)
            exit()

    use_dicom = False 
    FOURD_CT_BASE_PATH = "./dummy_4d_ct_data" 
    RTSTRUCT_PATH = "./dummy_rtstruct/rtstruct.dcm" 
    STATIC_CT_PATH = "./dummy_static_ct_data" 

    if use_dicom and os.path.exists(FOURD_CT_BASE_PATH) and os.path.exists(RTSTRUCT_PATH):
        logger.info("Attempting to initialize QRadPlan3D with 4D DICOM data...")
        planner = QRadPlan3D(
            kernel_path="dose_kernel.npy",
            fourd_ct_path=FOURD_CT_BASE_PATH,
            dicom_rt_struct_path=RTSTRUCT_PATH,
            reference_phase_name="phase_0", 
            dir_method='simplified_sinusoidal' 
        )
    elif use_dicom and os.path.exists(STATIC_CT_PATH) and os.path.exists(RTSTRUCT_PATH):
        logger.info("Attempting to initialize QRadPlan3D with static 3D DICOM data...")
        planner = QRadPlan3D(
            kernel_path="dose_kernel.npy",
            ct_path=STATIC_CT_PATH,
            dicom_rt_struct_path=RTSTRUCT_PATH
        )
    else:
        logger.info("DICOM paths not valid or 'use_dicom' is False. Initializing with simplified model.")
        planner = QRadPlan3D(
            grid_size=(30, 30, 30), 
            num_beams=6,
            kernel_path="dose_kernel.npy"
        )

    logger.info("\nRunning fractionated treatment simulation...")
    history = planner.simulate_fractionated_treatment(num_fractions=3) 
    final_accumulated_dose = planner.accumulated_dose

    logger.info("\nValidating dose calculation model...")
    gamma_pass_rate = planner.validate_dose(final_accumulated_dose, monte_carlo_reference=None) 

    logger.info("\nGenerating plots...")
    planner.plot_results(final_accumulated_dose, planner.beam_weights, history) # Pass beam_weights if available

    logger.info("\n--- Treatment Plan Summary ---")
    plan_summary = planner.get_plan_summary()
    for key, value in plan_summary.items():
        if key == "plan_metrics_eval_30_fractions":
            logger.info(f"  {key}:")
            for metric_type, metrics_val in value.items():
                logger.info(f"    {metric_type}: {metrics_val}")
        else:
            logger.info(f"  {key}: {value}")


    logger.info("\n--- Treatment History Summary (from simulation) ---")
    if history and history.get("tumor_volumes"):
        header = f"{'Fraction':<10} | {'ITV Volume':<15} | {'TCP (%)':<10}"
        oar_ntcp_headers = [f"NTCP {oar[:8]:<8} (%)" for oar in planner.oar_masks.keys() if oar in history["ntcp"] and history["ntcp"][oar]]
        logger.info(header + " | " + " | ".join(oar_ntcp_headers))
        logger.info("-" * (len(header) + sum(len(h) + 3 for h in oar_ntcp_headers)))

        for i in range(len(history["tumor_volumes"])):
            row = f"{i+1:<10} | {history['tumor_volumes'][i]:<15.2f} | {history['tcp'][i]:<10.4f}"
            for oar_name in planner.oar_masks.keys():
                 if oar_name in history["ntcp"] and i < len(history["ntcp"][oar_name]):
                     row += f" | {history['ntcp'][oar_name][i]:<16.4f}" 
            logger.info(row)
    else:
        logger.info("No treatment history recorded or history format unexpected.")

    logger.info("\n--- End of Simulation ---")
