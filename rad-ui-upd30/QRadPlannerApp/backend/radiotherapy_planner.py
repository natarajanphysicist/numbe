#--- START OF FILE radiotherapy_planner.py ---

# q_rad_plan_may-13_enhanced_motion.py

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy.ndimage import convolve, map_coordinates, zoom 
from scipy.special import erf
import matplotlib.pyplot as plt 
import pydicom
import os
from numba import jit, prange
from scipy.optimize import minimize
from skimage.measure import label 

logger = logging.getLogger(__name__)

@jit(nopython=True, parallel=True)
def calculate_primary_fluence_numba(source, direction, density_grid, grid_shape):
    fluence = np.zeros(grid_shape, dtype=np.float32) 
    for i in prange(grid_shape[0]):
        for j in range(grid_shape[1]):
            for k in range(grid_shape[2]):
                r = np.array([i, j, k], dtype=np.float64)
                dist = np.linalg.norm(r - source)
                if dist == 0:
                    continue
                num_ray_points = 10 
                ray_path = np.empty((num_ray_points, source.shape[0]), dtype=np.float64)
                for dim_idx in range(source.shape[0]):
                    ray_path[:, dim_idx] = np.linspace(source[dim_idx], r[dim_idx], num=num_ray_points)
                path_density = 0.0
                for p_idx in range(num_ray_points):
                    p_coord = ray_path[p_idx]
                    idx = np.round(p_coord).astype(np.int32)
                    if (0 <= idx[0] < grid_shape[0] and 0 <= idx[1] < grid_shape[1] and 0 <= idx[2] < grid_shape[2]):
                        path_density += density_grid[idx[0], idx[1], idx[2]]
                mu = 0.02 
                fluence[i, j, k] = (1 / (dist**2 + 1e-9)) * np.exp(-mu * path_density)
    return fluence

class QRadPlan3D:
    def __init__(self, grid_size=(100, 100, 100), num_beams=8, kernel_path="dose_kernel.npy",
                 dicom_rt_struct_path=None, ct_path=None, fourd_ct_path=None,
                 reference_phase_name="phase_0",
                 patient_params=None, dir_method='simplified_sinusoidal'):

        self.grid_size = grid_size 
        try:
            self.dose_kernel = np.load(kernel_path).astype(np.float32)
            logger.info(f"Dose kernel loaded from {kernel_path} and cast to float32.")
        except FileNotFoundError:
            logger.error(f"Dose kernel file {kernel_path} not found. Planning will likely fail.")
            if os.path.exists(os.path.join(os.path.dirname(__file__), "generate_dose_kernel.py")):
                 try:
                    from .generate_dose_kernel import generate_updated_dose_kernel 
                    logger.info("Attempting to generate dose kernel...")
                    kernel_data = generate_updated_dose_kernel()
                    np.save(kernel_path, kernel_data) 
                    self.dose_kernel = kernel_data.astype(np.float32)
                    logger.info(f"Dose kernel generated and saved to {kernel_path}, cast to float32.")
                 except Exception as e_gen:
                    logger.error(f"Failed to generate dose kernel: {e_gen}. Please provide a valid kernel.")
                    self.dose_kernel = np.ones((21,21,21), dtype=np.float32) / (21**3) 
            else:
                logger.error("generate_dose_kernel.py not found. Using a placeholder kernel.")
                self.dose_kernel = np.ones((21,21,21), dtype=np.float32) / (21**3) 

        self.beam_directions = self._generate_beam_directions(num_beams)
        self.dir_method = dir_method
        self.reference_phase_name = reference_phase_name
        self.tumor_mask_name = "Tumor" 
        self.beam_weights: Optional[np.ndarray] = None

        self.density_grid: Optional[np.ndarray] = None
        self.tumor_mask: Optional[np.ndarray] = None # Planner internal orientation (cols,rows,slices)
        self.oar_masks: Dict[str, np.ndarray] = {} # Planner internal orientation (cols,rows,slices)
        self.density_grids_phases: List[np.ndarray] = []
        self.tumor_masks_phases: List[np.ndarray] = [] # Planner internal orientation (cols,rows,slices)
        self.oar_masks_phases: List[Dict[str, np.ndarray]] = [] # Planner internal orientation (cols,rows,slices)
        self.affine_transforms: List[np.ndarray] = []
        self.voxel_volume = 0.001 
        self.voxel_size_mm = np.array([1.0, 1.0, 1.0], dtype=np.float32) 

        self.num_phases = 1 
        self.respiratory_phase_weights = np.array([1.0], dtype=np.float32)
        self.accumulated_dose = np.zeros(self.grid_size, dtype=np.float32)
        self.tcp_value: Optional[float] = None
        self.ntcp_values: Dict[str, float] = {}
        self.dose_distribution: Optional[np.ndarray] = None

        if fourd_ct_path and dicom_rt_struct_path:
            logger.info(f"Attempting to load 4D CT data from: {fourd_ct_path}")
            try:
                loaded_data = self._load_4d_ct_data(fourd_ct_path, dicom_rt_struct_path)
                if loaded_data:
                    self.density_grids_phases, self.tumor_masks_phases, self.oar_masks_phases, self.affine_transforms = loaded_data
                    if self.density_grids_phases:
                        self.grid_size = self.density_grids_phases[0].shape # Should be (c,r,s)
                        self.accumulated_dose = np.zeros(self.grid_size, dtype=np.float32)
                        self.density_grid = np.mean(self.density_grids_phases, axis=0, dtype=np.float32)
                    self.num_phases = len(self.density_grids_phases) if self.density_grids_phases else 1
                    self.respiratory_phase_weights = np.ones(self.num_phases, dtype=np.float32) / self.num_phases if self.num_phases > 0 else np.array([1.0], dtype=np.float32)
                    logger.info(f"Successfully loaded {self.num_phases} respiratory phases from 4D CT.")
                    if self.tumor_masks_phases:
                        self.tumor_mask = np.any(self.tumor_masks_phases, axis=0) # ITV from phases
                    if self.oar_masks_phases and isinstance(self.oar_masks_phases[0], dict):
                        all_oar_names = set(key for phase_oars in self.oar_masks_phases for key in phase_oars)
                        self.oar_masks = {
                            oar_name: np.any([phase_oars.get(oar_name, np.zeros(self.grid_size, dtype=bool))
                                              for phase_oars in self.oar_masks_phases], axis=0)
                            for oar_name in all_oar_names
                        } # Union of OARs over phases
                else:
                    logger.error("Failed to load 4D CT data, _load_4d_ct_data returned None.")
            except Exception as e:
                logger.error(f"Error loading 4D DICOM data: {e}. Planner will use simplified model if data not set externally.", exc_info=True)
        
        elif dicom_rt_struct_path and ct_path:
            logger.info(f"Attempting to load static 3D CT data from: {ct_path} and RTStruct: {dicom_rt_struct_path}")
            try:
                ref_ct_series = self._load_ct_series(ct_path) # This sets self.voxel_size_mm, self.voxel_volume
                if ref_ct_series is None or not ref_ct_series:
                    raise ValueError("Failed to load static CT series.")

                rows_ct = int(ref_ct_series[0].Rows)
                cols_ct = int(ref_ct_series[0].Columns)
                num_slices_ct = len(ref_ct_series)
                self.grid_size = (cols_ct, rows_ct, num_slices_ct) 
                self.accumulated_dose = np.zeros(self.grid_size, dtype=np.float32)

                ct_pixel_data_list = [s.pixel_array.astype(np.float32) * float(s.RescaleSlope) + float(s.RescaleIntercept) for s in ref_ct_series]
                ct_volume_data_zyx = np.stack(ct_pixel_data_list, axis=0) # (slices, rows, cols)
                ct_volume_data_crs = ct_volume_data_zyx.transpose(2, 1, 0) # (cols, rows, slices) for planner
                self.density_grid = self._hu_to_density(ct_volume_data_crs)
                
                # _load_rt_struct should return masks in planner orientation (cols, rows, slices)
                tumor_mask_crs, oar_masks_crs, roi_names_map = self._load_rt_struct(dicom_rt_struct_path, ref_ct_series)
                
                self.tumor_mask_name = "Tumor" 
                for roi_num_str, name_roi in roi_names_map.items(): 
                    if any(kw in name_roi.lower() for kw in ["tumor", "gtv", "ptv"]):
                        self.tumor_mask_name = name_roi
                        break
                
                self.num_phases = 1
                self.respiratory_phase_weights = np.array([1.0], dtype=np.float32)
                self.density_grids_phases = [self.density_grid.copy()]
                self.tumor_mask = tumor_mask_crs if tumor_mask_crs is not None else np.zeros(self.grid_size, dtype=bool)
                self.tumor_masks_phases = [self.tumor_mask.copy()]
                self.oar_masks = oar_masks_crs if oar_masks_crs is not None else {}
                self.oar_masks_phases = [self.oar_masks.copy()]
                self.affine_transforms = [np.eye(4, dtype=np.float32)]
                logger.info(f"Static 3D CT/RTStruct loaded. Grid (c,r,s): {self.grid_size}. OARs: {list(self.oar_masks.keys())}")
            except Exception as e:
                logger.error(f"Error loading static DICOM data: {e}. Planner will use simplified model.", exc_info=True)
        else: 
             logger.info(f"QRadPlan3D initialized. Grid (c,r,s)={self.grid_size}. Use set_patient_data() or planner uses simplified model.")

        default_params = {
            "tumor": {"alpha": 0.3, "beta": 0.03, "alpha_beta": 10, "N0_density": 1e7}, 
            "lung": {"alpha_beta": 3, "TD50": 24.5, "m": 0.3, "n": 1.0}, 
            "heart": {"alpha_beta": 3, "TD50": 40, "m": 0.1, "n": 0.5} 
        }
        self.radiobiological_params = patient_params if patient_params is not None else default_params
        
        if np.array_equal(self.voxel_size_mm, np.array([1.0,1.0,1.0])) and self.voxel_volume == 0.001:
            logger.info(f"Voxel info not from DICOM yet, using default {self.voxel_volume} cm^3.")

    def set_patient_data(self, ct_volume_hu_zyx: np.ndarray, 
                         image_properties: Dict, 
                         tumor_mask_detected_zyx: Optional[np.ndarray] = None, 
                         oar_masks_loaded_zyx: Optional[Dict[str, np.ndarray]] = None): 
        logger.info("QRadPlan3D: Setting patient data. Volume, tumor, and OARs (if provided).")

        slices_zyx, rows_zyx, cols_zyx = ct_volume_hu_zyx.shape
        self.grid_size = (cols_zyx, rows_zyx, slices_zyx) # Planner grid: (cols, rows, slices)
        logger.info(f"Planner grid_size (c,r,s) updated to: {self.grid_size} from provided CT volume (s,r,c): {ct_volume_hu_zyx.shape}.")
        self.accumulated_dose = np.zeros(self.grid_size, dtype=np.float32)

        ct_volume_planner_oriented_crs = np.transpose(ct_volume_hu_zyx, (2, 1, 0)).astype(np.float32) # c,r,s
        self.density_grid = self._hu_to_density(ct_volume_planner_oriented_crs)
        
        self.num_phases = 1
        self.respiratory_phase_weights = np.array([1.0], dtype=np.float32)
        self.density_grids_phases = [self.density_grid.copy()]
        self.affine_transforms = [np.eye(4, dtype=np.float32)]

        if tumor_mask_detected_zyx is not None:
            tumor_mask_planner_oriented_crs = np.transpose(tumor_mask_detected_zyx, (2, 1, 0)).astype(bool) # c,r,s
            self.set_tumor_data(tumor_mask_input_crs=tumor_mask_planner_oriented_crs) # Expects (c,r,s)
        else:
            logger.warning("No tumor mask provided to set_patient_data. Tumor target may be undefined.")
            self.tumor_mask = None 
            self.tumor_masks_phases = []

        self.oar_masks.clear() 
        if oar_masks_loaded_zyx:
            for name, mask_data_zyx in oar_masks_loaded_zyx.items():
                if mask_data_zyx.shape == (slices_zyx, rows_zyx, cols_zyx): 
                    self.oar_masks[name] = np.transpose(mask_data_zyx, (2, 1, 0)).astype(bool) # c,r,s
                else:
                    logger.warning(f"OAR '{name}' shape {mask_data_zyx.shape} "
                                   f"mismatches volume ZYX shape {(slices_zyx,rows_zyx,cols_zyx)}. Skipping OAR.")
            logger.info(f"Set OAR masks (stored as c,r,s): {list(self.oar_masks.keys())}")
        else:
            logger.info("No OAR masks provided or loaded to set_patient_data.")
        self.oar_masks_phases = [self.oar_masks.copy()]

        try: 
            # image_properties['pixel_spacing'] is [row_spacing, col_spacing]
            spacing_xyz_planner = [
                image_properties.get('pixel_spacing', [1.0, 1.0])[1],  # col_spacing (planner x)
                image_properties.get('pixel_spacing', [1.0, 1.0])[0],  # row_spacing (planner y)
                image_properties.get('slice_thickness', 1.0)           # slice_thk (planner z)
            ]
            self.voxel_size_mm = np.array(spacing_xyz_planner, dtype=np.float32) # (col, row, slice_thk)
            self.voxel_volume = np.prod(self.voxel_size_mm) * 1e-3
            logger.info(f"Voxel props set: size_mm (c,r,s)={self.voxel_size_mm}, vol_cm3={self.voxel_volume:.4e}")
        except Exception as e:
            logger.error(f"Error setting voxel properties: {e}. Using defaults.", exc_info=True)
            self.voxel_size_mm = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            self.voxel_volume = 0.001

    def _ensure_data_loaded(self):
        if self.density_grid is None or self.tumor_mask is None:
            logger.warning("Essential data (density/tumor) not loaded. Initializing simplified model.")
            self._initialize_simplified_model(self.grid_size) 

    def _initialize_simplified_model(self, grid_size_param_crs):
        self.grid_size = grid_size_param_crs # (cols, rows, slices)
        self.num_phases = 1 
        self.respiratory_phase_weights = np.array([1.0], dtype=np.float32)
        
        if self.density_grid is None: self.density_grid = np.ones(self.grid_size, dtype=np.float32)
        if not self.density_grids_phases: self.density_grids_phases = [self.density_grid.copy()]

        tumor_center_crs = np.array(self.grid_size, dtype=float) / 2.0 # (cx, cy, cz)
        if self.tumor_mask is None:
            self.tumor_mask = self._create_spherical_mask(tumor_center_crs, 10) # Expects planner coords (c,r,s)
            self.tumor_mask_name = "Simulated Tumor"
        if not self.tumor_masks_phases: self.tumor_masks_phases = [self.tumor_mask.copy()]

        if not self.oar_masks: 
            # These create_xxx_mask methods use self.grid_size (c,r,s) and centers in (c,r,s)
            self.oar_masks = {
                "Simulated Lung": self._create_ellipsoid_mask(tumor_center_crs + np.array([20, 0, 0]), (30, 20, 20)),
                "Simulated Heart": self._create_spherical_mask(tumor_center_crs + np.array([-20, 0, 0]), 15)
            }
        if not self.oar_masks_phases: # oar_masks is already dict of (c,r,s) masks
            self.oar_masks_phases = [self.oar_masks.copy()] 
        
        if not self.affine_transforms: self.affine_transforms = [np.eye(4, dtype=np.float32)]
        self.accumulated_dose = np.zeros(self.grid_size, dtype=np.float32) 
        
        # Ensure simplified model uses consistent voxel properties if not set by DICOM
        if np.array_equal(self.voxel_size_mm, np.array([1.0,1.0,1.0])) and self.voxel_volume == 0.001:
            self.voxel_size_mm = np.array([1.0, 1.0, 1.0], dtype=np.float32) # (col, row, slice_thk)
            self.voxel_volume = 1.0 * 1.0 * 1.0 * 1e-3 # cm^3
        logger.info(f"Initialized/completed with simplified model components. Grid (c,r,s): {self.grid_size}")

    def set_tumor_data(self, tumor_center_crs=None, tumor_size_crs=None, tumor_mask_input_crs=None): # Suffix indicates planner coords
        try:
            current_grid_size_crs = self.grid_size 
            if tumor_mask_input_crs is not None: # Expected as (cols, rows, slices)
                if tumor_mask_input_crs.shape != current_grid_size_crs:
                    logger.warning(f"Resizing input tumor mask from {tumor_mask_input_crs.shape} to {current_grid_size_crs}")
                    zoom_factors = tuple(gs_d / tm_d for gs_d, tm_d in zip(current_grid_size_crs, tumor_mask_input_crs.shape))
                    resized_mask_float = zoom(tumor_mask_input_crs.astype(float), zoom_factors, order=0, mode='constant', cval=0.0) 
                    resized_mask_bool = resized_mask_float.astype(bool)
                    if resized_mask_bool.shape != current_grid_size_crs: 
                         logger.warning(f"Resized mask shape {resized_mask_bool.shape} mismatch. Cropping/padding.")
                         # Simplified crop/pad
                         final_mask = np.zeros(current_grid_size_crs, dtype=bool)
                         s = tuple(slice(0, min(g,r)) for g,r in zip(current_grid_size_crs, resized_mask_bool.shape))
                         final_mask[s] = resized_mask_bool[s]
                         self.tumor_mask = final_mask
                    else: self.tumor_mask = resized_mask_bool
                else: self.tumor_mask = tumor_mask_input_crs.astype(bool)
            elif tumor_center_crs is not None and tumor_size_crs is not None: # tumor_center_crs and tumor_size_crs are in planner (c,r,s) voxels
                center_arr_crs = np.asarray(tumor_center_crs)
                if isinstance(tumor_size_crs, (int, float)): # radius
                    self.tumor_mask = self._create_spherical_mask(center_arr_crs, tumor_size_crs) # Uses self.grid_size (c,r,s)
                else: # radii (rx, ry, rz)
                    self.tumor_mask = self._create_ellipsoid_mask(center_arr_crs, tumor_size_crs)
            else:
                if self.tumor_mask is None: 
                     logger.warning("No tumor data provided & no existing mask. Initializing simplified tumor.")
                     self._ensure_data_loaded() 
                     if self.tumor_mask is None: raise ValueError("Failed to initialize a tumor mask.")
                else: logger.info("Using existing tumor mask.")

            if self.tumor_mask is not None:
                self.tumor_masks_phases = [self.tumor_mask.copy() for _ in range(self.num_phases)]
            
            if self.tumor_masks_phases and self.num_phases > 1 : 
                 itv_mask = np.any(self.tumor_masks_phases, axis=0)
                 if np.any(itv_mask): self.tumor_mask = itv_mask
            elif self.tumor_mask is None and self.tumor_masks_phases: 
                 self.tumor_mask = self.tumor_masks_phases[0]

            if self.tumor_mask is not None:
                logger.info(f"Tumor data set. Planner ITV shape (c,r,s): {self.tumor_mask.shape}, Voxels: {np.sum(self.tumor_mask)}")
            else: logger.error("Tumor mask is still None after set_tumor_data.")
            return True
        except Exception as e:
            logger.error(f"Error in set_tumor_data: {e}", exc_info=True); return False

    def _generate_beam_directions(self, num_beams: int) -> List[Tuple[float, float, float]]:
        logger.debug(f"Generating {num_beams} simplified beam directions.")
        directions = []
        for i in range(num_beams):
            angle = 2 * np.pi * i / num_beams
            directions.append((np.cos(angle), np.sin(angle), 0.0)) 
        return [tuple(d / (np.linalg.norm(d) + 1e-9)) for d in directions]

    def _load_4d_ct_data(self, fourd_ct_path: str, dicom_rt_struct_path: str) -> Optional[Tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, np.ndarray]], List[np.ndarray]]]:
        logger.warning(f"_load_4d_ct_data is placeholder. Paths: {fourd_ct_path}, {dicom_rt_struct_path}")
        # This should return lists of density_grids_crs, tumor_masks_crs, oar_masks_phases_crs, affine_transforms
        # For now, returns simplified data assuming self.grid_size is already (c,r,s)
        num_placeholder_phases = 2
        placeholder_densities_crs = [np.ones(self.grid_size, dtype=np.float32) for _ in range(num_placeholder_phases)]
        placeholder_tumors_crs = [self._create_spherical_mask(np.array(self.grid_size)/2, 10+i*2) for i in range(num_placeholder_phases)]
        placeholder_oars_list_crs = []
        for _ in range(num_placeholder_phases):
            oars_crs = {"SampleOAR_4D": self._create_spherical_mask(np.array(self.grid_size)/3, 15)}
            placeholder_oars_list_crs.append(oars_crs)
        placeholder_affines = [np.eye(4, dtype=np.float32) for _ in range(num_placeholder_phases)]
        # Voxel size should be set from actual DICOM data if this were a real implementation
        logger.info(f"Placeholder _load_4d_ct_data using voxel size: {self.voxel_size_mm} and volume: {self.voxel_volume}")
        return placeholder_densities_crs, placeholder_tumors_crs, placeholder_oars_list_crs, placeholder_affines

    def _hu_to_density(self, hu_array_crs: np.ndarray) -> np.ndarray: # Expects (c,r,s)
        logger.debug("Converting HU to density using simple ramp.")
        density_array = np.ones_like(hu_array_crs, dtype=np.float32) 
        density_array[hu_array_crs <= -1000] = 0.001 
        density_array[hu_array_crs > 0] = 1.0 + (hu_array_crs[hu_array_crs > 0] / 1000.0) * 0.1 
        density_array[hu_array_crs > 1000] = 1.1 + (hu_array_crs[hu_array_crs > 1000] - 1000) / 1000.0 * 0.5 
        return np.clip(density_array, 0.001, 3.0)

    def _load_rt_struct(self, rt_struct_path: str, ct_series: List[pydicom.Dataset]) -> Tuple[Optional[np.ndarray], Optional[Dict[str, np.ndarray]], Dict[str, str]]:
        logger.warning(f"_load_rt_struct is placeholder. Path: {rt_struct_path}")
        # This should return tumor_mask_crs, oar_masks_crs, roi_names_map
        # where masks are in planner (cols, rows, slices) orientation.
        if not ct_series: logger.error("CT series empty in _load_rt_struct placeholder."); return None, None, {}
        rows_ct, cols_ct, num_slices_ct = int(ct_series[0].Rows), int(ct_series[0].Columns), len(ct_series)
        grid_shape_crs = (cols_ct, rows_ct, num_slices_ct)
        
        tumor_center_crs = np.array(grid_shape_crs, dtype=float) / 2.0
        tumor_mask_crs = self._create_spherical_mask(tumor_center_crs, radius=10, grid_shape_override=grid_shape_crs)
        oar_masks_crs = {"SampleOAR_Static_CRS": self._create_spherical_mask(tumor_center_crs + np.array([20,0,0]), radius=15, grid_shape_override=grid_shape_crs)}
        roi_names_map = {"1": "Simulated Tumor", "2": "SampleOAR_Static_CRS"} # Placeholder mapping

        # Voxel properties should be set by _load_ct_series, but confirm/set if needed.
        # self.voxel_size_mm and self.voxel_volume are based on (col,row,slice_thk)
        return tumor_mask_crs, oar_masks_crs, roi_names_map

    def _create_spherical_mask(self, center_crs: np.ndarray, radius: float, grid_shape_override: Optional[Tuple[int,int,int]] = None) -> np.ndarray:
        grid_crs = grid_shape_override if grid_shape_override is not None else self.grid_size # (cols,rows,slices)
        coords_x, coords_y, coords_z = np.ogrid[:grid_crs[0], :grid_crs[1], :grid_crs[2]]
        distance_sq = (coords_x - center_crs[0])**2 + (coords_y - center_crs[1])**2 + (coords_z - center_crs[2])**2
        return (distance_sq <= radius**2).astype(bool)

    def _create_ellipsoid_mask(self, center_crs: np.ndarray, radii_crs: Tuple[float,float,float], grid_shape_override: Optional[Tuple[int,int,int]] = None) -> np.ndarray:
        grid_crs = grid_shape_override if grid_shape_override is not None else self.grid_size
        coords_x, coords_y, coords_z = np.ogrid[:grid_crs[0], :grid_crs[1], :grid_crs[2]]
        distance_norm_sq = ((coords_x - center_crs[0]) / (radii_crs[0] + 1e-9))**2 + \
                           ((coords_y - center_crs[1]) / (radii_crs[1] + 1e-9))**2 + \
                           ((coords_z - center_crs[2]) / (radii_crs[2] + 1e-9))**2
        return (distance_norm_sq <= 1.0).astype(bool)

    def _get_source_position(self, direction_norm: Tuple[float, float, float]) -> np.ndarray: # direction is normalized
        grid_center_crs = np.array(self.grid_size) / 2.0
        source_distance = 5 * np.max(self.grid_size) 
        source_pos_crs = grid_center_crs - np.array(direction_norm) * source_distance
        return source_pos_crs.astype(np.float64)

    def _calculate_tcp(self, dose_grid_crs: np.ndarray) -> float: # dose_grid is (c,r,s)
        logger.debug("Calculating TCP (simplified placeholder).")
        if self.tumor_mask is None or not np.any(self.tumor_mask): return 0.0
        mean_tumor_dose = np.mean(dose_grid_crs[self.tumor_mask]) if np.any(self.tumor_mask) else 0
        D50, gamma = 50.0, 1.5
        if D50 <= 0: return 0.0
        tcp = 1.0 / (1.0 + (D50 / (mean_tumor_dose + 1e-9))**(4 * gamma / np.log(3)))
        return float(np.clip(tcp, 0.0, 1.0))

    def _calculate_ntcp(self, dose_grid_crs: np.ndarray, oar_name: str) -> float: # dose_grid (c,r,s)
        logger.debug(f"Calculating NTCP for {oar_name} (simplified placeholder).")
        if oar_name not in self.oar_masks or self.oar_masks[oar_name] is None or not np.any(self.oar_masks[oar_name]):
            return 0.0
        oar_mask_crs = self.oar_masks[oar_name] # (c,r,s)
        mean_oar_dose = np.mean(dose_grid_crs[oar_mask_crs]) if np.any(oar_mask_crs) else 0.0
        
        oar_key = oar_name.lower()
        oar_params = self.radiobiological_params.get(oar_key, self.radiobiological_params.get(oar_name, {}))
        TD50 = oar_params.get("TD50", 30); m = oar_params.get("m", 0.25)
        if not oar_params: logger.warning(f"No radiobio params for OAR: {oar_name}. Using generic.")
        if TD50 <= 0: return 0.0
        
        t_val = (mean_oar_dose - TD50) / (m * TD50 + 1e-9)
        ntcp_val = 0.5 * (1 + erf(t_val / np.sqrt(2)))
        return float(np.clip(ntcp_val, 0.0, 1.0))

    def _load_ct_series(self, ct_dir_path: str) -> Optional[List[pydicom.Dataset]]:
        logger.info(f"Loading CT series from: {ct_dir_path}")
        if not os.path.isdir(ct_dir_path): logger.error(f"CT dir not found: {ct_dir_path}"); return None
        dicom_files_ds = []
        for f_name in os.listdir(ct_dir_path):
            if f_name.endswith('.dcm'):
                try:
                    ds = pydicom.dcmread(os.path.join(ct_dir_path, f_name))
                    if hasattr(ds, 'SOPClassUID') and ds.SOPClassUID == pydicom.uid.CTImageStorage:
                         dicom_files_ds.append(ds)
                except Exception as e: logger.warning(f"Could not read/parse DICOM {f_name}: {e}")
        
        ct_series = [ds for ds in dicom_files_ds if hasattr(ds, 'ImagePositionPatient') and hasattr(ds, 'InstanceNumber')]
        if not ct_series: logger.warning(f"No valid CT images found in {ct_dir_path}"); return None

        if all(hasattr(s, 'SliceLocation') for s in ct_series):
            try: ct_series.sort(key=lambda s: float(s.SliceLocation))
            except ValueError: ct_series.sort(key=lambda s: int(s.InstanceNumber))
        elif all(hasattr(s, 'InstanceNumber') for s in ct_series): ct_series.sort(key=lambda s: int(s.InstanceNumber))
        else: 
            try: ct_series.sort(key=lambda s: float(s.ImagePositionPatient[2]))
            except Exception as e_sort: logger.error(f"Could not sort CT slices: {e_sort}"); return None
        logger.info(f"Loaded and sorted {len(ct_series)} CT slices.")
        try:
            # PixelSpacing from DICOM is [RowSpacing, ColumnSpacing]
            # VoxelSizeMM for planner is [ColSpacing, RowSpacing, SliceThickness]
            ps_dicom = np.array(ct_series[0].PixelSpacing, dtype=float)
            st_dicom = float(ct_series[0].SliceThickness)
            self.voxel_size_mm = np.array([ps_dicom[1], ps_dicom[0], st_dicom], dtype=np.float32)
            self.voxel_volume = np.prod(self.voxel_size_mm) * 1e-3 # cm^3
            logger.info(f"Derived voxel size (c,r,s in mm): {self.voxel_size_mm}, Voxel volume (cm^3): {self.voxel_volume:.4e}")
        except Exception as e:
            logger.warning(f"Could not derive voxel size from CT: {e}. Using defaults.", exc_info=True)
            self.voxel_size_mm = np.array([1.0, 1.0, 1.0], dtype=np.float32); self.voxel_volume = 0.001
        return ct_series

    def calculate_dose(self, beam_weights_in: np.ndarray) -> np.ndarray: # beam_weights_in is alias for beam_weights
        self._ensure_data_loaded() 
        logger.info(f"CALC_DOSE: Starting dose calculation. Input beam_weights_in: {beam_weights_in}")
        logger.info(f"CALC_DOSE: Dose kernel stats: min={self.dose_kernel.min():.4e}, max={self.dose_kernel.max():.4e}, mean={self.dose_kernel.mean():.4e}, shape={self.dose_kernel.shape}")

        final_dose_crs = np.zeros(self.grid_size, dtype=np.float32) 
        if not self.density_grids_phases: 
            logger.error("CALC_DOSE: No density grids for dose calculation.")
            if self.density_grid is not None: 
                logger.warning("CALC_DOSE: Using average density_grid for dose calc as phases missing.")
                self.density_grids_phases = [self.density_grid.copy()]
                self.num_phases = 1; self.respiratory_phase_weights = np.array([1.0], dtype=np.float32)
            else: return final_dose_crs

        if len(beam_weights_in) != len(self.beam_directions):
            logger.error(f"CALC_DOSE: Beam weights length {len(beam_weights_in)} and beam_directions length {len(self.beam_directions)} mismatch. Returning zero dose.")
            return final_dose_crs

        for phase_idx in range(self.num_phases):
            logger.info(f"CALC_DOSE: Processing phase {phase_idx + 1}/{self.num_phases}")
            phase_dose_contrib_crs = np.zeros(self.grid_size, dtype=np.float32) 
            current_density_grid_crs = self.density_grids_phases[phase_idx]
            logger.info(f"CALC_DOSE: Phase {phase_idx} density grid stats: min={current_density_grid_crs.min():.4f}, max={current_density_grid_crs.max():.4f}, mean={current_density_grid_crs.mean():.4f}")

            mean_density_phase = np.mean(current_density_grid_crs[current_density_grid_crs > 1e-6])
            if not (mean_density_phase > 1e-6):
                mean_density_phase = 1.0
                logger.warning(f"CALC_DOSE: Phase {phase_idx} mean_density_phase was <= 1e-6, set to 1.0 to avoid division by zero.")
            logger.info(f"CALC_DOSE: Phase {phase_idx} mean_density_phase for scaling: {mean_density_phase:.4f}")

            for i, (direction, weight) in enumerate(zip(self.beam_directions, beam_weights_in)):
                logger.info(f"CALC_DOSE: Phase {phase_idx}, Beam {i}: weight={weight:.4f}, direction=({direction[0]:.2f},{direction[1]:.2f},{direction[2]:.2f})")
                if weight == 0:
                    logger.info(f"CALC_DOSE: Phase {phase_idx}, Beam {i}: weight is 0, skipping.")
                    continue

                source_crs = self._get_source_position(direction)
                logger.info(f"CALC_DOSE: Phase {phase_idx}, Beam {i}: source_crs=({source_crs[0]:.1f},{source_crs[1]:.1f},{source_crs[2]:.1f})")

                fluence_crs = calculate_primary_fluence_numba(source_crs, np.array(direction, dtype=np.float64), current_density_grid_crs, self.grid_size)
                logger.info(f"CALC_DOSE: Phase {phase_idx}, Beam {i}: fluence_crs stats: min={fluence_crs.min():.4e}, max={fluence_crs.max():.4e}, mean={fluence_crs.mean():.4e}, sum={np.sum(fluence_crs):.4e}")

                if np.sum(fluence_crs) == 0:
                    logger.warning(f"CALC_DOSE: Phase {phase_idx}, Beam {i}: Fluence is all zero. Partial dose will be zero.")

                partial_dose_beam_crs = convolve(fluence_crs, self.dose_kernel, mode="constant", cval=0.0)
                logger.info(f"CALC_DOSE: Phase {phase_idx}, Beam {i}: after convolve stats: min={partial_dose_beam_crs.min():.4e}, max={partial_dose_beam_crs.max():.4e}, mean={partial_dose_beam_crs.mean():.4e}, sum={np.sum(partial_dose_beam_crs):.4e}")

                # Store pre-density scaling stats
                pre_density_scale_sum = np.sum(partial_dose_beam_crs)

                partial_dose_beam_crs *= (current_density_grid_crs / mean_density_phase)
                logger.info(f"CALC_DOSE: Phase {phase_idx}, Beam {i}: after density scaling stats: min={partial_dose_beam_crs.min():.4e}, max={partial_dose_beam_crs.max():.4e}, mean={partial_dose_beam_crs.mean():.4e}, sum={np.sum(partial_dose_beam_crs):.4e}")

                if pre_density_scale_sum > 0 and np.sum(partial_dose_beam_crs) == 0:
                    logger.warning(f"CALC_DOSE: Phase {phase_idx}, Beam {i}: Dose became zero after density scaling. Check density grid values relative to mean_density_phase.")

                phase_dose_contrib_crs += weight * partial_dose_beam_crs
                logger.info(f"CALC_DOSE: Phase {phase_idx}, Beam {i}: updated phase_dose_contrib_crs sum={np.sum(phase_dose_contrib_crs):.4e}")

            final_dose_crs += self.respiratory_phase_weights[phase_idx] * phase_dose_contrib_crs
            logger.info(f"CALC_DOSE: Phase {phase_idx} contribution added. Current final_dose_crs sum={np.sum(final_dose_crs):.4e}")

        logger.info(f"CALC_DOSE: Before normalization: final_dose_crs stats: min={final_dose_crs.min():.4e}, max={final_dose_crs.max():.4e}, mean={final_dose_crs.mean():.4e}, sum={np.sum(final_dose_crs):.4e}")
        
        tumor_vol_vox = np.sum(self.tumor_mask) if self.tumor_mask is not None else 0
        logger.info(f"CALC_DOSE: Normalization: tumor_mask exists: {self.tumor_mask is not None}, tumor_vol_vox: {tumor_vol_vox}")

        tumor_vol_cc = tumor_vol_vox * self.voxel_volume 
        base_dose_fx = 2.0 
        if tumor_vol_cc > 0: 
            if tumor_vol_cc <= 5: base_dose_fx = 4.0 
            elif tumor_vol_cc >= 30: base_dose_fx = 2.0
            else: 
                scale = (tumor_vol_cc - 5.0) / (25.0)
                base_dose_fx = 4.0 * (1.0 - scale) + 2.0 * scale
        logger.info(f"CALC_DOSE: Normalization: base_dose_fx: {base_dose_fx:.2f} Gy (Tumor vol {tumor_vol_cc:.2f} cc).")

        max_dose_itv = 0.0
        if self.tumor_mask is not None and np.any(self.tumor_mask):
            doses_itv = final_dose_crs[self.tumor_mask]
            if doses_itv.size > 0 and np.any(doses_itv):
                max_dose_itv = np.max(doses_itv)
                logger.info(f"CALC_DOSE: Normalization: max_dose_itv found: {max_dose_itv:.4e}")
            else:
                logger.warning(f"CALC_DOSE: Normalization: No positive dose found in ITV (size {doses_itv.size}). max_dose_itv remains 0.")
        else:
            logger.warning("CALC_DOSE: Normalization: Tumor mask is None or empty. Cannot calculate max_dose_itv.")

        # Determine global max AFTER all phase contributions but BEFORE normalization
        global_max_dose_val = np.max(final_dose_crs) if np.any(final_dose_crs) else 0.0
        logger.info(f"CALC_DOSE: Normalization: global_max_dose_val before normalization: {global_max_dose_val:.4e}")


        if max_dose_itv > 1e-10: # Lowered threshold
            norm_factor = base_dose_fx / max_dose_itv
            logger.info(f"CALC_DOSE: Normalization: Normalizing to max_dose_itv ({max_dose_itv:.4e}). norm_factor = {base_dose_fx:.2f} / {max_dose_itv:.4e} = {norm_factor:.4e}")
            final_dose_crs *= norm_factor
            logger.info(f"CALC_DOSE: Dose normalized to ITV. Max ITV dose pre-norm: {max_dose_itv:.4e}, target post-norm: {base_dose_fx:.2f} Gy.")
        elif global_max_dose_val > 1e-10: # Lowered threshold
            norm_factor = base_dose_fx / global_max_dose_val
            logger.warning(f"CALC_DOSE: Normalization: No significant dose in ITV (max_dose_itv={max_dose_itv:.4e}). Normalizing to global max dose ({global_max_dose_val:.4e}). norm_factor: {norm_factor:.4e}")
            final_dose_crs *= norm_factor
        else:
            logger.warning(f"CALC_DOSE: Normalization: Max dose in ITV ({max_dose_itv:.4e}) and global max dose ({global_max_dose_val:.4e}) are both <= 1e-10. Dose remains very low or zero.")
            # final_dose_crs is not changed, effectively returning near-zero dose.

        logger.info(f"CALC_DOSE: After normalization: final_dose_crs stats: min={final_dose_crs.min():.4e}, max={final_dose_crs.max():.4e}, mean={final_dose_crs.mean():.4e}, sum={np.sum(final_dose_crs):.4e}")
        return final_dose_crs.astype(np.float32)

    def optimize_beams(self) -> np.ndarray: 
        self._ensure_data_loaded()
        logger.info(f"Starting beam optimization. Num_phases={self.num_phases}, Num_beams={len(self.beam_directions)}")
        num_beams = len(self.beam_directions)
        if num_beams == 0: logger.warning("No beams defined."); self.beam_weights=np.array([],dtype=np.float32); return self.beam_weights

        dose_influence_phases_crs = np.zeros((self.num_phases, num_beams) + self.grid_size, dtype=np.float32)
        logger.info(f"Calculating dose influence matrix...")
        for phase_idx in range(self.num_phases):
            density_grid_crs = self.density_grids_phases[phase_idx]
            mean_density = np.mean(density_grid_crs[density_grid_crs > 1e-6]); mean_density = mean_density if mean_density > 1e-6 else 1.0
            for i_beam in range(num_beams):
                source = self._get_source_position(self.beam_directions[i_beam])
                fluence = calculate_primary_fluence_numba(source, np.array(self.beam_directions[i_beam],dtype=np.float64), density_grid_crs, self.grid_size)
                partial_dose = convolve(fluence, self.dose_kernel, mode="constant", cval=0.0)
                partial_dose *= (density_grid_crs / mean_density)
                dose_influence_phases_crs[phase_idx, i_beam, :, :, :] = partial_dose
        logger.info("Dose influence matrix calculated.")
        
        def objective_fn(weights_in: np.ndarray) -> float:
            weights = np.maximum(0, weights_in)
            total_dose_avg_phases_crs = np.zeros(self.grid_size, dtype=np.float32)
            for phase_idx_obj in range(self.num_phases):
                 dose_this_phase_crs = np.tensordot(weights, dose_influence_phases_crs[phase_idx_obj], axes=([0], [0]))
                 total_dose_avg_phases_crs += self.respiratory_phase_weights[phase_idx_obj] * dose_this_phase_crs
            tumor_term = -1.0 * np.mean(total_dose_avg_phases_crs[self.tumor_mask]) if self.tumor_mask is not None and np.any(self.tumor_mask) else 0.0
            oar_term = 0.0; oar_penalty_factor = 0.5
            for oar_name, oar_mask_crs in self.oar_masks.items(): 
                 if oar_mask_crs is not None and np.any(oar_mask_crs):
                      if oar_name.lower() in self.tumor_mask_name.lower() or self.tumor_mask_name.lower() in oar_name.lower(): continue 
                      oar_term += oar_penalty_factor * np.mean(total_dose_avg_phases_crs[oar_mask_crs])
            opp_penalty = 0.0; opp_penalty_factor = 2.0
            for i in range(num_beams):
                for j in range(i + 1, num_beams):
                    angle_deg = np.degrees(np.arccos(np.clip(np.dot(self.beam_directions[i], self.beam_directions[j]), -1.0, 1.0)))
                    if angle_deg > 160: opp_penalty += opp_penalty_factor * weights[i] * weights[j] 
            return tumor_term + oar_term + opp_penalty

        logger.info("Running L-BFGS-B optimization...")
        initial_w = np.ones(num_beams, dtype=np.float32) / num_beams; bounds_w = [(0, None)] * num_beams
        result_opt = minimize(objective_fn, initial_w, method='L-BFGS-B', bounds=bounds_w, options={'maxiter': 200, 'disp': False, 'ftol': 1e-7, 'gtol': 1e-5})
        opt_cont_weights = result_opt.x
        logger.info(f"Optimization finished. Success: {result_opt.success}, Cost: {result_opt.fun:.4f}")
        
        threshold_w = np.mean(opt_cont_weights[opt_cont_weights > 1e-3]) * 0.5 if np.any(opt_cont_weights > 1e-3) else 0.1
        best_w_binary = (opt_cont_weights > threshold_w).astype(np.float32) 
        if np.sum(best_w_binary) == 0 and num_beams > 0:
            logger.warning("Optimization thresholding yielded all zero weights. Activating top N beams.")
            num_activate = min(max(1, num_beams // 4), num_beams)
            top_indices = np.argsort(opt_cont_weights)[-num_activate:]
            best_w_binary = np.zeros(num_beams, dtype=np.float32); best_w_binary[top_indices] = 1.0
        self.beam_weights = best_w_binary 
        logger.info(f"Final binary beam selection ({np.sum(self.beam_weights)} active): {self.beam_weights}"); return self.beam_weights

    def simulate_fractionated_treatment(self, num_fractions: int = 5) -> Dict:
        self._ensure_data_loaded()
        history: Dict[str, Any] = {"tumor_volumes_voxels": [], "tumor_volumes_cc": [], "tcp_fractional": [], "tcp_cumulative": [],
            "ntcp_fractional": {oar: [] for oar in self.oar_masks.keys()}, "ntcp_cumulative": {oar: [] for oar in self.oar_masks.keys()},
            "mean_tumor_dose_fractional": [], "mean_oar_doses_fractional": {oar: [] for oar in self.oar_masks.keys()}}
        
        init_tumor_vol_vox = np.sum(self.tumor_mask) if self.tumor_mask is not None else 0
        history["tumor_volumes_voxels"].append(init_tumor_vol_vox); history["tumor_volumes_cc"].append(init_tumor_vol_vox * self.voxel_volume)
        self.accumulated_dose = np.zeros(self.grid_size, dtype=np.float32) 

        if not self.tumor_masks_phases and self.tumor_mask is not None: self.tumor_masks_phases = [self.tumor_mask.copy() for _ in range(self.num_phases)]
        elif not self.tumor_masks_phases and self.tumor_mask is None: logger.error("Cannot simulate: tumor_mask and tumor_masks_phases uninitialized."); return history

        for fraction_num in range(num_fractions):
            logger.info(f"--- Simulating fraction {fraction_num + 1}/{num_fractions} ---")
            self.optimize_beams() # Sets self.beam_weights
            if self.beam_weights is None or not np.any(self.beam_weights): 
                logger.error("Beam weights not set after optimization. Cannot simulate fraction."); break 
            fraction_dose_crs = self.calculate_dose(self.beam_weights) 
            self.accumulated_dose += fraction_dose_crs
            
            alpha_t = self.radiobiological_params["tumor"]["alpha"]; beta_t = self.radiobiological_params["tumor"]["beta"]
            temp_phase_masks_crs = []
            for phase_idx_bio in range(self.num_phases): 
                current_phase_tumor_mask_crs = self.tumor_masks_phases[phase_idx_bio]
                if not np.any(current_phase_tumor_mask_crs): temp_phase_masks_crs.append(current_phase_tumor_mask_crs.copy()); continue 
                dose_in_phase_mask_crs = fraction_dose_crs[current_phase_tumor_mask_crs]
                sf_map_frac = np.exp(-(alpha_t * dose_in_phase_mask_crs + beta_t * (dose_in_phase_mask_crs**2)))
                surviving_vox_flat = (np.random.rand(*dose_in_phase_mask_crs.shape) < sf_map_frac)
                updated_phase_mask_crs = current_phase_tumor_mask_crs.copy()
                updated_phase_mask_crs[current_phase_tumor_mask_crs] = surviving_vox_flat
                temp_phase_masks_crs.append(updated_phase_mask_crs)
            self.tumor_masks_phases = temp_phase_masks_crs
            self.tumor_mask = np.any(self.tumor_masks_phases, axis=0) if self.tumor_masks_phases else np.zeros(self.grid_size, dtype=bool)
            
            curr_tumor_vol_vox = np.sum(self.tumor_mask)
            history["tumor_volumes_voxels"].append(curr_tumor_vol_vox); history["tumor_volumes_cc"].append(curr_tumor_vol_vox * self.voxel_volume)
            history["tcp_fractional"].append(self._calculate_tcp(fraction_dose_crs))
            self.tcp_value = self._calculate_tcp(self.accumulated_dose); history["tcp_cumulative"].append(self.tcp_value)
            history["mean_tumor_dose_fractional"].append(np.mean(fraction_dose_crs[self.tumor_mask]) if curr_tumor_vol_vox > 0 else 0.0)

            current_ntcp_vals = {}
            for oar_name_h in self.oar_masks.keys(): 
                history["ntcp_fractional"][oar_name_h].append(self._calculate_ntcp(fraction_dose_crs, oar_name_h))
                current_ntcp_vals[oar_name_h] = self._calculate_ntcp(self.accumulated_dose, oar_name_h)
                history["ntcp_cumulative"][oar_name_h].append(current_ntcp_vals[oar_name_h])
                history["mean_oar_doses_fractional"][oar_name_h].append(np.mean(fraction_dose_crs[self.oar_masks[oar_name_h]]) if np.any(self.oar_masks[oar_name_h]) else 0.0)
            self.ntcp_values = current_ntcp_vals

            logger.info(f"Fraction {fraction_num + 1}: ITV Vol = {history['tumor_volumes_cc'][-1]:.2f} cc, TCP (cum) = {self.tcp_value:.4f}")
            if not self.tumor_mask.any():
                logger.info("Tumor eradicated!"); break
        
        self.dose_distribution = self.accumulated_dose.copy() 
        logger.info("--- Fractionated treatment simulation finished. ---"); return history

    def get_beam_visualization_data(self) -> Optional[Dict[str, Any]]:
        """
        Provides data needed for visualizing beams in 3D.
        Returns dict with 'beam_directions', 'beam_weights', 'source_positions', 'isocenter_planner_coords'.
        Or None if essential data is missing.
        """
        if self.beam_weights is None:
            logger.warning("Beam weights not optimized. Visualization might be incomplete or misleading.")
            # Allow proceeding if other data is present, as user might want to see beam setup
            # before full optimization.
            # return None # Strict: only show if fully planned

        if self.tumor_mask is None or not np.any(self.tumor_mask):
            logger.warning("Tumor mask not defined. Using grid center as isocenter for beam visualization.")
            isocenter_planner_coords = np.array(self.grid_size, dtype=float) / 2.0 # Fallback
        else:
            coords_crs = np.argwhere(self.tumor_mask) # (N, 3) array of (col, row, slice)
            if coords_crs.size > 0:
                isocenter_planner_coords = np.mean(coords_crs, axis=0) # (mean_col, mean_row, mean_slice)
            else: # Should be caught by np.any(self.tumor_mask)
                isocenter_planner_coords = np.array(self.grid_size, dtype=float) / 2.0
        
        logger.debug(f"Isocenter for beam viz (planner coords c,r,s): {isocenter_planner_coords}")

        source_positions_crs = [self._get_source_position_for_viz(direction, isocenter_planner_coords) 
                                for direction in self.beam_directions]

        return {
            "beam_directions": self.beam_directions, # List of tuples (dx,dy,dz)
            "beam_weights": self.beam_weights if self.beam_weights is not None else np.zeros(len(self.beam_directions), dtype=np.float32),
            "source_positions_planner_coords": [pos.tolist() for pos in source_positions_crs], # List of lists [x,y,z]
            "isocenter_planner_coords": isocenter_planner_coords.tolist() # List [x,y,z]
        }

    def _get_source_position_for_viz(self, direction_vector: Tuple[float,float,float], target_point_planner_crs: np.ndarray) -> np.ndarray:
        """
        Calculates source position for visualization, relative to a target point in planner_crs.
        """
        source_distance_factor = 2.0 
        # Use actual voxel dimensions for a more physically scaled source distance
        # Voxel size is (col_mm, row_mm, slice_mm)
        # Grid size is (num_cols, num_rows, num_slices)
        # Physical grid dimensions: self.grid_size * self.voxel_size_mm
        max_physical_grid_extent = np.max(np.array(self.grid_size) * self.voxel_size_mm) if self.voxel_size_mm is not None else np.max(self.grid_size)
        
        source_pos_crs = target_point_planner_crs - np.array(direction_vector, dtype=float) * max_physical_grid_extent * source_distance_factor
        return source_pos_crs # Returns as numpy array (cx, cy, cz)


# Example usage (illustrative, typically done externally)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    
    # Test with simplified model first
    planner_test_simplified = QRadPlan3D(grid_size=(30, 30, 30), num_beams=4) # Smaller grid for faster test
    planner_test_simplified._ensure_data_loaded() # Initialize simplified model
    logger.info(f"Initial Simplified Test Planner State: Grid={planner_test_simplified.grid_size}, "
                f"TumorVol={np.sum(planner_test_simplified.tumor_mask) * planner_test_simplified.voxel_volume:.2f}cc")
    if planner_test_simplified.oar_masks: 
        logger.info(f"  Simplified OARs: {list(planner_test_simplified.oar_masks.keys())}")

    # Test beam visualization data for simplified model
    viz_data_simple = planner_test_simplified.get_beam_visualization_data()
    if viz_data_simple:
        logger.info(f"Beam Viz Data (Simple Model): Isocenter={viz_data_simple['isocenter_planner_coords']}, "
                    f"First Source={viz_data_simple['source_positions_planner_coords'][0] if viz_data_simple['source_positions_planner_coords'] else 'N/A'}")


    logger.info("\n--- Testing Planner with set_patient_data ---")
    planner_test_custom = QRadPlan3D(grid_size=(30,30,30), num_beams=4) 
    s_z, s_r, s_c = 30, 30, 30 
    test_ct_zyx = np.ones((s_z, s_r, s_c), dtype=np.float32) * 50 
    test_img_props = {
        'pixel_spacing': [1.0, 1.0], 
        'slice_thickness': 1.0,    
        'origin': [0.0, 0.0, 0.0],  
    } 
    test_tumor_zyx = np.zeros((s_z, s_r, s_c), dtype=bool)
    test_tumor_zyx[10:20, 10:20, 10:20] = True 
    test_oars_zyx = {"TestLung_ZYX": np.zeros((s_z, s_r, s_c), dtype=bool)}
    test_oars_zyx["TestLung_ZYX"][5:10, 5:25, 5:25] = True 
    
    planner_test_custom.set_patient_data(test_ct_zyx, test_img_props, test_tumor_zyx, test_oars_zyx)
    logger.info(f"After set_patient_data: Grid (c,r,s)={planner_test_custom.grid_size}, "
                f"TumorMask (c,r,s) sum={np.sum(planner_test_custom.tumor_mask)}, "
                f"VoxelVol={planner_test_custom.voxel_volume:.4e} cm^3, "
                f"VoxelSize (c,r,s) mm={planner_test_custom.voxel_size_mm}")
    if planner_test_custom.oar_masks: 
        logger.info(f"  OARs in planner (c,r,s): {list(planner_test_custom.oar_masks.keys())}, "
                    f"Lung sum: {np.sum(planner_test_custom.oar_masks.get('TestLung_ZYX', np.array([])))}")
    else:
        logger.info("  No OARs set in planner after set_patient_data.")

    # Optimize beams to get weights for visualization
    planner_test_custom.optimize_beams()
    viz_data_custom = planner_test_custom.get_beam_visualization_data()
    if viz_data_custom:
        logger.info(f"Beam Viz Data (Custom Model): Isocenter={viz_data_custom['isocenter_planner_coords']}, "
                    f"First Source={viz_data_custom['source_positions_planner_coords'][0] if viz_data_custom['source_positions_planner_coords'] else 'N/A'}, "
                    f"Weights={viz_data_custom['beam_weights']}")


    treatment_hist_test_custom = planner_test_custom.simulate_fractionated_treatment(num_fractions=2)
    logger.info(f"Custom Data Sim Results: Final TumorVol={treatment_hist_test_custom['tumor_volumes_cc'][-1]:.2f} cc, TCP={planner_test_custom.tcp_value:.4f}")
    if planner_test_custom.ntcp_values: 
        logger.info(f"  Custom Data Final NTCPs: {planner_test_custom.ntcp_values}")

    if planner_test_custom.dose_distribution is not None and np.any(planner_test_custom.dose_distribution):
        try:
            slice_idx_crs = planner_test_custom.grid_size[2] // 2 
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(planner_test_custom.dose_distribution[:, :, slice_idx_crs].T, cmap='magma', origin='lower',
                       extent=[0, planner_test_custom.grid_size[0], 0, planner_test_custom.grid_size[1]]) 
            plt.title(f"Dose Distribution (Slice Z={slice_idx_crs})")
            plt.xlabel("X-axis (cols)")
            plt.ylabel("Y-axis (rows)")
            plt.colorbar(label="Dose (Gy)")
            if planner_test_custom.tumor_mask is not None:
                 plt.contour(planner_test_custom.tumor_mask[:, :, slice_idx_crs].T, colors='cyan', levels=[0.5], 
                             linewidths=1, origin='lower', 
                             extent=[0, planner_test_custom.grid_size[0], 0, planner_test_custom.grid_size[1]])
            if planner_test_custom.oar_masks.get('TestLung_ZYX') is not None:
                 oar_mask_crs = planner_test_custom.oar_masks['TestLung_ZYX']
                 plt.contour(oar_mask_crs[:, :, slice_idx_crs].T, colors='lime', levels=[0.5],
                             linewidths=1, origin='lower',
                             extent=[0, planner_test_custom.grid_size[0], 0, planner_test_custom.grid_size[1]])
            plt.subplot(1, 2, 2)
            plt.text(0.5, 0.5, "DVH / Analysis Placeholder", ha='center', va='center')
            plt.title("Analysis Area")
            plt.tight_layout()
            plt.savefig("qradplan3d_custom_data_simulation_results.png")
            logger.info("Plot saved to qradplan3d_custom_data_simulation_results.png")
        except ImportError: logger.warning("Matplotlib not installed. Skipping dose visualization.")
        except Exception as e_plt: logger.error(f"Error during plotting: {e_plt}", exc_info=True)
    else: logger.info("No dose distribution from custom data simulation to visualize or dose is all zero.")
