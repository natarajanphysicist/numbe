# q_rad_plan_may-13_enhanced_motion.py

import numpy as np
from scipy.ndimage import convolve, map_coordinates
from scipy.special import erf
import matplotlib.pyplot as plt
import pydicom
from pydicom.fileset import FileSet # For easier DICOM series loading if needed
import os
from numba import jit, prange # numba is already in the original code
from scipy.optimize import minimize # Import classical optimizer

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


class QRadPlan3D:
    def __init__(self, grid_size=(100, 100, 100), num_beams=8, kernel_path="dose_kernel.npy",
                 dicom_rt_struct_path=None, ct_path=None, fourd_ct_path=None,
                 reference_phase_name="phase_0", # Name of the subdirectory for the reference CT phase
                 patient_params=None, dir_method='simplified_sinusoidal'): # dir_method: 'simplified_sinusoidal', 'external_sitk' (future)

        self.grid_size = grid_size # This might be overridden by DICOM data
        self.dose_kernel = np.load(kernel_path)
        self.beam_directions = self._generate_beam_directions(num_beams)
        self.dir_method = dir_method
        self.reference_phase_name = reference_phase_name

        self.num_phases = 10 # Default, can be overridden by 4D CT
        self.respiratory_phase_weights = np.ones(self.num_phases) / self.num_phases

        # Initialize accumulated dose
        self.accumulated_dose = np.zeros(self.grid_size)

        if fourd_ct_path and dicom_rt_struct_path:
            print(f"Attempting to load 4D CT data from: {fourd_ct_path}")
            print(f"Attempting to load RTStruct from: {dicom_rt_struct_path}")
            try:
                self.density_grids_phases, self.tumor_masks_phases, self.oar_masks_phases, self.affine_transforms = \
                    self._load_4d_ct_data(fourd_ct_path, dicom_rt_struct_path)
                
                # Update grid_size based on loaded CT data if necessary
                if self.density_grids_phases:
                    self.grid_size = self.density_grids_phases[0].shape
                    print(f"Grid size updated to {self.grid_size} based on loaded CT data.")
                    # Re-initialize accumulated dose with correct size
                    self.accumulated_dose = np.zeros(self.grid_size)


                self.num_phases = len(self.density_grids_phases)
                self.respiratory_phase_weights = np.ones(self.num_phases) / self.num_phases
                print(f"Successfully loaded {self.num_phases} respiratory phases.")

                self.tumor_mask = np.any(self.tumor_masks_phases, axis=0) # ITV
                # Correctly create overall OAR masks (union over phases)
                if self.oar_masks_phases and isinstance(self.oar_masks_phases[0], dict):
                     all_oar_names = set(key for phase_oars in self.oar_masks_phases for key in phase_oars)
                     self.oar_masks = {
                         oar_name: np.any([phase_oars.get(oar_name, np.zeros(self.grid_size, dtype=bool))
                                           for phase_oars in self.oar_masks_phases], axis=0)
                         for oar_name in all_oar_names
                     }
                else: # Fallback if oar_masks_phases is not as expected
                    self.oar_masks = {}

                self.density_grid = np.mean(self.density_grids_phases, axis=0) # Average density

            except Exception as e:
                print(f"Error loading 4D DICOM data: {e}. Falling back to simplified model.")
                self._initialize_simplified_model(grid_size)

        elif dicom_rt_struct_path and ct_path: # Static 3D CT case
            print(f"Attempting to load static 3D CT data from: {ct_path}")
            print(f"Attempting to load RTStruct from: {dicom_rt_struct_path}")
            try:
                # This part needs a function similar to _load_dicom_data from original,
                # but for a single CT series. For now, we'll treat it as a single phase.
                ref_ct_series = self._load_ct_series(ct_path) # Placeholder for loading one CT
                if ref_ct_series is None or not ref_ct_series:
                    raise ValueError("Failed to load static CT series.")
                
                # Assume grid_size is derived from this CT
                # (rows, cols, slices) -> (cols, rows, slices) for (x,y,z)
                rows = ref_ct_series[0].Rows
                cols = ref_ct_series[0].Columns
                num_slices = len(ref_ct_series)
                self.grid_size = (cols, rows, num_slices)
                print(f"Grid size set to {self.grid_size} from static CT.")
                # Re-initialize accumulated dose with correct size
                self.accumulated_dose = np.zeros(self.grid_size)


                ct_pixel_data = np.stack([s.pixel_array * s.RescaleSlope + s.RescaleIntercept for s in ref_ct_series], axis=-1)
                # Transpose to match (cols, rows, slices)
                ct_pixel_data = ct_pixel_data.transpose(1,0,2)

                density_grid_ref = self._hu_to_density(ct_pixel_data)
                tumor_mask_ref, oar_masks_ref, _ = self._load_rt_struct(dicom_rt_struct_path, ref_ct_series)

                self.density_grids_phases = [density_grid_ref] * self.num_phases
                self.tumor_masks_phases = [tumor_mask_ref] * self.num_phases
                self.oar_masks_phases = [oar_masks_ref] * self.num_phases
                self.affine_transforms = [np.eye(4)] * self.num_phases # Identity transform for static case

                self.tumor_mask = tumor_mask_ref
                self.oar_masks = oar_masks_ref
                self.density_grid = density_grid_ref
                print("Successfully loaded static 3D CT and RTStruct.")

            except Exception as e:
                print(f"Error loading static DICOM data: {e}. Falling back to simplified model.")
                self._initialize_simplified_model(grid_size)
        else:
            print("No DICOM paths provided. Initializing with simplified model.")
            self._initialize_simplified_model(grid_size)

        default_params = {
            "tumor": {"alpha": 0.3, "beta": 0.03, "alpha_beta": 10, "N0_density": 1e7}, # Added N0_density (cells/cm^3)
            "lung": {"alpha_beta": 3, "TD50": 24.5, "m": 0.3, "n": 1},
            "heart": {"alpha_beta": 3, "TD50": 40, "m": 0.1, "n": 0.5}
        }
        self.radiobiological_params = patient_params if patient_params else default_params
        # self.clonogenic_density = 1e7 # Moved to params dict
        # self.voxel_volume = 0.001 # Assuming 1mm^3 voxels = 0.001 cm^3
        # Need to derive voxel volume from DICOM if loaded
        self.voxel_volume = 0.001 # Default for simplified model
        if dicom_rt_struct_path or ct_path or fourd_ct_path:
             # Attempt to get voxel size from DICOM metadata if available
             try:
                 # This requires storing pixel_spacing and slice_thickness during DICOM load
                 # For simplicity, let's assume 1mm isotropic if DICOM loaded but details missing
                 # or calculate from affine if available
                 if hasattr(self, 'voxel_size_mm'): # If _load_dicom_ct or similar set this
                      self.voxel_volume = np.prod(self.voxel_size_mm) * 1e-3 # mm^3 to cm^3
                 else:
                      # Fallback: Assume 1mm isotropic if DICOM was attempted
                      self.voxel_volume = 1.0 * 1.0 * 1.0 * 1e-3 # 1mm^3 in cm^3
                      print("Warning: Could not determine voxel volume from DICOM. Assuming 1 mm^3.")
             except Exception as e:
                 print(f"Error determining voxel volume from DICOM: {e}. Assuming 1 mm^3.")
                 self.voxel_volume = 1.0 * 1.0 * 1.0 * 1e-3 # 1mm^3 in cm^3

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
                    # Resize mask if needed
                    from scipy.ndimage import zoom
                    zoom_factors = tuple(t/s for t, s in zip(self.grid_size, tumor_mask_input.shape))
                    tumor_mask_input = zoom(tumor_mask_input, zoom_factors, order=0)
                self.tumor_mask = tumor_mask_input.astype(bool)
            
            elif tumor_center is not None and tumor_size is not None:
                tumor_center = np.asarray(tumor_center)
                if isinstance(tumor_size, (int, float)):
                    self.tumor_mask = self._create_spherical_mask(tumor_center, tumor_size)
                else:
                    self.tumor_mask = self._create_ellipsoid_mask(tumor_center, tumor_size)
            else:
                raise ValueError("Must provide either tumor_mask_input or both tumor_center and tumor_size")
            
            # Update tumor masks for all phases in static case
            if not hasattr(self, 'tumor_masks_phases'):
                self.tumor_masks_phases = [self.tumor_mask] * self.num_phases
            else:
                # In 4D case, each phase gets a copy for now
                # This could be enhanced to use deformation if available
                self.tumor_masks_phases = [self.tumor_mask.copy() for _ in range(self.num_phases)]
            
            # Update ITV (union of tumor masks across phases)
            self.tumor_mask = np.any(self.tumor_masks_phases, axis=0)
            
            return True
            
        except Exception as e:
            print(f"Error in set_tumor_data: {e}")
            return False

    def _initialize_simplified_model(self, grid_size_param):
        self.grid_size = grid_size_param # Use the passed grid_size
        self.tumor_center = np.array(self.grid_size) / 2
        self.tumor_mask = self._create_spherical_mask(self.tumor_center, 10)
        self.oar_masks = {
            "lung": self._create_ellipsoid_mask(self.tumor_center + np.array([20, 0, 0]), (30, 20, 20)),
            "heart": self._create_spherical_mask(self.tumor_center + np.array([-20, 0, 0]), 15)
        }
        self.density_grid = np.ones(self.grid_size)
        self.tumor_masks_phases = [self.tumor_mask] * self.num_phases
        self.oar_masks_phases = [{oar: mask for oar, mask in self.oar_masks.items()}] * self.num_phases
        self.density_grids_phases = [self.density_grid] * self.num_phases
        self.affine_transforms = [np.eye(4)] * self.num_phases # Identity transforms
        self.accumulated_dose = np.zeros(self.grid_size) # Initialize accumulated dose
        self.voxel_volume = 1.0 * 1.0 * 1.0 * 1e-3 # 1mm^3 in cm^3 for simplified model
        print(f"Initialized with simplified model, grid_size: {self.grid_size}")


    def _load_ct_series(self, ct_dir_path):
        """Loads a single CT series from a directory, sorted by slice location."""
        dicom_files = [pydicom.dcmread(os.path.join(ct_dir_path, f))
                       for f in os.listdir(ct_dir_path) if f.endswith('.dcm')]
        # Filter for CT images and sort them
        ct_series = [ds for ds in dicom_files if hasattr(ds, 'ImagePositionPatient') and hasattr(ds, 'InstanceNumber')]
        # Sort by instance number or slice location. SliceLocation is more robust.
        if all(hasattr(s, 'SliceLocation') for s in ct_series):
            ct_series.sort(key=lambda s: s.SliceLocation)
        elif all(hasattr(s, 'InstanceNumber') for s in ct_series):
             ct_series.sort(key=lambda s: s.InstanceNumber)
        else:
            print("Warning: CT slices could not be reliably sorted.")

        if not ct_series:
            print(f"Warning: No CT images found in {ct_dir_path}")
            return None
        
        # Store voxel size from the series
        try:
            pixel_spacing = np.array(ct_series[0].PixelSpacing, dtype=float)
            slice_thickness = float(ct_series[0].SliceThickness)
            self.voxel_size_mm = np.array([pixel_spacing[1], pixel_spacing[0], slice_thickness]) # (x, y, z) in mm
            print(f"Derived voxel size (mm): {self.voxel_size_mm}")
        except Exception as e:
            print(f"Warning: Could not derive voxel size from CT series: {e}")
            self.voxel_size_mm = np.array([1.0, 1.0, 1.0]) # Default to 1mm isotropic


        return ct_series

    def _get_affine_transform_from_dicom(self, dicom_slice):
        """
        Computes the affine transformation matrix from image coordinates (pixels)
        to patient coordinates (mm) for a DICOM slice.
        Assumes standard DICOM attributes.
        Returns a 4x4 affine matrix.
        """
        # ImagePositionPatient: (x, y, z) of the top-left corner of the first pixel
        ipp = np.array(dicom_slice.ImagePositionPatient, dtype=float)
        # ImageOrientationPatient: (Rx, Ry, Rz, Cx, Cy, Cz)
        # Rx, Ry, Rz are components of the first row vector (direction of x-pixels)
        # Cx, Cy, Cz are components of the first col vector (direction of y-pixels)
        iop = np.array(dicom_slice.ImageOrientationPatient, dtype=float)
        row_vec = iop[:3]
        col_vec = iop[3:]
        # PixelSpacing: (row_spacing, col_spacing)
        ps = np.array(dicom_slice.PixelSpacing, dtype=float)

        # Create the 3x3 rotation and scaling part of the matrix
        # This maps (col_idx, row_idx) to a displacement in patient coords
        rotation_scaling = np.zeros((3, 3))
        rotation_scaling[:, 0] = row_vec * ps[1] # Column index corresponds to change along row vector
        rotation_scaling[:, 1] = col_vec * ps[0] # Row index corresponds to change along col vector
        # For 3D volume, need slice direction vector
        # Approximated by cross product if not directly available or constant slice thickness
        # For this example, we'll handle slice direction later if forming a volume transform

        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = rotation_scaling
        affine_matrix[:3, 3] = ipp # Translation part

        return affine_matrix


    def _load_4d_ct_data(self, fourd_ct_path, rt_struct_path):
        phase_dirs = sorted([d for d in os.listdir(fourd_ct_path) if os.path.isdir(os.path.join(fourd_ct_path, d))])
        if not phase_dirs:
            raise FileNotFoundError(f"No phase directories found in {fourd_ct_path}")

        all_density_grids = []
        all_tumor_masks = []
        all_oar_masks_phases = [] # List of dicts, one dict per phase
        all_affine_transforms = []

        # Load reference CT series (e.g., phase_0) and RT Struct
        # The RT Struct contours are defined relative to one of the CT series.
        ref_phase_path = os.path.join(fourd_ct_path, self.reference_phase_name)
        if not os.path.isdir(ref_phase_path):
            ref_phase_path = os.path.join(fourd_ct_path, phase_dirs[0]) # Fallback to first phase dir
            print(f"Warning: Reference phase '{self.reference_phase_name}' not found. Using '{phase_dirs[0]}'.")

        print(f"Loading reference CT from: {ref_phase_path}")
        ref_ct_series = self._load_ct_series(ref_phase_path)
        if not ref_ct_series:
            raise ValueError(f"Could not load reference CT series from {ref_phase_path}")

        # Derive grid size from reference CT
        # Assuming (slices, rows, cols) for storage then re-orient to (x,y,z) as needed
        # For now, let's make grid_size (cols, rows, slices) to align with typical access patterns
        # This needs careful thought based on how DICOM arrays are stacked and used.
        # A common convention: X = columns, Y = rows, Z = slices
        rows = ref_ct_series[0].Rows
        cols = ref_ct_series[0].Columns
        num_slices_ref = len(ref_ct_series)
        self.grid_size = (cols, rows, num_slices_ref) # (X, Y, Z)
        print(f"Derived grid_size from reference CT: {self.grid_size}")


        tumor_mask_ref, oar_masks_ref, roi_names_ref = self._load_rt_struct(rt_struct_path, ref_ct_series)
        # Get affine for the reference volume
        # This requires constructing a volume affine from slice affines, considering slice thickness
        # For simplicity now, let's assume a global affine for the ref volume can be derived
        # or we use slice-by-slice logic for contour_to_mask.
        # Let's assume first slice's affine represents the start for now.
        ref_volume_affine = self._get_affine_transform_from_dicom(ref_ct_series[0]) # Simplified


        for phase_idx, phase_dir_name in enumerate(phase_dirs):
            phase_ct_path = os.path.join(fourd_ct_path, phase_dir_name)
            print(f"  Processing phase {phase_idx + 1}/{len(phase_dirs)}: {phase_dir_name}")
            current_ct_series = self._load_ct_series(phase_ct_path)
            if not current_ct_series:
                print(f"  Warning: Could not load CT for phase {phase_dir_name}. Skipping.")
                continue
            
            # Ensure grid consistency (crude check, real DIR handles this)
            if current_ct_series[0].Rows != rows or current_ct_series[0].Columns != cols or len(current_ct_series) != num_slices_ref:
                print(f"  Warning: Phase {phase_dir_name} has different dimensions. Skipping. Real DIR needed.")
                # In a real system, you would resample or use DIR that handles different grids.
                continue

            # Stack pixels and convert HU to density for the current phase
            ct_pixel_data_phase = np.stack([s.pixel_array * s.RescaleSlope + s.RescaleIntercept for s in current_ct_series], axis=-1)
            # Transpose to match self.grid_size (cols, rows, slices) if necessary
            ct_pixel_data_phase = ct_pixel_data_phase.transpose(1,0,2) # if original stack is (rows, cols, slices)
            
            density_grid_phase = self._hu_to_density(ct_pixel_data_phase)
            all_density_grids.append(density_grid_phase)

            # Get affine for current phase volume
            current_volume_affine = self._get_affine_transform_from_dicom(current_ct_series[0]) # Simplified
            all_affine_transforms.append(current_volume_affine)


            # --- Deformable Image Registration (DIR) Step ---
            # If current phase IS the reference phase, use original masks
            if phase_dir_name == self.reference_phase_name or phase_ct_path == ref_phase_path:
                print(f"    Phase {phase_dir_name} is the reference phase. Using original masks.")
                deformed_tumor_mask = tumor_mask_ref.copy()
                deformed_oar_masks = {name: mask.copy() for name, mask in oar_masks_ref.items()}
            else:
                print(f"    Deforming structures for phase {phase_dir_name}...")
                # Placeholder for actual DIR.
                # Inputs to DIR would be:
                #   - reference_ct_image (derived from ref_ct_series)
                #   - current_phase_ct_image (derived from current_ct_series)
                #   - tumor_mask_ref, oar_masks_ref
                #   - Potentially ref_volume_affine, current_volume_affine
                # Output would be:
                #   - deformed_tumor_mask, deformed_oar_masks for the current phase
                if self.dir_method == 'simplified_sinusoidal':
                    # Use the simplified sinusoidal shift as a fallback/example
                    # This needs the original _deform_structures logic adapted
                    # For now, let's just show where it would go.
                    # displacement = 5 * np.sin(2 * np.pi * phase_idx / len(phase_dirs))
                    # deformed_tumor_mask, deformed_oar_masks = self._apply_simplified_deformation(
                    #     tumor_mask_ref, oar_masks_ref, displacement_z=displacement
                    # )
                    print("      DIR Method: Simplified Sinusoidal (Example - requires _apply_simplified_deformation)")
                    # For now, let's just copy reference if not implementing full sinusoidal here yet
                    deformed_tumor_mask = tumor_mask_ref.copy() # Replace with actual deformation
                    deformed_oar_masks = {name: mask.copy() for name, mask in oar_masks_ref.items()} # Replace

                elif self.dir_method == 'external_sitk': # Example using SimpleITK
                    print("      DIR Method: External SimpleITK (Placeholder - NOT IMPLEMENTED)")
                    # try:
                    #   import SimpleITK as sitk
                    #   # 1. Convert numpy arrays (ref_ct, current_ct, ref_masks) to SITK images
                    #   # 2. Set up SITK registration method (e.g., Demons, B-spline)
                    #   # 3. Execute registration to get a transform
                    #   # 4. Apply transform to reference masks to get deformed masks
                    #   # This is a complex step omitted for brevity.
                    #   deformed_tumor_mask = tumor_mask_ref.copy() # Placeholder
                    #   deformed_oar_masks = {name: mask.copy() for name, mask in oar_masks_ref.items()} # Placeholder
                    # except ImportError:
                    #   print("      SimpleITK not installed. Cannot perform advanced DIR.")
                    deformed_tumor_mask = tumor_mask_ref.copy()
                    deformed_oar_masks = {name: mask.copy() for name, mask in oar_masks_ref.items()}
                else: # No deformation or other methods
                    print(f"      DIR Method: {self.dir_method} - using reference masks as is.")
                    deformed_tumor_mask = tumor_mask_ref.copy()
                    deformed_oar_masks = {name: mask.copy() for name, mask in oar_masks_ref.items()}

            all_tumor_masks.append(deformed_tumor_mask)
            all_oar_masks_phases.append(deformed_oar_masks)

        if not all_density_grids:
            raise ValueError("Failed to load any density grids from 4D CT.")

        return all_density_grids, all_tumor_masks, all_oar_masks_phases, all_affine_transforms

    def _apply_simplified_deformation(self, tumor_mask_ref, oar_masks_ref, displacement_vector):
        """Applies a rigid displacement to masks using map_coordinates."""
        # displacement_vector should be (dx, dy, dz) in voxel units
        # For the original example, it was just dz.
        # This function needs to be robust to the shape of masks and displacement.
        
        coords = np.indices(self.grid_size, dtype=float) # Use float for sub-pixel shifts
        
        # Assuming displacement_vector is (dx, dy, dz)
        # We want to find original coords: new_coord - displacement = old_coord
        # So, for map_coordinates, the input coords should be original_grid_coords - displacement
        shifted_coords = [coords[i] - displacement_vector[i] for i in range(len(displacement_vector))]

        deformed_tumor_mask = map_coordinates(tumor_mask_ref.astype(float), shifted_coords, order=0, mode='constant', cval=0.0).astype(bool)
        
        deformed_oar_masks = {}
        for oar_name, oar_mask_ref in oar_masks_ref.items():
            deformed_oar_masks[oar_name] = map_coordinates(oar_mask_ref.astype(float), shifted_coords, order=0, mode='constant', cval=0.0).astype(bool)
            
        return deformed_tumor_mask, deformed_oar_masks


    def _load_rt_struct(self, rt_struct_path, ref_ct_series):
        """
        Loads RT Structure Set and converts contours to masks.
        Aligns masks with the grid of ref_ct_series.
        """
        rt_struct = pydicom.dcmread(rt_struct_path)
        
        # Get mapping from SOPInstanceUID to slice index in ref_ct_series
        sop_uid_to_index = {s.SOPInstanceUID: i for i, s in enumerate(ref_ct_series)}
        
        # Image Position Patient for each slice, and pixel spacing, orientation
        # These are needed to map contour data (in patient coordinates) to voxel indices
        # For simplicity, assuming all slices in ref_ct_series have same orientation and pixel spacing
        ref_slice_origin = np.array(ref_ct_series[0].ImagePositionPatient)
        pixel_spacing = np.array(ref_ct_series[0].PixelSpacing, dtype=float) # (row_spacing, col_spacing)
        # Calculate slice thickness robustly
        slice_locations = sorted([s.ImagePositionPatient[2] for s in ref_ct_series])
        slice_thickness = np.mean(np.diff(slice_locations)) if len(slice_locations) > 1 else pixel_spacing[0] # Approx if only one slice
        self.voxel_size_mm = np.array([pixel_spacing[1], pixel_spacing[0], slice_thickness]) # (x, y, z) in mm
        self.voxel_volume = np.prod(self.voxel_size_mm) * 1e-3 # mm^3 to cm^3
        print(f"Derived voxel size (mm): {self.voxel_size_mm}")
        print(f"Derived voxel volume (cm^3): {self.voxel_volume:.6f}")


        # Image Orientation Patient
        iop = ref_ct_series[0].ImageOrientationPatient
        row_dir = np.array(iop[0:3]) # Direction of first row
        col_dir = np.array(iop[3:6]) # Direction of first column
        slice_dir = np.cross(row_dir, col_dir) # Normal to the plane

        # Create world_to_voxel affine matrix for the reference CT volume
        # This is a simplified affine based on the first slice and assuming axial slices
        # A full solution would build this from individual slice affines or DICOM Frame of Reference UID
        T_patient_to_grid = np.eye(4)
        # Note: DICOM X is typically columns, Y is rows, Z is slice location.
        # Array indexing is typically (slice, row, col) or (z, y, x).
        # We are using grid_size (cols, rows, slices) -> (x, y, z)
        # The matrix should map patient (x,y,z) to grid (col, row, slice)
        # Patient X corresponds to col_vec * ps[1]
        # Patient Y corresponds to row_vec * ps[0]
        # Patient Z corresponds to slice_dir * slice_thickness
        # So the matrix columns should be (col_vec*ps[1], row_vec*ps[0], slice_dir*slice_thickness)
        # And the translation is -origin
        R_patient_to_grid = np.column_stack((col_dir * pixel_spacing[1], row_dir * pixel_spacing[0], slice_dir * slice_thickness))
        try:
            T_patient_to_grid[:3,:3] = np.linalg.inv(R_patient_to_grid)
        except np.linalg.LinAlgError:
            print("Warning: Could not invert rotation matrix for RTStruct loading. Using identity.")
            T_patient_to_grid[:3,:3] = np.eye(3) # Fallback
            
        T_patient_to_grid[:3,3] = -T_patient_to_grid[:3,:3] @ ref_slice_origin


        # Initialize masks based on the grid_size derived from reference CT
        overall_tumor_mask = np.zeros(self.grid_size, dtype=bool)
        overall_oar_masks = {}
        roi_names = {} # To store actual ROI names

        for i, roi_contour in enumerate(rt_struct.ROIContourSequence):
            roi_number = roi_contour.ReferencedROINumber
            # Find matching StructureSetROISequence item
            structure_set_roi = next((s for s in rt_struct.StructureSetROISequence if s.ROINumber == roi_number), None)
            if not structure_set_roi: continue
            
            roi_name_orig = structure_set_roi.ROIName
            roi_name_lower = roi_name_orig.lower()
            print(f"    Processing ROI: {roi_name_orig}")
            roi_names[roi_number] = roi_name_orig

            current_roi_mask = np.zeros(self.grid_size, dtype=bool)

            if hasattr(roi_contour, 'ContourSequence'):
                for contour_sequence in roi_contour.ContourSequence:
                    contour_data = np.array(contour_sequence.ContourData).reshape(-1, 3) # (x,y,z) in patient coords
                    
                    # Determine which slice this contour belongs to
                    # Use the SOPInstanceUID (best)
                    slice_idx = None
                    if hasattr(contour_sequence, 'ContourImageSequence') and contour_sequence.ContourImageSequence:
                         if contour_sequence.ContourImageSequence[0].ReferencedSOPInstanceUID in sop_uid_to_index:
                             slice_idx = sop_uid_to_index[contour_sequence.ContourImageSequence[0].ReferencedSOPInstanceUID]
                         else: # Fallback: match by z-coordinate (less robust)
                             contour_z = contour_data[0, 2]
                             slice_z_diffs = [abs(s.ImagePositionPatient[2] - contour_z) for s in ref_ct_series]
                             if slice_z_diffs:
                                 slice_idx = np.argmin(slice_z_diffs)
                    
                    if slice_idx is None:
                        print(f"      Warning: Could not map contour for ROI {roi_name_orig} to a CT slice. Skipping contour.")
                        continue

                    # Convert patient coordinates to voxel indices for this slice
                    contour_data_homog = np.hstack((contour_data, np.ones((contour_data.shape[0], 1))))
                    # Apply the inverse affine transform
                    contour_voxels_float = (T_patient_to_grid @ contour_data_homog.T).T[:, :3]
                    
                    # For polygon filling, we need 2D points on the specific slice
                    # Points should be (col, row) or (x,y) indices for that slice_idx
                    # contour_voxels_float[:,0] are x-indices (cols), contour_voxels_float[:,1] are y-indices (rows)
                    
                    from skimage.draw import polygon
                    # Ensure contour_voxels_float are within bounds for skimage.draw.polygon
                    # Clip to grid boundaries [0, dim_size-1]
                    # Note: skimage.draw.polygon expects (rows, cols) i.e., (y, x)
                    rr, cc = polygon(
                        np.clip(contour_voxels_float[:, 1], 0, self.grid_size[1] - 1),  # row indices (y)
                        np.clip(contour_voxels_float[:, 0], 0, self.grid_size[0] - 1)   # col indices (x)
                    )
                    
                    # Ensure rr, cc are within valid mask dimensions for the slice
                    valid_indices = (rr >= 0) & (rr < self.grid_size[1]) & \
                                    (cc >= 0) & (cc < self.grid_size[0])
                    
                    # Apply the mask to the correct slice in the 3D volume
                    if 0 <= slice_idx < self.grid_size[2]:
                         current_roi_mask[cc[valid_indices], rr[valid_indices], slice_idx] = True
                    else:
                         print(f"      Warning: Contour for ROI {roi_name_orig} maps to out-of-bounds slice index {slice_idx}. Skipping contour.")


            if "tumor" in roi_name_lower or "gtv" in roi_name_lower or "ctv" in roi_name_lower:
                overall_tumor_mask |= current_roi_mask
            elif "ptv" in roi_name_lower: # Often PTV is the target
                # Decide if PTV is target or OAR. Usually target.
                overall_tumor_mask |= current_roi_mask
            else: # Assume OAR
                if roi_name_orig not in overall_oar_masks:
                    overall_oar_masks[roi_name_orig] = np.zeros(self.grid_size, dtype=bool)
                overall_oar_masks[roi_name_orig] |= current_roi_mask
        
        return overall_tumor_mask, overall_oar_masks, roi_names


    def _create_spherical_mask(self, center, radius):
        """Create a spherical boolean mask centered at given coordinates with given radius."""
        x, y, z = np.ogrid[:self.grid_size[0], :self.grid_size[1], :self.grid_size[2]]
        center = np.asarray(center)
        dist_squared = ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
        return dist_squared <= radius**2
    
    def _create_ellipsoid_mask(self, center, radii):
        """Create an ellipsoidal boolean mask with given center and radii."""
        x, y, z = np.ogrid[:self.grid_size[0], :self.grid_size[1], :self.grid_size[2]]
        center = np.asarray(center)
        radii = np.asarray(radii)
        dist_squared = ((x - center[0])**2 / radii[0]**2 + 
                       (y - center[1])**2 / radii[1]**2 + 
                       (z - center[2])**2 / radii[2]**2)
        return dist_squared <= 1.0

    def _hu_to_density(self, hu_data):
        # Simple linear approximation for density from HU
        # Water = 0 HU -> density 1.0
        # Air = -1000 HU -> density 0.0
        # Bone = 1000 HU -> density ~1.8 (example)
        # Density = (HU + 1000) / 1000 * density_at_1000HU
        # A more common approach:
        # density = 1.0 + HU / 1000.0 # Simple linear scaling
        
        # Using piece-wise linear approximation based on common values
        density = np.ones_like(hu_data, dtype=float)
        density[hu_data <= -500] = 0.2 # Lung-like
        density[(hu_data > -500) & (hu_data <= 50)] = 1.0 # Soft tissue-like
        density[hu_data > 50] = 1.2 # Bone-like (simplified)
        
        return density


    def _generate_beam_directions(self, num_beams):
        # Generate directions on a sphere (e.g., using golden spiral)
        # For simplicity, let's keep it coplanar in the xy plane for now
        angles = np.linspace(0, 2 * np.pi, num_beams, endpoint=False)
        directions = [(np.cos(a), np.sin(a), 0) for a in angles] # Assume coplanar for now
        return directions

    def _get_source_position(self, direction):
        # Place source far enough outside the grid
        # Center of grid:
        center = np.array(self.grid_size) / 2.0
        # Max dimension of grid to ensure source is outside
        max_dim = np.max(self.grid_size)
        # Move source along negative direction from center by a large distance (e.g., 2 * max_dim)
        source_pos = center - np.array(direction) * max_dim * 1.5
        return source_pos


    def _calculate_dose_for_specific_phase(self, beam_weights, target_phase_index):
        """
        Calculates dose for a given beam_weights configuration AT a specific respiratory phase.
        This is used to build the dose influence matrix for optimization.
        """
        phase_dose_at_target = np.zeros(self.grid_size)
        density_grid_target_phase = self.density_grids_phases[target_phase_index]
        # tumor_mask_target_phase = self.tumor_masks_phases[target_phase_index] # Not needed for dose calculation itself

        for i, (direction, weight) in enumerate(zip(self.beam_directions, beam_weights)):
            if weight == 0:
                continue
            source = self._get_source_position(direction)
            # Calculate fluence using Numba function
            fluence = calculate_primary_fluence_numba(source, np.array(direction), density_grid_target_phase, self.grid_size)
            
            # Convolve fluence with dose kernel
            partial_dose = convolve(fluence, self.dose_kernel, mode="constant")
            
            # Simple heterogeneity correction (scaling by local density / mean density)
            mean_density_this_phase = np.mean(density_grid_target_phase[density_grid_target_phase > 0])
            if mean_density_this_phase > 1e-9: # Avoid division by zero
                partial_dose *= (density_grid_target_phase / mean_density_this_phase)
            
            phase_dose_at_target += weight * partial_dose
        
        # Note: This function calculates the dose *per unit weight*.
        # The scaling based on tumor volume/max dose is applied *later* in calculate_dose
        # or implicitly handled by the optimization objective.
        # For building the influence matrix, we need the raw dose per unit weight.
        # The original code scaled here, which might be incorrect for influence matrix.
        # Let's remove the scaling here and rely on the optimization objective to handle dose levels.
        
        return phase_dose_at_target


    def calculate_dose(self, beam_weights):
        """
        Calculates the total dose distribution averaged over respiratory phases.
        This is the dose delivered in *one fraction* for the given beam weights.
        """
        dose = np.zeros(self.grid_size)
        if not self.density_grids_phases: # Should not happen if initialized
            print("Error: No density grids available for dose calculation.")
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
                if mean_density_phase > 1e-9: # Avoid division by zero
                     partial_dose *= (density_grid / mean_density_phase)
                phase_dose += weight * partial_dose
            
            # Average dose over phases
            dose += self.respiratory_phase_weights[phase] * phase_dose
        
        # Adaptive scaling using the overall ITV (union of tumor masks)
        # This scales the *total fractional dose* to achieve a target dose level in the ITV.
        tumor_volume = np.sum(self.tumor_mask)
        # Target dose per fraction in the ITV (e.g., based on volume)
        # This logic seems to scale the *fractional* dose based on ITV volume, which is unusual.
        # Typically, the total prescription dose is set, and optimization aims to deliver that.
        # Let's keep the scaling logic from the original code for consistency, but note it's a simplification.
        base_dose_per_fraction = 4.0 # Example target dose per fraction for small tumors
        if tumor_volume > 0: # Avoid issues with empty tumor mask
            if tumor_volume <= 5: base_dose_per_fraction = 4.0
            elif tumor_volume >= 30: base_dose_per_fraction = 2.0
            else: base_dose_per_fraction = 2.0 + 2.0 * (30.0 - tumor_volume) / 25.0
        
        max_dose_in_itv = 0
        if np.any(self.tumor_mask) and np.any(dose[self.tumor_mask]):
             max_dose_in_itv = np.max(dose[self.tumor_mask])

        if max_dose_in_itv > 1e-6:
            dose *= base_dose_per_fraction / max_dose_in_itv
        elif np.max(dose) > 1e-6: # If no dose in ITV, scale by overall max
            dose *= base_dose_per_fraction / np.max(dose)
        # If max dose is still zero, dose remains zero.

        return dose

    def optimize_beams(self):
        """
        Optimizes beam weights using classical optimization (SciPy minimize).
        Replaces the Qiskit QAOA step.
        """
        print("  [optimize_beams] Starting classical optimization...")
        num_beams = len(self.beam_directions)
        
        # Calculate dose influence matrix for each phase and each beam
        # D_phase_beam[p, b, x, y, z] = dose at voxel (x,y,z) from beam b in phase p with weight 1
        dose_influence_phases = np.zeros((self.num_phases, num_beams) + self.grid_size)
        
        print(f"  [optimize_beams] Calculating dose influence matrix for {self.num_phases} phases and {num_beams} beams...")
        for phase_opt_idx in range(self.num_phases):
            # print(f"    [optimize_beams] Processing phase {phase_opt_idx + 1}/{self.num_phases} for influence matrix...")
            density_grid_phase = self.density_grids_phases[phase_opt_idx]

            for i_beam in range(num_beams):
                # Calculate dose from beam i_beam with weight 1 in phase phase_opt_idx
                # This is similar to _calculate_dose_for_specific_phase but we store the raw dose map
                temp_beam_weights = [1.0 if j == i_beam else 0.0 for j in range(num_beams)]
                
                phase_dose_map = np.zeros(self.grid_size)
                source = self._get_source_position(self.beam_directions[i_beam])
                fluence = calculate_primary_fluence_numba(source, np.array(self.beam_directions[i_beam]), density_grid_phase, self.grid_size)
                partial_dose = convolve(fluence, self.dose_kernel, mode="constant")
                
                mean_density_phase = np.mean(density_grid_phase[density_grid_phase > 0])
                if mean_density_phase > 1e-9:
                     partial_dose *= (density_grid_phase / mean_density_phase)
                
                dose_influence_phases[phase_opt_idx, i_beam, :, :, :] = partial_dose

        print("  [optimize_beams] Dose influence matrix calculated.")

        # Define the objective function for classical optimization
        # We want to maximize tumor dose and minimize OAR dose, averaged over phases.
        # Objective = - (Weighted Tumor Dose) + (Weighted OAR Dose) + Penalties
        
        def objective_function(weights):
            # Ensure weights are non-negative
            weights = np.maximum(0, weights)

            # Calculate the total dose distribution for these weights, averaged over phases
            # D_total(w) = sum_b w_b * (sum_p w_p * D_phase_beam[p, b])
            # This is incorrect. The dose calculation averages over phases *after* summing beams.
            # D_total(w) = sum_p w_p * (sum_b w_b * D_phase_beam[p, b])
            
            total_dose_averaged_phases = np.zeros(self.grid_size)
            for phase_idx in range(self.num_phases):
                 # Dose in this phase = sum_b w_b * D_phase_beam[phase_idx, b]
                 dose_this_phase = np.tensordot(weights, dose_influence_phases[phase_idx], axes=([0], [0]))
                 total_dose_averaged_phases += self.respiratory_phase_weights[phase_idx] * dose_this_phase

            # --- Calculate Objective Terms ---
            tumor_term = 0
            if np.any(self.tumor_mask):
                 # Maximize mean dose in the overall ITV
                 mean_tumor_dose = np.mean(total_dose_averaged_phases[self.tumor_mask])
                 tumor_term = -1.0 * mean_tumor_dose # Minimize negative mean dose

            oar_term = 0
            for oar_name, oar_mask in self.oar_masks.items():
                 if np.any(oar_mask):
                      # Minimize mean dose in each overall OAR
                      mean_oar_dose = np.mean(total_dose_averaged_phases[oar_mask])
                      # Assign a weight to each OAR term. Example: 0.5 for all OARs
                      oar_term += 0.5 * mean_oar_dose # Minimize OAR dose

            # Penalty for opposing beams (based on weights)
            opposing_penalty = 0
            for i in range(num_beams):
                for j in range(i + 1, num_beams):
                    angle_rad = np.arccos(np.clip(np.dot(self.beam_directions[i], self.beam_directions[j]), -1.0, 1.0))
                    angle_deg = np.degrees(angle_rad)
                    if angle_deg > 160:
                        # Penalize if both beams have significant weight
                        opposing_penalty += 2.0 * weights[i] * weights[j] # Quadratic penalty

            # Penalty for total number of beams (encourage sparsity or target K beams)
            # Simple penalty for using too many beams, or deviation from target K
            K_target_beams = 3
            total_weight = np.sum(weights)
            # Penalty = alpha * (total_weight - K_target)^2 or similar
            # Let's use a penalty for deviating from a target sum of weights (representing K beams)
            # This is tricky with continuous weights. A simpler approach is a penalty on the L1 or L2 norm of weights.
            # Or, penalize if sum of weights is too low (less than K_target * min_weight_per_beam)
            # Let's use a simple L2 penalty on weights to encourage smaller values
            # weight_magnitude_penalty = 0.1 * np.sum(weights**2) # Example

            # Combine terms
            cost = tumor_term + oar_term + opposing_penalty # + weight_magnitude_penalty

            return cost

        # --- Classical Optimization ---
        print("  [optimize_beams] Running classical optimization (SLSQP)...")
        # Initial guess: equal weights for all beams
        initial_weights = np.ones(num_beams) / num_beams
        
        # Bounds: weights must be between 0 and 1 (or just non-negative if scaling is done later)
        # Let's optimize non-negative weights, and normalize the final result.
        bounds = [(0, None)] * num_beams # Non-negative weights

        # Constraints: e.g., sum of weights = 1 (if optimizing intensity)
        # If optimizing binary selection, this is not needed here.
        # Since we are replacing QAOA (binary), let's aim for a sparse solution
        # and threshold the result. No sum constraint needed for thresholding.

        # Use SLSQP method which handles bounds and constraints (though we have no equality/inequality constraints here)
        # Other methods like L-BFGS-B or TNC could also work with bounds.
        # Let's use L-BFGS-B as it's generally robust for bounded problems.
        result = minimize(
            objective_function,
            initial_weights,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'disp': False} # Increased maxiter
        )

        # Extract optimized continuous weights
        optimized_weights = result.x
        print(f"  [optimize_beams] Classical optimization finished. Success: {result.success}, Message: {result.message}")
        # print(f"  [optimize_beams] Optimized continuous weights: {optimized_weights}")

        # Convert continuous weights to binary selection (0 or 1)
        # Simple thresholding
        threshold = 0.1 # Example threshold
        best_weights_binary = (optimized_weights > threshold).astype(float) # Use float for weights

        # Fallback if no beams are selected
        if np.sum(best_weights_binary) == 0:
            print("Warning: Classical optimization resulted in all zero weights after thresholding. Using default (first 3 beams).")
            num_to_activate = min(3, num_beams)
            best_weights_binary = np.zeros(num_beams, dtype=float)
            if num_beams > 0: best_weights_binary[:num_to_activate] = 1.0
        # Optional: Ensure a minimum number of beams are selected if desired
        # while np.sum(best_weights_binary) < K_target_beams and np.sum(optimized_weights) > 1e-9:
        #     # Activate the beam with the highest continuous weight among inactive ones
        #     inactive_indices = np.where(best_weights_binary == 0)[0]
        #     if len(inactive_indices) == 0: break # All beams already active
        #     highest_weight_inactive_idx = inactive_indices[np.argmax(optimized_weights[inactive_indices])]
        #     best_weights_binary[highest_weight_inactive_idx] = 1.0
        #     print(f"  [optimize_beams] Activated beam {highest_weight_inactive_idx} to meet minimum count.")


        print(f"  [optimize_beams] Final binary beam selection: {best_weights_binary}")
        print("  [optimize_beams] Finished.")
        return best_weights_binary


    def simulate_fractionated_treatment(self, num_fractions=5):
        history = {"tumor_volumes": [], "tcp": [], "ntcp": {oar: [] for oar in self.oar_masks}} # Initialize NTCP history correctly

        # Initialize accumulated dose for the entire treatment
        self.accumulated_dose = np.zeros(self.grid_size)

        for fraction in range(num_fractions):
            print(f"Processing fraction {fraction + 1}/{num_fractions}")
            
            # Optimize beams for the current tumor geometry
            beam_weights = self.optimize_beams()
            
            # Calculate the dose delivered in this fraction (averaged over phases)
            fraction_dose = self.calculate_dose(beam_weights) 

            # Add fractional dose to the total accumulated dose
            self.accumulated_dose += fraction_dose

            # --- Tumor Response Modeling ---
            # Update tumor mask based on the *fractional* dose delivered in this step
            # This simulates cell kill per fraction, leading to geometry changes
            alpha_tumor = self.radiobiological_params["tumor"]["alpha"]
            beta_tumor = self.radiobiological_params["tumor"]["beta"]
            
            # Apply biological effect from the *fractional dose* to the tumor mask of each phase
            # This is a simplified model of response per fraction
            for phase in range(self.num_phases):
                current_phase_tumor_mask = self.tumor_masks_phases[phase]
                if not np.any(current_phase_tumor_mask): continue # If tumor already gone in this phase

                # Calculate survival probability per voxel for THIS FRACTION's dose
                survival_prob_map_fraction = np.exp(-(alpha_tumor * fraction_dose + beta_tumor * fraction_dose**2))
                
                # Update surviving voxels ONLY where the tumor existed in this phase
                # Use probabilistic cell kill
                random_survival = np.random.rand(*fraction_dose.shape)
                surviving_voxels_this_phase = (random_survival < survival_prob_map_fraction) & current_phase_tumor_mask
                
                # Ensure at least one voxel remains if the tumor existed in this phase (optional, prevents premature eradication)
                # if np.any(current_phase_tumor_mask) and not np.any(surviving_voxels_this_phase):
                #     # Find the voxel with the minimum fractional dose within the original mask
                #     original_tumor_indices = np.argwhere(current_phase_tumor_mask)
                #     if original_tumor_indices.size > 0:
                #         doses_in_original_mask = fraction_dose[current_phase_tumor_mask]
                #         min_dose_local_idx = np.argmin(doses_in_original_mask)
                #         global_min_dose_coord = tuple(original_tumor_indices[min_dose_local_idx])
                        
                #         surviving_voxels_this_phase = np.zeros_like(current_phase_tumor_mask, dtype=bool)
                #         surviving_voxels_this_phase[global_min_dose_coord] = True
                #     else:
                #         # Should not happen if np.any(current_phase_tumor_mask) is true, but safety
                #         pass


                self.tumor_masks_phases[phase] = surviving_voxels_this_phase
            
            # Update the overall ITV mask based on the new phase masks
            self.tumor_mask = np.any(self.tumor_masks_phases, axis=0) 

            # --- Radiobiological Endpoint Calculation (using ACCUMULATED dose) ---
            # TCP and NTCP are calculated based on the *total accumulated dose* up to this fraction
            tcp_val = self._calculate_tcp(self.accumulated_dose) 
            
            history["tumor_volumes"].append(np.sum(self.tumor_mask)) # Store current ITV volume
            history["tcp"].append(tcp_val)

            for oar_name in self.oar_masks.keys(): # Iterate over OARs defined in the overall self.oar_masks
                ntcp_val = self._calculate_ntcp(self.accumulated_dose, oar_name)
                if oar_name not in history["ntcp"]: history["ntcp"][oar_name] = [] # Ensure list exists
                history["ntcp"][oar_name].append(ntcp_val)
            
            print(f"Fraction {fraction + 1}: ITV Volume = {np.sum(self.tumor_mask)}, TCP = {tcp_val:.4f}")
            if not self.tumor_mask.any():
                print("Tumor eradicated!")
                # Optionally break if tumor is gone, or continue to assess OAR dose
                # break # Uncomment to stop simulation if tumor is gone

        return history

    def _calculate_tcp(self, total_accumulated_dose):
        """
        Calculate Tumor Control Probability (TCP) using the LQ-Poisson model
        based on the total accumulated dose.
        """
        # TCP is calculated based on the overall ITV mask
        if not np.any(self.tumor_mask): return 1.0 # Tumor eradicated

        alpha_tumor = self.radiobiological_params["tumor"]["alpha"]
        beta_tumor = self.radiobiological_params["tumor"]["beta"]
        N0_density = self.radiobiological_params["tumor"]["N0_density"] # Initial clonogenic cell density

        # Calculate BED (Biologically Effective Dose) for the total accumulated dose
        # Assuming total dose D is delivered in N fractions of d=D/N.
        # BED = N * d * (1 + d / (alpha/beta)) = D * (1 + d / (alpha/beta))
        # If dose_per_fraction is constant, d = total_accumulated_dose / num_fractions_delivered
        # However, dose per fraction might vary in adaptive planning.
        # A more robust way is to sum BED per fraction, but that requires tracking fractional doses.
        # Simplification: Use total dose in LQ model directly, ignoring fractionation effect on TCP for simplicity here.
        # Or, assume a nominal dose per fraction for BED calculation if total fractions is known.
        # Let's use the total dose directly in the LQ model for simplicity as in some examples.
        # SF_total = exp(-(alpha*D_total + beta*D_total^2))
        
        # Calculate survival fraction per voxel based on total accumulated dose
        sf_map_total = np.exp(-(alpha_tumor * total_accumulated_dose + beta_tumor * total_accumulated_dose**2))
        
        # Consider only voxels within the current ITV
        # Total expected surviving cells = sum_voxels (N0_density * Voxel_Volume * SF_voxel)
        surviving_cells_per_voxel_in_itv = (N0_density * self.voxel_volume *
                                            sf_map_total[self.tumor_mask])
        
        total_surviving_cells = np.sum(surviving_cells_per_voxel_in_itv)
        
        # TCP = exp(- Total Expected Surviving Cells)
        tcp = np.exp(-total_surviving_cells)
        
        return tcp * 100.0 # Return as percentage


    def _calculate_ntcp(self, total_accumulated_dose, oar_name):
        """
        Calculate Normal Tissue Complication Probability (NTCP) using the LKB model
        based on the total accumulated dose.
        """
        if oar_name not in self.radiobiological_params or oar_name not in self.oar_masks:
            # print(f"Warning: OAR {oar_name} not found in params or masks for NTCP calculation.")
            return 0.0
        # NTCP is calculated based on the overall OAR mask (union over phases)
        if not np.any(self.oar_masks[oar_name]): return 0.0 # OAR is empty

        params = self.radiobiological_params[oar_name]
        n, m, TD50 = params["n"], params["m"], params["TD50"]
        alpha_beta_oar = params["alpha_beta"]
        
        current_oar_mask = self.oar_masks[oar_name] # Use the overall OAR mask (union over phases)
        
        # Get total accumulated dose values within the OAR mask
        total_dose_oar_voxels = total_accumulated_dose[current_oar_mask]
        
        if not total_dose_oar_voxels.size: return 0.0 # No OAR voxels with dose

        # Calculate EQD2 for OAR voxels based on total dose
        # Assuming total dose D is delivered in N fractions of d=D/N.
        # EQD2 = D * ( (D/N) + alpha_beta ) / ( 2 + alpha_beta )
        # This requires knowing N. Let's assume a nominal fraction size d_ref=2Gy for EQD2 conversion.
        d_ref = 2.0 # Reference dose per fraction for EQD2 conversion
        eqd2_oar_voxels = total_dose_oar_voxels * \
                          (total_dose_oar_voxels + alpha_beta_oar) / (d_ref + alpha_beta_oar)
        
        # gEUD calculation (Generalized Equivalent Uniform Dose)
        # gEUD = ( sum_i v_i * D_i^(1/n) )^n / (sum v_i)^(1/n)  where v_i is voxel volume
        # Assuming uniform voxel volume, this simplifies to:
        # gEUD = ( mean(D_i^(1/n)) )^n
        
        # Handle potential division by zero or invalid values for n
        if abs(n) < 1e-9: # If n is close to 0, gEUD approaches max dose for n>0, min dose for n<0
             # This is a simplification; the limit is more complex. Mean dose is often used as a fallback for n=1.
             # Let's use mean dose as a fallback for n=0 or very small n.
             gEUD = np.mean(eqd2_oar_voxels)
        else:
            try:
                # Ensure argument to power is non-negative if 1/n is not an integer
                # For typical LKB (n>0), doses are non-negative, so eqd2 is non-negative.
                gEUD = np.mean(eqd2_oar_voxels**(1/n))**n
            except Exception: # Catch potential issues like negative doses or complex results
                 print(f"Warning: Error calculating gEUD for {oar_name} with n={n}. Falling back to mean EQD2.")
                 gEUD = np.mean(eqd2_oar_voxels)


        # Calculate NTCP using the LKB model (Probit function form)
        # t = (gEUD - TD50) / (m * TD50)
        t_numerator = gEUD - TD50
        t_denominator = m * TD50
        if abs(t_denominator) < 1e-9: # Avoid division by zero if m or TD50 is zero
            # If gEUD > TD50, complication is likely high, else low.
            # This is a degenerate case, NTCP would be 0 or 1.
            return 100.0 if t_numerator > 0 else 0.0
            
        t = t_numerator / t_denominator 
        ntcp = 0.5 * (1 + erf(t / np.sqrt(2))) # Standard normal CDF form of LKB
        
        return ntcp * 100.0 # Return as percentage

    def validate_dose(self, dose, monte_carlo_reference=None):
        """
        Validates a calculated dose distribution against a reference using Gamma analysis.
        """
        # The 'dose' input here should ideally be a single beam dose map for validation,
        # not the total accumulated dose from simulation.
        # Let's assume for validation purposes, we calculate the dose for a single beam
        # and compare it to a mock or loaded reference for that single beam.
        
        print("\nPerforming dose validation...")
        # For validation, let's calculate the dose for the first beam with weight 1
        # in the reference phase (phase 0).
        if not self.density_grids_phases or not self.beam_directions:
             print("Cannot perform dose validation: No density grids or beam directions available.")
             return 0.0
             
        eval_dose_single_beam = self._calculate_dose_for_specific_phase([1.0] + [0.0]*(len(self.beam_directions)-1), 0)

        if monte_carlo_reference is None:
            # Create a slightly noisy version of the calculated dose as a mock MC reference
            # This is for demonstration if no real MC data is available.
            print("No Monte Carlo reference provided. Creating a mock reference.")
            noise = np.random.normal(0, 0.02 * np.max(eval_dose_single_beam), eval_dose_single_beam.shape) if np.max(eval_dose_single_beam) > 0 else np.zeros_like(eval_dose_single_beam)
            monte_carlo_reference = eval_dose_single_beam + noise
            monte_carlo_reference[monte_carlo_reference < 0] = 0 # Dose cannot be negative
        else:
             # Assume monte_carlo_reference is a dose map for the first beam
             print("Using provided Monte Carlo reference.")
             # Optional: Resample/align MC reference if grid/resolution differs
             if monte_carlo_reference.shape != self.grid_size:
                  print("Warning: MC reference shape does not match grid size. Gamma analysis may fail or be inaccurate.")
                  # In a real system, resampling/registration would be needed here.
                  # For this example, we'll proceed but expect potential issues.


        gamma_pass_rate = self._gamma_analysis(eval_dose_single_beam, monte_carlo_reference, distance_voxels=3, dose_diff_percent=3)
        print(f"Gamma Pass Rate (3 voxels, 3%): {gamma_pass_rate * 100:.2f}%")
        return gamma_pass_rate

    def _gamma_analysis(self, eval_dose, ref_dose, distance_voxels, dose_diff_percent):
        """
        Performs 3D Gamma analysis.
        Distance is in voxels, dose difference is in percent relative to max reference dose.
        """
        # Ensure doses are numpy arrays
        eval_dose = np.asarray(eval_dose)
        ref_dose = np.asarray(ref_dose)

        if eval_dose.shape != ref_dose.shape:
            print("Error: Dose grids for gamma analysis must have the same shape.")
            return 0.0

        # Normalization dose for dose difference: typically max of reference dose
        ref_dose_max = np.max(ref_dose)
        if ref_dose_max < 1e-9: # If reference dose is effectively zero everywhere
            # If eval_dose is also zero, 100% pass. Otherwise, 0% pass (unless thresholding)
            print("Gamma Analysis: Reference dose is zero everywhere.")
            return 1.0 if np.max(eval_dose) < 1e-9 else 0.0

        dose_diff_abs = (dose_diff_percent / 100.0) * ref_dose_max
        
        passed_points = 0
        total_points_evaluated = 0 
        
        # Define a low dose threshold (e.g., 10% of max reference dose)
        low_dose_threshold = 0.10 * ref_dose_max

        # Iterate over each point in the evaluated dose grid
        # Only evaluate points where the reference dose is above the threshold
        eval_indices = np.argwhere(ref_dose >= low_dose_threshold)

        if eval_indices.size == 0:
             print("Warning: No points above low dose threshold for Gamma Analysis.")
             return 1.0 # Convention: 100% pass if no points meet threshold

        total_points_evaluated = eval_indices.shape[0]

        # Pre-calculate squared criteria
        dist_criterion_sq = distance_voxels**2
        dose_criterion_sq = dose_diff_abs**2

        # Iterate through the indices above the threshold
        for eval_idx_tuple in eval_indices:
            eval_idx = tuple(eval_idx_tuple)
            eval_dose_val = eval_dose[eval_idx]
            ref_dose_val_at_eval_idx = ref_dose[eval_idx] # Reference dose at the same point

            min_gamma_sq_for_point = np.inf

            # Search in a neighborhood around the current point in the reference grid
            # Define search bounds carefully
            min_coords = [max(0, eval_idx[d] - distance_voxels) for d in range(eval_dose.ndim)]
            max_coords = [min(eval_dose.shape[d], eval_idx[d] + distance_voxels + 1) for d in range(eval_dose.ndim)]

            # Create a meshgrid of indices for the search neighborhood
            search_ranges = [range(min_c, max_c) for min_c, max_c in zip(min_coords, max_coords)]
            
            # Iterate through all points in the search neighborhood in the *reference* grid
            for ref_idx_tuple in np.array(np.meshgrid(*search_ranges)).T.reshape(-1, eval_dose.ndim):
                ref_idx = tuple(ref_idx_tuple)
                
                # Calculate squared spatial distance between eval_idx and ref_idx
                dist_spatial_sq = sum([(eval_idx[d] - ref_idx[d])**2 for d in range(eval_dose.ndim)])

                # Calculate dose deviation between eval_dose_val and ref_dose[ref_idx]
                dose_deviation_val = eval_dose_val - ref_dose[ref_idx]
                
                # Calculate Gamma squared for this pair of points
                # gamma_sq = (spatial_distance / DTA)^2 + (dose_difference / Dose_Criterion)^2
                
                term1_sq = dist_spatial_sq / dist_criterion_sq if dist_criterion_sq > 1e-9 else 0
                term2_sq = (dose_deviation_val**2) / dose_criterion_sq if dose_criterion_sq > 1e-9 else 0
                
                current_gamma_sq = term1_sq + term2_sq

                if current_gamma_sq < min_gamma_sq_for_point:
                    min_gamma_sq_for_point = current_gamma_sq
            
            # Check if the minimum gamma for this point is <= 1
            if min_gamma_sq_for_point <= 1.0 + 1e-9: # Add small tolerance for float comparison
                passed_points += 1
        
        pass_rate = passed_points / total_points_evaluated if total_points_evaluated > 0 else 0.0
        return pass_rate


    def plot_results(self, dose, beam_weights, history):
        """
        Plots dose distribution, DVH, and treatment history.
        The 'dose' input here should be the *final accumulated dose*.
        """
        fig = plt.figure(figsize=(18, 6)) # Adjusted figure size
        
        # 1. Dose Distribution (ITV slice) - using final accumulated dose
        ax1 = fig.add_subplot(131)
        if np.any(self.tumor_mask):
            # Show a central slice of the ITV
            itv_indices_z = np.where(self.tumor_mask.any(axis=(0,1)))[0]
            slice_idx_z = itv_indices_z[len(itv_indices_z)//2] if len(itv_indices_z) > 0 else self.grid_size[2] // 2
            
            # Display dose on this slice
            dose_slice = dose[:, :, slice_idx_z]
            im = ax1.imshow(dose_slice.T, cmap='jet', origin='lower', aspect='auto') # Transpose for typical image view
            
            # Overlay ITV contour on this slice
            itv_contour_slice = self.tumor_mask[:, :, slice_idx_z]
            ax1.contour(itv_contour_slice.T, colors='w', linewidths=0.8, origin='lower')
            
            fig.colorbar(im, ax=ax1, label="Dose (Gy_eff)")
            ax1.set_title(f"Final Accumulated Dose on Slice z={slice_idx_z} (ITV)")
            ax1.set_xlabel("X-voxel")
            ax1.set_ylabel("Y-voxel")
        else:
            ax1.text(0.5, 0.5, "No Tumor Mask (ITV)", ha='center', va='center')
            ax1.set_title("Final Accumulated Dose Distribution")

        # 2. Dose-Volume Histogram (DVH) - using final accumulated dose
        ax2 = fig.add_subplot(132)
        max_dose_plot = np.max(dose) if np.max(dose) > 0 else 1.0 # Ensure max_dose > 0 for range
        
        if np.any(self.tumor_mask):
            doses_tumor = dose[self.tumor_mask]
            if doses_tumor.size > 0:
                hist_tumor, bins_tumor = np.histogram(doses_tumor, bins=100, range=(0, max_dose_plot), density=False)
                cumulative_tumor = np.cumsum(hist_tumor[::-1])[::-1] * 100.0 / np.sum(self.tumor_mask) # Differential to cumulative
                ax2.plot(bins_tumor[:-1], cumulative_tumor, label="Tumor (ITV)")

        for region_name, oar_mask in self.oar_masks.items():
            if np.any(oar_mask):
                doses_oar = dose[oar_mask]
                if doses_oar.size > 0: # Ensure there are voxels in the mask
                    hist_oar, bins_oar = np.histogram(doses_oar, bins=100, range=(0, max_dose_plot), density=False)
                    cumulative_oar = np.cumsum(hist_oar[::-1])[::-1] * 100.0 / np.sum(oar_mask)
                    ax2.plot(bins_oar[:-1], cumulative_oar, label=region_name)
        
        ax2.set_title("Cumulative Dose-Volume Histogram (Final Dose)")
        ax2.set_xlabel("Dose (Gy_eff)")
        ax2.set_ylabel("Volume (%)")
        ax2.legend()
        ax2.grid(True, linestyle=':')
        ax2.set_xlim(left=0)
        ax2.set_ylim(0, 100)


        # 3. Treatment History
        ax3 = fig.add_subplot(133)
        fractions = range(1, len(history["tumor_volumes"]) + 1)
        
        if history["tumor_volumes"]:
            ax3.plot(fractions, history["tumor_volumes"], label="ITV Volume", marker='o')
        if history["tcp"]:
            ax3.plot(fractions, history["tcp"], label="TCP (%)", marker='s')
        
        for oar_name, ntcp_history in history["ntcp"].items():
            if ntcp_history: # Check if list is not empty
                 ax3.plot(fractions, ntcp_history, label=f"NTCP {oar_name} (%)", marker='^', linestyle='--')
        
        ax3.set_title("Treatment History")
        ax3.set_xlabel("Fraction Number")
        ax3.set_ylabel("Value")
        ax3.legend()
        ax3.grid(True, linestyle=':')
        ax3.set_ylim(bottom=0) # Ensure y-axis starts at 0 for TCP/NTCP/Volume

        plt.tight_layout()
        plt.savefig("results_enhanced.png")
        print("Plot saved as 'results_enhanced.png'")
        # plt.show() # Optionally show plot
        plt.close(fig)


if __name__ == "__main__":
    # Ensure the dose kernel exists
    if not os.path.exists("dose_kernel.npy"):
        print("Dose kernel not found. Generating a new one...")
        # Assuming generate_dose_kernel.py is in the same directory or accessible
        try:
            from generate_dose_kernel import generate_updated_dose_kernel
            kernel = generate_updated_dose_kernel()
            np.save("dose_kernel.npy", kernel)
            print("Dose kernel generated successfully!")
        except ImportError:
            print("Could not import generate_updated_dose_kernel. Please ensure generate_dose_kernel.py is available.")
            exit()
        except Exception as e:
            print(f"Error generating dose kernel: {e}")
            exit()

    # --- Option 1: Use Simplified Model (no DICOM) ---
    # print("Initializing QRadPlan3D with simplified model...")
    # planner = QRadPlan3D(
    #     grid_size=(50, 50, 50),
    #     num_beams=8,
    #     kernel_path="dose_kernel.npy",
    #     dir_method='simplified_sinusoidal' # This won't be used unless 4D CT paths are given
    # )

    # --- Option 2: Attempt to use DICOM data (requires dummy or real data) ---
    # Create dummy DICOM structure for testing if you don't have real data
    # This is complex to do fully. For now, let's assume paths are set if you want to test.
    # Replace with actual paths if you have DICOM data
    FOURD_CT_BASE_PATH = "./dummy_4d_ct_data" # Example path
    RTSTRUCT_PATH = "./dummy_rtstruct/rtstruct.dcm" # Example path
    STATIC_CT_PATH = "./dummy_static_ct_data" # Example path

    # Create dummy directories for testing if they don't exist (won't create DICOM files)
    # if not os.path.exists(FOURD_CT_BASE_PATH): os.makedirs(FOURD_CT_BASE_PATH)
    # for i in range(2): # Create a couple of dummy phase folders
    #     if not os.path.exists(os.path.join(FOURD_CT_BASE_PATH, f"phase_{i*10}")):
    #         os.makedirs(os.path.join(FOURD_CT_BASE_PATH, f"phase_{i*10}"))
    # if not os.path.exists(os.path.dirname(RTSTRUCT_PATH)): os.makedirs(os.path.dirname(RTSTRUCT_PATH))
    # if not os.path.exists(STATIC_CT_PATH): os.makedirs(STATIC_CT_PATH)
    # print(f"Please place your 4D CT phase directories in: {FOURD_CT_BASE_PATH}")
    # print(f"And your RTSTRUCT file at: {RTSTRUCT_PATH}")
    # print(f"Or static CT DICOMs in: {STATIC_CT_PATH}")
    # print("If these paths are not populated with valid DICOMs, the simplified model will be used.")

    use_dicom = False # Set to True to attempt DICOM loading

    if use_dicom and os.path.exists(FOURD_CT_BASE_PATH) and os.path.exists(RTSTRUCT_PATH):
        print("Attempting to initialize QRadPlan3D with 4D DICOM data...")
        planner = QRadPlan3D(
            kernel_path="dose_kernel.npy",
            fourd_ct_path=FOURD_CT_BASE_PATH,
            dicom_rt_struct_path=RTSTRUCT_PATH,
            reference_phase_name="phase_0", # Adjust if your reference phase dir is named differently
            dir_method='simplified_sinusoidal' # or 'external_sitk' if you implement it
        )
    elif use_dicom and os.path.exists(STATIC_CT_PATH) and os.path.exists(RTSTRUCT_PATH):
        print("Attempting to initialize QRadPlan3D with static 3D DICOM data...")
        planner = QRadPlan3D(
            kernel_path="dose_kernel.npy",
            ct_path=STATIC_CT_PATH,
            dicom_rt_struct_path=RTSTRUCT_PATH
        )
    else:
        print("DICOM paths not valid or 'use_dicom' is False. Initializing with simplified model.")
        planner = QRadPlan3D(
            grid_size=(30, 30, 30), # Smaller for faster simplified model testing
            num_beams=6,
            kernel_path="dose_kernel.npy"
        )

    print("\nRunning fractionated treatment simulation...")
    # Reduce fractions for quicker testing, especially if DICOM loading is slow or grid is large
    history = planner.simulate_fractionated_treatment(num_fractions=3) # Increased fractions for better history plot

    # The final dose for plotting and validation should be the total accumulated dose
    final_accumulated_dose = planner.accumulated_dose

    print("\nValidating dose calculation model...")
    # Validate the dose calculation model itself (e.g., single beam)
    # This requires a reference dose map for a single beam.
    # If you have a Monte Carlo reference for the first beam in the reference phase, load it here.
    # mc_reference_dose = np.load("path/to/your/mc_dose_beam0_phase0.npy") # Example
    # gamma_pass_rate = planner.validate_dose(final_accumulated_dose, monte_carlo_reference=mc_reference_dose)
    
    # For demonstration without a real MC reference, validate against a noisy version of a single calculated beam
    gamma_pass_rate = planner.validate_dose(final_accumulated_dose, monte_carlo_reference=None) # Pass None to use mock reference


    print("\nGenerating plots...")
    # Plot results using the final accumulated dose
    planner.plot_results(final_accumulated_dose, None, history) # Beam weights are not plotted in this version

    print("\n--- Treatment History Summary ---")
    if history["tumor_volumes"]:
        header = f"{'Fraction':<10} | {'ITV Volume':<15} | {'TCP (%)':<10}"
        oar_ntcp_headers = [f"NTCP {oar[:8]:<8} (%)" for oar in planner.oar_masks.keys() if oar in history["ntcp"] and history["ntcp"][oar]]
        print(header + " | " + " | ".join(oar_ntcp_headers))
        print("-" * (len(header) + sum(len(h) + 3 for h in oar_ntcp_headers)))

        for i in range(len(history["tumor_volumes"])):
            row = f"{i+1:<10} | {history['tumor_volumes'][i]:<15.2f} | {history['tcp'][i]:<10.4f}"
            for oar_name in planner.oar_masks.keys():
                 if oar_name in history["ntcp"] and i < len(history["ntcp"][oar_name]):
                     row += f" | {history['ntcp'][oar_name][i]:<16.4f}" # Adjusted width for NTCP oar name
            print(row)
    else:
        print("No treatment history recorded.")

    print("\n--- End of Simulation ---")
