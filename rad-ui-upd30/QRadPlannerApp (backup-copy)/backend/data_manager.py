import os
import logging
import tempfile
import shutil
from typing import Optional, List, Dict, Any

from typing import Optional, List, Dict, Any, Tuple # Added Tuple

import numpy as np
import pydicom
import SimpleITK as sitk

from QRadPlannerApp.utils import dicom_utils
from QRadPlannerApp.features.tumor_detector import TumorDetector
from QRadPlannerApp.backend.radiotherapy_planner import QRadPlan3D
from QRadPlannerApp.utils.plan_eval_utils import (
    calculate_plan_metrics_external,
    generate_dvh_data_external
)

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self):
        self.dicom_series_paths: Optional[List[str]] = None
        self.pydicom_datasets: Optional[List[pydicom.Dataset]] = None
        self.volume_data: Optional[np.ndarray] = None
        self.sitk_image: Optional[sitk.Image] = None
        self.patient_metadata: Optional[Dict[str, Any]] = None
        self.image_properties: Optional[Dict[str, Any]] = None # pixel_spacing, slice_thickness, origin, orientation, dimensions
        self.tumor_mask: Optional[np.ndarray] = None
        self.rt_struct_data: Optional[Dict[str, Any]] = None # Placeholder
        self.dose_distribution: Optional[np.ndarray] = None
        self.plan_results: Optional[Dict[str, Any]] = {} # Initialize as empty dict

        self.tumor_detector = TumorDetector()
        self.planner: Optional[QRadPlan3D] = None
        logger.info("DataManager initialized.")

    def load_dicom_from_zip(self, zip_file_path: str) -> bool:
        """
        Loads DICOM series from a ZIP file.
        Extracts to a temporary directory, loads from there, and cleans up.
        """
        if not os.path.exists(zip_file_path):
            logger.error(f"ZIP file not found: {zip_file_path}")
            return False
        
        temp_dir = dicom_utils.process_zip_file(zip_file_path)
        if not temp_dir:
            logger.error(f"Failed to process ZIP file: {zip_file_path}")
            return False
        
        logger.info(f"DICOM files extracted to temporary directory: {temp_dir}")
        
        success = self.load_dicom_from_folder(temp_dir)
        
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary directory {temp_dir}: {e}")
            
        return success

    def load_dicom_from_folder(self, folder_path: str) -> bool:
        """
        Loads DICOM series from a specified folder.
        Populates instance attributes with loaded data.
        """
        if not os.path.isdir(folder_path):
            logger.error(f"DICOM folder not found: {folder_path}")
            return False

        self.dicom_series_paths = dicom_utils.find_dicom_files(folder_path)
        if not self.dicom_series_paths:
            logger.error(f"No DICOM files found in folder: {folder_path}")
            return False
        logger.info(f"Found {len(self.dicom_series_paths)} DICOM files in {folder_path}.")

        datasets, volume, patient_meta, image_props = dicom_utils.load_dicom_series_from_directory(folder_path)

        if not datasets or volume is None or not patient_meta or not image_props:
            logger.error(f"Failed to load DICOM series from folder: {folder_path}")
            self._reset_data() # Ensure data is cleared on failure
            return False

        self.pydicom_datasets = datasets
        self.volume_data = volume
        self.patient_metadata = patient_meta
        self.image_properties = image_props
        
        logger.info(f"Successfully loaded DICOM data for Patient: {self.patient_metadata.get('PatientName', 'N/A')}")
        logger.info(f"Volume dimensions: {self.image_properties.get('dimensions')}, Voxel data shape: {self.volume_data.shape}")
        logger.info(f"Image properties: Origin: {self.image_properties.get('origin')}, Spacing: {self.image_properties.get('pixel_spacing') + [self.image_properties.get('slice_thickness')]}")

        # Normalize volume data (e.g., to 0-1 range for tumor detector)
        # The tumor detector itself does normalization, but we can do it here too if needed for other purposes
        # For now, we rely on the tumor detector's internal normalization.
        # If specific HU windowing (e.g. for CT) is desired before tumor detection, it would go here.
        # self.volume_data_normalized = dicom_utils._normalize_volume(self.volume_data, 0, 1) 
        # For now, pass original Hounsfield Units or raw pixel values, detector handles normalization

        # Create SimpleITK image
        try:
            # SimpleITK expects spacing in (x, y, z) order.
            # Our image_properties stores pixel_spacing as [x,y] and slice_thickness as z.
            # Volume data is expected as (z, y, x) by GetImageFromArray
            # However, pydicom pixel_array is (Rows, Columns), so volume_data is (Slices, Rows, Columns)
            # which corresponds to (z, y, x) for ITK.
            
            sitk_volume = sitk.GetImageFromArray(self.volume_data)
            
            # Spacing: [spacing_x, spacing_y, spacing_z]
            spacing_xyz = [
                self.image_properties['pixel_spacing'][0], # x
                self.image_properties['pixel_spacing'][1], # y
                self.image_properties['slice_thickness']   # z
            ]
            sitk_volume.SetSpacing(spacing_xyz)

            # Origin: [origin_x, origin_y, origin_z]
            origin_xyz = self.image_properties['origin']
            sitk_volume.SetOrigin(origin_xyz)
            
            # Direction Cosines (Orientation)
            # SimpleITK expects a flattened 9-element list/tuple for direction cosines:
            # (d_xx, d_xy, d_xz, d_yx, d_yy, d_yz, d_zx, d_zy, d_zz)
            # Our image_properties['orientation_matrix_3x3'] is [[Xx, Yx, Zx], [Xy, Yy, Zy], [Xz, Yz, Zz]]
            # This matrix represents column vectors for X, Y, Z axes in patient coordinates.
            # Convert to ITK's row-major flattened representation of the direction matrix.
            # The orientation_matrix_3x3 is stored as list of lists, convert to numpy array first.
            orientation_matrix_np = np.array(self.image_properties['orientation_matrix_3x3'])
            orientation_flat = orientation_matrix_np.flatten(order='F').tolist() # Fortran order for column-major to row-major
            sitk_volume.SetDirection(orientation_flat)

            self.sitk_image = sitk_volume
            logger.info(f"SimpleITK image created successfully. Size: {self.sitk_image.GetSize()}, Spacing: {self.sitk_image.GetSpacing()}, Origin: {self.sitk_image.GetOrigin()}, Direction: {self.sitk_image.GetDirection()}")

        except Exception as e:
            logger.error(f"Failed to create SimpleITK image: {e}")
            # Continue without sitk_image, or handle as critical error
            self.sitk_image = None


        # Perform tumor detection
        # The TumorDetector expects a numpy array (slices, rows, cols)
        # and currently performs its own normalization.
        if self.volume_data is not None:
            logger.info("Starting tumor detection...")
            try:
                self.tumor_mask = self.tumor_detector.detect_tumors(self.volume_data)
                if self.tumor_mask is not None:
                    logger.info(f"Tumor detection completed. Mask shape: {self.tumor_mask.shape}, Unique values: {np.unique(self.tumor_mask, return_counts=True)}")
                    if np.sum(self.tumor_mask) == 0:
                        logger.warning("Tumor detector ran but did not find any tumor regions.")
                else:
                    logger.error("Tumor detection failed to produce a mask.")
            except Exception as e:
                logger.error(f"Error during tumor detection: {e}")
                self.tumor_mask = None
        else:
            logger.warning("Volume data is not available, skipping tumor detection.")
            
        return True

    def _reset_data(self):
        """Resets all data attributes to their initial empty state."""
        self.dicom_series_paths = None
        self.pydicom_datasets = None
        self.volume_data = None
        self.sitk_image = None
        self.patient_metadata = None
        self.image_properties = None
        self.tumor_mask = None
        self.rt_struct_data = None
        self.dose_distribution = None
        self.plan_results = None
        logger.info("DataManager data attributes reset.")

    # Helper methods _extract_metadata and _normalize_volume are now primarily in dicom_utils.py
    # If specific DataManager versions were needed, they would go here.

    def initialize_planner(self, grid_size_override: Optional[Tuple[int,int,int]] = None, num_beams_override: Optional[int] = None) -> bool:
        """
        Initializes the QRadPlan3D planner.
        Derives grid_size from image_properties if not overridden.
        """
        logger.info("Initializing planner...")
        
        grid_size: Optional[Tuple[int, int, int]] = None
        if grid_size_override:
            grid_size = grid_size_override
            logger.info(f"Using overridden grid size: {grid_size}")
        elif self.image_properties and 'dimensions' in self.image_properties:
            # image_properties['dimensions'] is [cols, rows, slices] -> (x, y, z)
            # QRadPlan3D expects (nx, ny, nz) which corresponds to (cols, rows, slices)
            dims = self.image_properties['dimensions']
            if len(dims) == 3 and all(isinstance(d, int) and d > 0 for d in dims):
                grid_size = (dims[0], dims[1], dims[2]) # cols, rows, slices
                logger.info(f"Derived grid size from image properties: {grid_size}")
            else:
                logger.error(f"Invalid dimensions in image_properties: {dims}. Cannot derive grid_size.")
                return False
        else:
            # Default grid size if no information is available - this is less ideal
            # grid_size = (100, 100, 50) # A fallback, consider raising error instead
            logger.error("Image properties or dimensions not available to derive grid_size, and no override provided.")
            return False

        num_beams = num_beams_override if num_beams_override is not None else 8 # Default number of beams
        logger.info(f"Using number of beams: {num_beams}")

        try:
            self.planner = QRadPlan3D(grid_size=grid_size, num_beams=num_beams)
            logger.info(f"QRadPlan3D planner initialized successfully with grid_size={grid_size}, num_beams={num_beams}.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize QRadPlan3D planner: {e}", exc_info=True)
            self.planner = None
            return False

    def set_planner_tumor_mask(self) -> bool:
        """
        Sets the tumor mask in the initialized planner.
        """
        logger.info("Setting tumor mask in planner...")
        if not self.planner:
            logger.error("Planner not initialized. Cannot set tumor mask.")
            return False
        
        if self.tumor_mask is None:
            logger.error("Tumor mask not available in DataManager. Cannot set in planner.")
            # Optionally, could allow setting a zero mask if that's a valid state for the planner
            # For now, require a mask.
            return False

        if not isinstance(self.tumor_mask, np.ndarray) or self.tumor_mask.ndim != 3:
            logger.error(f"Invalid tumor mask type ({type(self.tumor_mask)}) or dimensions ({self.tumor_mask.ndim}). Must be a 3D NumPy array.")
            return False

        # Ensure tumor mask dimensions match planner's grid size if possible
        # Planner's grid_size is (nx, ny, nz) -> (cols, rows, slices)
        # Tumor mask is (slices, rows, cols)
        planner_grid_shape_xyz = self.planner.grid_size # (cols, rows, slices)
        mask_shape_zxy = self.tumor_mask.shape # (slices, rows, cols)
        
        # Reorder mask shape to match planner's (cols, rows, slices) for comparison
        mask_shape_xyz_equivalent = (mask_shape_zxy[2], mask_shape_zxy[1], mask_shape_zxy[0])

        if mask_shape_xyz_equivalent != planner_grid_shape_xyz:
            logger.error(f"Tumor mask dimensions {mask_shape_zxy} (interpreted as {mask_shape_xyz_equivalent} for xyz) "
                         f"do not match planner grid dimensions {planner_grid_shape_xyz}.")
            # Consider allowing this if planner can handle resampling, or error out.
            # For QRadPlan3D, it expects the mask to match its grid.
            return False
            
        try:
            # QRadPlan3D's set_tumor_data expects the mask in (slices, rows, cols) which is tumor_mask.shape
            self.planner.set_tumor_data(tumor_mask_input=self.tumor_mask.astype(bool)) # Ensure boolean
            logger.info("Tumor mask successfully set in planner.")
            return True
        except Exception as e:
            logger.error(f"Failed to set tumor mask in planner: {e}", exc_info=True)
            return False

    def run_beam_optimization(self) -> bool:
        """
        Runs beam optimization using the initialized planner.
        Stores beam weights in self.plan_results.
        """
        logger.info("Running beam optimization...")
        if not self.planner:
            logger.error("Planner not initialized. Cannot run beam optimization.")
            return False
        
        if not self.planner.tumor_mask is not None and not np.any(self.planner.tumor_mask):
            # Check if tumor_mask is set in the planner instance itself
            # This implies set_planner_tumor_mask (or planner.set_tumor_data) should have been called.
            logger.warning("Tumor mask not set in planner or is empty. Optimization might be suboptimal or fail.")
            # Depending on planner requirements, this might be an error or just a warning.
            # QRadPlan3D's optimize_beams might handle this, or might error if tumor_present is False.

        try:
            beam_weights = self.planner.optimize_beams()
            if beam_weights is None:
                logger.error("Beam optimization failed to return weights (returned None).")
                return False
            
            self.plan_results['beam_weights'] = beam_weights
            logger.info(f"Beam optimization successful. Weights: {beam_weights}")
            # Log TCP/NTCP if available from planner after optimization (if it calculates them)
            if hasattr(self.planner, 'tcp_value') and self.planner.tcp_value is not None:
                 logger.info(f"Planner's post-optimization TCP: {self.planner.tcp_value:.4f}")
            if hasattr(self.planner, 'ntcp_values') and self.planner.ntcp_values:
                 logger.info(f"Planner's post-optimization NTCP: {self.planner.ntcp_values}")

            return True
        except Exception as e:
            logger.error(f"Error during beam optimization: {e}", exc_info=True)
            return False

    def calculate_dose_distribution(self) -> bool:
        """
        Calculates the dose distribution using the optimized beam weights.
        Stores the dose in self.dose_distribution.
        """
        logger.info("Calculating dose distribution...")
        if not self.planner:
            logger.error("Planner not initialized. Cannot calculate dose.")
            return False
        
        if 'beam_weights' not in self.plan_results or self.plan_results['beam_weights'] is None:
            logger.error("Beam weights not available in plan_results. Run optimization first.")
            return False
        
        beam_weights = self.plan_results['beam_weights']
        
        try:
            # The calculate_dose method in QRadPlan3D might need to be checked
            # if it implicitly uses self.beam_weights or needs them passed.
            # Based on the provided QRadPlan3D structure, it takes beam_weights as an argument.
            dose_volume = self.planner.calculate_dose(beam_weights=beam_weights)
            
            if dose_volume is None:
                logger.error("Dose calculation failed to return a dose volume (returned None).")
                return False

            self.dose_distribution = dose_volume
            logger.info(f"Dose distribution calculated successfully. Shape: {self.dose_distribution.shape}, "
                        f"Min: {self.dose_distribution.min():.2f}, Max: {self.dose_distribution.max():.2f}, "
                        f"Mean: {self.dose_distribution.mean():.2f}")
            
            # Store dose in plan_results as well, if needed for other components or future reference
            self.plan_results['dose_distribution'] = self.dose_distribution
            return True
        except Exception as e:
            logger.error(f"Error during dose calculation: {e}", exc_info=True)
            return False

    def run_fractionated_simulation(self, num_fractions: int) -> bool:
        """
        Runs a fractionated treatment simulation.
        Stores simulation history and updates the accumulated dose.
        """
        logger.info(f"Running fractionated simulation for {num_fractions} fractions...")
        if not self.planner:
            logger.error("Planner not initialized. Cannot run fractionated simulation.")
            return False

        if 'beam_weights' not in self.plan_results or self.plan_results['beam_weights'] is None:
            logger.error("Beam weights not available. Run optimization before simulation.")
            # QRadPlan3D's simulate_fractionated_treatment uses self.beam_weights,
            # which should be set by its optimize_beams or manually.
            # Here, we assume optimize_beams has been called and weights are in self.plan_results
            # and potentially also set within the planner instance if optimize_beams does that.
            # For robustness, we could explicitly set them in the planner if they aren't there.
            # if self.planner.beam_weights is None:
            #    self.planner.beam_weights = self.plan_results['beam_weights']
            # However, QRadPlan3D.optimize_beams already sets self.beam_weights.
            pass


        try:
            # The simulate_fractionated_treatment method in QRadPlan3D uses self.beam_weights
            # which should have been set by a prior call to self.planner.optimize_beams()
            simulation_history = self.planner.simulate_fractionated_treatment(num_fractions=num_fractions)
            
            if simulation_history is None:
                logger.error("Fractionated simulation failed to return history (returned None).")
                return False

            self.plan_results['simulation_history'] = simulation_history
            
            # Update the main dose_distribution with the accumulated dose from the simulation
            if self.planner.accumulated_dose is not None:
                self.dose_distribution = self.planner.accumulated_dose
                self.plan_results['dose_distribution_accumulated'] = self.planner.accumulated_dose # Also store in plan_results
                logger.info(f"Fractionated simulation successful. History length: {len(simulation_history)}. "
                            f"Accumulated dose updated: Min={self.dose_distribution.min():.2f}, "
                            f"Max={self.dose_distribution.max():.2f}, Mean={self.dose_distribution.mean():.2f}")
            else:
                logger.warning("Fractionated simulation completed, but no accumulated_dose found in planner.")
                # self.dose_distribution might remain as single fraction dose or be None.

            return True
        except Exception as e:
            logger.error(f"Error during fractionated simulation: {e}", exc_info=True)
            return False

    def get_plan_metrics(self, target_prescription_dose: float, num_fractions_for_radiobio: int = 30) -> bool:
        """
        Calculates plan metrics using external utility function and stores them.
        """
        logger.info("Calculating plan metrics via DataManager...")
        if self.dose_distribution is None:
            logger.error("Dose distribution not available. Cannot calculate metrics.")
            return False
        if self.tumor_mask is None: # Required for tumor metrics
            logger.warning("Tumor mask not available. Tumor metrics will be limited/absent.")
            # Allow proceeding, as OAR metrics might still be calculable.
        if not self.planner:
            logger.error("Planner (and its radiobiological params/OAR masks) not initialized. Cannot calculate full metrics.")
            return False
        
        # Determine if self.dose_distribution is fractional or total accumulated
        dose_to_evaluate = self.dose_distribution
        current_num_fractions = num_fractions_for_radiobio

        # If simulation history exists, dose_distribution is likely total accumulated dose from simulation.
        # In this case, for metrics evaluation (especially TCP/NTCP which expect total dose),
        # we should use the number of fractions from the simulation.
        if self.plan_results.get('simulation_history'):
            if hasattr(self.planner, 'num_fractions_for_radiobio'): # If planner stored it from simulation
                current_num_fractions = self.planner.num_fractions_for_radiobio
                logger.info(f"Using num_fractions_for_radiobio from planner: {current_num_fractions} for metrics based on accumulated dose.")
            else: # Fallback if not stored, but this means NTCP's EQD2 might be less accurate
                logger.warning(f"Simulation history present, but num_fractions used for that simulation is not explicitly stored in planner. Using provided {num_fractions_for_radiobio} for metrics.")
        else:
            # dose_distribution is fractional, scale it to total for metrics evaluation
            logger.info(f"Dose distribution appears to be fractional. Scaling by {num_fractions_for_radiobio} for metrics evaluation.")
            dose_to_evaluate = self.dose_distribution * num_fractions_for_radiobio
            # current_num_fractions is already num_fractions_for_radiobio

        try:
            metrics = calculate_plan_metrics_external(
                dose_distribution=dose_to_evaluate,
                tumor_mask=self.tumor_mask,
                oar_masks=self.planner.oar_masks if self.planner else {},
                radiobiological_params=self.planner.radiobiological_params if self.planner else {},
                voxel_volume=self.planner.voxel_volume if self.planner else 0.001, # Default voxel vol if no planner
                target_prescription_dose=target_prescription_dose, # This is total prescription dose
                num_fractions_for_radiobio=current_num_fractions 
            )
            self.plan_results['metrics'] = metrics
            logger.info("Plan metrics calculated and stored successfully.")
            return True
        except Exception as e:
            logger.error(f"Error calculating plan metrics: {e}", exc_info=True)
            return False

    def get_dvh_data(self) -> bool:
        """
        Generates DVH data using external utility function and stores it.
        """
        logger.info("Generating DVH data via DataManager...")
        if self.dose_distribution is None:
            logger.error("Dose distribution not available. Cannot generate DVH data.")
            return False
        
        # DVH should be generated based on the current self.dose_distribution,
        # which could be fractional or total accumulated. User/UI should be aware of this context.
        
        try:
            dvh_data = generate_dvh_data_external(
                dose_distribution=self.dose_distribution,
                tumor_mask=self.tumor_mask,
                oar_masks=self.planner.oar_masks if self.planner else {},
                tumor_mask_name=self.planner.tumor_mask_name if self.planner else "Tumor"
            )
            self.plan_results['dvh_data'] = dvh_data
            logger.info("DVH data generated and stored successfully.")
            return True
        except Exception as e:
            logger.error(f"Error generating DVH data: {e}", exc_info=True)
            return False

if __name__ == '__main__':
    # This part is for basic testing of DataManager.
    # It requires dummy DICOM files or a ZIP.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    
    dm = DataManager()

    # Create a dummy DICOM directory for testing
    # This setup is illustrative. You'd need actual minimal DICOM files.
    temp_test_dicom_dir = tempfile.mkdtemp(prefix="qrad_test_dicoms_")
    print(f"Attempting to create dummy DICOM files in: {temp_test_dicom_dir}")

    try:
        # Create 3 dummy DICOM files
        for i in range(3):
            meta = pydicom.Dataset()
            meta.PatientName = "Test^Patient"
            meta.PatientID = "DMGR_TEST_001"
            meta.StudyDate = "20240101"
            meta.Modality = "CT"
            meta.InstanceNumber = i + 1
            meta.SliceLocation = i * 2.0  # Slices are 2.0mm apart
            meta.PixelSpacing = [0.5, 0.5] # Pixel spacing 0.5mm x 0.5mm
            meta.SliceThickness = 2.0 # Slice thickness 2.0mm
            # ImagePositionPatient: x, y, z of the center of the first transmitted pixel
            meta.ImagePositionPatient = [-50.0, -50.0, -100.0 + i * 2.0] 
            # ImageOrientationPatient: [Xx, Xy, Xz, Yx, Yy, Yz] (standard axial)
            meta.ImageOrientationPatient = [1, 0, 0, 0, 1, 0] 
            meta.Rows = 4 # Minimal size 4x4
            meta.Columns = 4
            meta.SOPInstanceUID = pydicom.uid.generate_uid()
            meta.SeriesInstanceUID = pydicom.uid.generate_uid() # All should ideally share this

            # Create minimal pixel data (e.g., simple gradient)
            # For CT, pixel data is often int16 or uint16 representing Hounsfield Units + intercept
            pixel_array = np.arange(16, dtype=np.int16).reshape((4, 4)) + (i * 10)
            meta.PixelData = pixel_array.tobytes()
            meta.SamplesPerPixel = 1
            meta.PhotometricInterpretation = "MONOCHROME2"
            meta.BitsAllocated = 16
            meta.BitsStored = 16
            meta.HighBit = 15
            meta.PixelRepresentation = 1 # 0 for unsigned, 1 for signed (2's complement)
            # meta.RescaleIntercept = -1024 # Example for CT Hounsfield Units
            # meta.RescaleSlope = 1 

            # Set transfer syntax
            meta.is_little_endian = True
            meta.is_implicit_VR = False # Explicit VR Little Endian is common
            # pydicom.filewriter.dcmwrite handles FileMetaInformation if not present
            
            file_path = os.path.join(temp_test_dicom_dir, f"slice_{i+1}.dcm")
            pydicom.dcmwrite(file_path, meta, write_like_original=False)
            print(f"Created dummy file: {file_path}")

        print(f"Dummy DICOM files created in: {temp_test_dicom_dir}")

        # Test loading from folder
        print("\n--- Testing load_dicom_from_folder ---")
        load_success = dm.load_dicom_from_folder(temp_test_dicom_dir)
        print(f"load_dicom_from_folder success: {load_success}")

        if load_success:
            print(f"Patient Name: {dm.patient_metadata.get('PatientName')}")
            print(f"Volume Data Shape: {dm.volume_data.shape if dm.volume_data is not None else 'None'}")
            print(f"Image Properties: {dm.image_properties}")
            if dm.sitk_image:
                print(f"SITK Image Size: {dm.sitk_image.GetSize()}")
                print(f"SITK Image Spacing: {dm.sitk_image.GetSpacing()}")
                print(f"SITK Image Origin: {dm.sitk_image.GetOrigin()}")
                print(f"SITK Image Direction: {dm.sitk_image.GetDirection()}")

            if dm.tumor_mask is not None:
                print(f"Tumor Mask Shape: {dm.tumor_mask.shape}")
                print(f"Tumor Mask Sum: {np.sum(dm.tumor_mask)}")
            else:
                print("Tumor Mask is None.")
        
        # Clean up dummy files and directory
        print(f"\nCleaning up dummy DICOM directory: {temp_test_dicom_dir}")
        shutil.rmtree(temp_test_dicom_dir)
        print("Cleanup complete.")

    except Exception as e:
        print(f"Error in DataManager __main__ test setup or execution: {e}")
        # Ensure cleanup if error occurs mid-way
        if os.path.exists(temp_test_dicom_dir):
            shutil.rmtree(temp_test_dicom_dir)
            print(f"Cleaned up dummy directory {temp_test_dicom_dir} due to error.")

    # To test load_dicom_from_zip, you would:
    # 1. Create a dummy zip file containing the DICOMs from temp_test_dicom_dir
    # 2. Call dm.load_dicom_from_zip("path/to/dummy.zip")
    # Example:
    # dummy_zip_path = os.path.join(os.getcwd(), "dummy_dicom_test.zip")
    # shutil.make_archive(dummy_zip_path.replace('.zip',''), 'zip', temp_test_dicom_dir)
    # print(f"\n--- Testing load_dicom_from_zip ---")
    # zip_load_success = dm.load_dicom_from_zip(dummy_zip_path)
    # print(f"load_dicom_from_zip success: {zip_load_success}")
    # if os.path.exists(dummy_zip_path):
    #     os.remove(dummy_zip_path)
    pass
