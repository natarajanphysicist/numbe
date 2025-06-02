#--- START OF FILE data_manager.py ---

import os
import logging
import tempfile
import shutil
from typing import Optional, List, Dict, Any, Tuple 

import numpy as np
import pydicom
import SimpleITK as sitk

from QRadPlannerApp.utils import dicom_utils
from QRadPlannerApp.features.tumor_detector import TumorDetector
# Ensure QRadPlan3D is correctly pathed if it's not in backend directly
# from QRadPlannerApp.backend.radiotherapy_planner import QRadPlan3D
# Assuming radiotherapy_planner.py is in the same directory or appropriately in PYTHONPATH
from .radiotherapy_planner import QRadPlan3D # Adjusted for local testing if radiotherapy_planner.py is in same dir
from QRadPlannerApp.utils.plan_eval_utils import (
    calculate_plan_metrics_external,
    generate_dvh_data_external,
    create_mask_from_slice_contours 
)

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self):
        self.dicom_series_paths: Optional[List[str]] = None
        self.current_dicom_folder_path: Optional[str] = None 
        self.pydicom_datasets: Optional[List[pydicom.Dataset]] = None
        self.volume_data: Optional[np.ndarray] = None # Expected (slices, rows, cols)
        self.sitk_image: Optional[sitk.Image] = None
        self.patient_metadata: Optional[Dict[str, Any]] = None
        self.image_properties: Optional[Dict[str, Any]] = None 
        self.tumor_mask: Optional[np.ndarray] = None # Expected (slices, rows, cols)
        self.rt_struct_data: Optional[Dict[str, Any]] = None # Could store the raw RTStruct pydicom dataset
        self.oar_masks_from_rtstruct: Optional[Dict[str, np.ndarray]] = None # Store ZYX OAR masks
        self.dose_distribution: Optional[np.ndarray] = None # Expected (cols, rows, slices) from planner
        self.plan_results: Optional[Dict[str, Any]] = {} 

        self.tumor_detector = TumorDetector()
        self.planner: Optional[QRadPlan3D] = None
        logger.info("DataManager initialized.")

    def load_dicom_from_zip(self, zip_file_path: str) -> bool:
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
        if not os.path.isdir(folder_path):
            logger.error(f"DICOM folder not found: {folder_path}")
            return False
        
        self.current_dicom_folder_path = folder_path 

        # load_dicom_series_from_directory now also returns oar_masks_dict_zyx
        ct_datasets, volume_zyx, patient_meta, image_props, oar_masks_zyx = \
            dicom_utils.load_dicom_series_from_directory(folder_path)

        if not ct_datasets or volume_zyx is None or not patient_meta or not image_props:
            logger.error(f"Failed to load DICOM series from folder: {folder_path}")
            self._reset_data()
            return False

        self.pydicom_datasets = ct_datasets
        self.volume_data = volume_zyx # (slices, rows, cols)
        self.patient_metadata = patient_meta
        self.image_properties = image_props
        self.oar_masks_from_rtstruct = oar_masks_zyx if oar_masks_zyx else {} # Store ZYX OAR masks

        logger.info(f"Successfully loaded DICOM data for Patient: {self.patient_metadata.get('PatientName', 'N/A')}")
        logger.info(f"Volume dimensions (image_props c,r,s): {self.image_properties.get('dimensions')}, Voxel data shape (s,r,c): {self.volume_data.shape}")
        
        if self.oar_masks_from_rtstruct:
            logger.info(f"Loaded OARs from RTStruct: {list(self.oar_masks_from_rtstruct.keys())}")
            for name, mask in self.oar_masks_from_rtstruct.items():
                 logger.debug(f"  OAR '{name}' mask shape (s,r,c): {mask.shape}, Sum: {np.sum(mask)}")
        else:
            logger.info("No OARs loaded from RTStruct (or RTStruct not found/parsed).")

        img_props_spacing_rc = self.image_properties.get('pixel_spacing', [0.0, 0.0]) # row, col
        img_props_slice_thk = self.image_properties.get('slice_thickness', 0.0)
        logger.info(f"Image properties from DICOM: Origin: {self.image_properties.get('origin')}, "
                    f"Spacing (Row, Col, SliceThk): ({img_props_spacing_rc[0]}, {img_props_spacing_rc[1]}, {img_props_slice_thk})")

        try:
            sitk_volume = sitk.GetImageFromArray(self.volume_data) # Expects ZYX
            # ITK Spacing is X,Y,Z -> ColSpacing, RowSpacing, SliceThickness
            spacing_xyz_itk = [
                self.image_properties['pixel_spacing'][1], # ColSpacing
                self.image_properties['pixel_spacing'][0], # RowSpacing
                self.image_properties['slice_thickness']   # SliceThickness
            ]
            sitk_volume.SetSpacing(spacing_xyz_itk)
            origin_xyz_lps = self.image_properties['origin'] # LPS
            sitk_volume.SetOrigin(origin_xyz_lps)
            
            orientation_matrix_itk_style = np.array(self.image_properties['orientation_matrix_3x3'])
            # The orientation_matrix_3x3 in image_properties is already ITK-style
            # where columns are patient axes for image X, Y, Z directions.
            # Flattening with 'F' (Fortran/column-major) correctly gives the 9-element vector.
            orientation_flat_itk = orientation_matrix_itk_style.flatten(order='F').tolist() 
            sitk_volume.SetDirection(orientation_flat_itk)
            self.sitk_image = sitk_volume
            logger.info(f"SimpleITK image created. Size: {self.sitk_image.GetSize()}, Spacing: {self.sitk_image.GetSpacing()}, Origin: {self.sitk_image.GetOrigin()}, Direction: {self.sitk_image.GetDirection()}")
        except Exception as e:
            logger.error(f"Failed to create SimpleITK image: {e}", exc_info=True)
            self.sitk_image = None

        if self.volume_data is not None:
            logger.info("Starting tumor detection...")
            try:
                self.tumor_mask = self.tumor_detector.detect_tumors(self.volume_data) # Expects (s,r,c)
                if self.tumor_mask is not None:
                    logger.info(f"Tumor detection completed. Mask shape (s,r,c): {self.tumor_mask.shape}, Sum: {np.sum(self.tumor_mask)}")
                    if np.sum(self.tumor_mask) == 0: logger.warning("Tumor detector found no tumor regions.")
                else: logger.error("Tumor detection failed (returned None).")
            except Exception as e:
                logger.error(f"Error during tumor detection: {e}", exc_info=True)
                self.tumor_mask = None
        else:
            logger.warning("Volume data not available, skipping tumor detection.")
        return True

    def _reset_data(self):
        self.dicom_series_paths = None
        self.current_dicom_folder_path = None
        self.pydicom_datasets = None
        self.volume_data = None
        self.sitk_image = None
        self.patient_metadata = None
        self.image_properties = None
        self.tumor_mask = None
        self.rt_struct_data = None
        self.oar_masks_from_rtstruct = None # Reset OARs
        self.dose_distribution = None
        self.plan_results = {} 
        logger.info("DataManager data attributes reset.")

    def initialize_planner(self, grid_size_override: Optional[Tuple[int,int,int]] = None, num_beams_override: Optional[int] = None) -> bool:
        logger.info("Initializing planner...")
        grid_size: Optional[Tuple[int, int, int]] = None # Planner expects (cols, rows, slices)
        if grid_size_override:
            grid_size = grid_size_override
            logger.info(f"Using overridden grid size (c,r,s): {grid_size}")
        elif self.image_properties and 'dimensions' in self.image_properties:
            dims_crs = self.image_properties['dimensions'] # (cols, rows, slices)
            if len(dims_crs) == 3 and all(isinstance(d, int) and d > 0 for d in dims_crs):
                grid_size = (dims_crs[0], dims_crs[1], dims_crs[2]) 
                logger.info(f"Derived grid size from image_props (c,r,s): {grid_size}")
            else:
                logger.error(f"Invalid dimensions in image_properties: {dims_crs}. Cannot derive grid_size.")
                return False
        else:
            logger.error("Image properties or dimensions not available for planner grid_size, and no override provided.")
            return False

        num_beams = num_beams_override if num_beams_override is not None else 8
        logger.info(f"DataManager: Initializing QRadPlan3D with grid_size (c,r,s)={grid_size}, num_beams={num_beams}.")

        try:
            self.planner = QRadPlan3D(grid_size=grid_size, num_beams=num_beams)
            logger.info(f"QRadPlan3D planner instance created.")
            
            if self.volume_data is not None and self.image_properties is not None:
                # Pass OARs (ZYX from self.oar_masks_from_rtstruct) to set_patient_data
                # QRadPlan3D.set_patient_data expects ct_volume_hu_zyx (s,r,c), tumor_mask_detected (s,r,c),
                # and oar_masks_loaded (Dict[str, np.ndarray (s,r,c)])
                self.planner.set_patient_data(
                    ct_volume_hu_zyx=self.volume_data,
                    image_properties=self.image_properties,
                    tumor_mask_detected_zyx=self.tumor_mask,
                    oar_masks_loaded_zyx=self.oar_masks_from_rtstruct
                )
                logger.info("Patient data (CT, image_props, tumor_mask, OARs) set into QRadPlan3D engine.")
                if self.oar_masks_from_rtstruct:
                    logger.info(f"  OARs passed to planner: {list(self.oar_masks_from_rtstruct.keys())}")
                if self.planner.oar_masks: # Check what planner ended up with
                     logger.info(f"  Planner now has OAR masks (c,r,s): {list(self.planner.oar_masks.keys())}")

            else:
                logger.warning("Volume data or image properties not available in DataManager. Planner will use simplified model if planning is attempted without explicit data setting by set_patient_data.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize QRadPlan3D planner or set patient data: {e}", exc_info=True)
            self.planner = None
            return False
            
    def set_planner_tumor_mask(self) -> bool:
        logger.info("Setting/updating tumor mask in planner...")
        if not self.planner:
            logger.error("Planner not initialized. Cannot set tumor mask."); return False
        if self.tumor_mask is None:
            logger.error("Tumor mask not available in DataManager. Cannot set in planner."); return False
        if not isinstance(self.tumor_mask, np.ndarray) or self.tumor_mask.ndim != 3:
            logger.error(f"Invalid tumor mask type/dims."); return False

        try:
            # self.tumor_mask is (slices, rows, cols)
            # QRadPlan3D.set_tumor_data expects tumor_mask_input in planner orientation (cols, rows, slices)
            tumor_mask_planner_oriented_crs = np.transpose(self.tumor_mask, (2, 1, 0)).astype(bool)
            self.planner.set_tumor_data(tumor_mask_input=tumor_mask_planner_oriented_crs)
            logger.info("Tumor mask successfully set/updated in planner via set_tumor_data.")
            return True
        except Exception as e:
            logger.error(f"Failed to set tumor mask in planner: {e}", exc_info=True)
            return False

    def run_beam_optimization(self) -> bool:
        logger.info("Running beam optimization...")
        if not self.planner: logger.error("Planner not initialized."); return False
        if self.planner.tumor_mask is None or not np.any(self.planner.tumor_mask):
            logger.warning("Tumor mask in planner is empty/None. Optimization may be suboptimal.")
        try:
            beam_weights = self.planner.optimize_beams() 
            if beam_weights is None: logger.error("Beam optimization returned None."); return False
            self.plan_results['beam_weights'] = beam_weights
            logger.info(f"Beam optimization successful. Weights: {beam_weights}")
            return True
        except Exception as e:
            logger.error(f"Error during beam optimization: {e}", exc_info=True); return False

    def calculate_dose_distribution(self) -> bool:
        logger.info("Calculating dose distribution...")
        if not self.planner: logger.error("Planner not initialized."); return False
        beam_weights_to_use = self.plan_results.get('beam_weights')
        if beam_weights_to_use is None:
            if self.planner.beam_weights is not None: beam_weights_to_use = self.planner.beam_weights
            else: logger.error("Beam weights not available."); return False
        try:
            dose_volume_crs = self.planner.calculate_dose(beam_weights_in=beam_weights_to_use)
            if dose_volume_crs is None: logger.error("Dose calculation returned None."); return False
            self.dose_distribution = dose_volume_crs 
            logger.info(f"Dose distribution calculated. Shape (c,r,s): {self.dose_distribution.shape}, "
                        f"Range: [{self.dose_distribution.min():.2f} - {self.dose_distribution.max():.2f}] Gy")
            self.plan_results['dose_distribution'] = self.dose_distribution
            return True
        except Exception as e:
            logger.error(f"Error during dose calculation: {e}", exc_info=True); return False

    def run_fractionated_simulation(self, num_fractions: int) -> bool:
        logger.info(f"Running fractionated simulation for {num_fractions} fractions...")
        if not self.planner: logger.error("Planner not initialized."); return False
        try:
            simulation_history = self.planner.simulate_fractionated_treatment(num_fractions=num_fractions)
            if simulation_history is None: logger.error("Fractionated simulation returned None."); return False
            self.plan_results['simulation_history'] = simulation_history
            if self.planner.dose_distribution is not None: 
                self.dose_distribution = self.planner.dose_distribution 
                self.plan_results['dose_distribution_accumulated'] = self.planner.dose_distribution 
                logger.info(f"Fractionated simulation successful. Accumulated dose updated.")
            else: logger.warning("Fractionated simulation done, but no final dose_distribution in planner.")
            return True
        except Exception as e:
            logger.error(f"Error during fractionated simulation: {e}", exc_info=True); return False

    def get_plan_metrics(self, target_prescription_dose: float, num_fractions_for_radiobio: int = 30) -> bool:
        logger.info("Calculating plan metrics...")
        if self.dose_distribution is None: logger.error("Dose distribution not available."); return False
        if not self.planner: logger.error("Planner not initialized."); return False
        
        tumor_mask_src_for_metrics = self.tumor_mask # (s,r,c)
        if tumor_mask_src_for_metrics is None:
            logger.warning("DataManager's tumor_mask (s,r,c) not available. Tumor metrics will be limited.")
            
        dose_to_evaluate_crs = self.dose_distribution # (c,r,s)
        current_num_fractions = num_fractions_for_radiobio

        if self.plan_results.get('simulation_history') and 'tumor_volumes_voxels' in self.plan_results['simulation_history']:
            num_simulated_fractions = len(self.plan_results['simulation_history']['tumor_volumes_voxels']) -1 
            if num_simulated_fractions > 0 : current_num_fractions = num_simulated_fractions
            logger.info(f"Using num_fractions from simulation history ({current_num_fractions}) for radiobio metrics.")
        else:
            logger.info(f"No sim history. Assuming dose is fractional. Scaling by {num_fractions_for_radiobio} for metrics.")
            dose_to_evaluate_crs = self.dose_distribution * num_fractions_for_radiobio
        try:
            metrics = calculate_plan_metrics_external(
                dose_distribution_crs=dose_to_evaluate_crs, 
                tumor_mask_src=tumor_mask_src_for_metrics, 
                oar_masks_crs=self.planner.oar_masks, # Planner stores OARs as (c,r,s)
                radiobiological_params=self.planner.radiobiological_params,
                voxel_volume_cm3=self.planner.voxel_volume, 
                target_prescription_dose=target_prescription_dose, 
                num_fractions_for_radiobio=current_num_fractions 
            )
            self.plan_results['metrics'] = metrics
            logger.info("Plan metrics calculated successfully.")
            return True
        except Exception as e:
            logger.error(f"Error calculating plan metrics: {e}", exc_info=True); return False

    def get_dvh_data(self) -> bool:
        logger.info("Generating DVH data...")
        if self.dose_distribution is None: logger.error("Dose distribution not available."); return False
        if not self.planner: logger.error("Planner not initialized for OAR/tumor names."); return False
        
        tumor_mask_src_for_dvh = self.tumor_mask # (s,r,c)
        oar_masks_crs_for_dvh = self.planner.oar_masks # (c,r,s) from planner
        tumor_name_for_dvh = self.planner.tumor_mask_name if hasattr(self.planner, 'tumor_mask_name') else "Detected Tumor"
        try:
            dvh_data = generate_dvh_data_external(
                dose_distribution_crs=self.dose_distribution, 
                tumor_mask_src=tumor_mask_src_for_dvh, 
                oar_masks_crs=oar_masks_crs_for_dvh, 
                tumor_mask_name=tumor_name_for_dvh
            )
            self.plan_results['dvh_data'] = dvh_data
            logger.info("DVH data generated successfully.")
            return True
        except Exception as e:
            logger.error(f"Error generating DVH data: {e}", exc_info=True); return False

    def set_tumor_mask_from_contours(self, slice_contours: Dict[int, List[List[tuple]]]) -> bool:
        if self.volume_data is None:
            logger.error("Cannot create mask from contours: Volume data not loaded."); return False
        volume_shape_zyx = self.volume_data.shape
        logger.info("Attempting to create tumor mask from user-drawn contours.")
        new_mask_zyx = create_mask_from_slice_contours(slice_contours, volume_shape_zyx)
        if new_mask_zyx is not None:
            self.tumor_mask = new_mask_zyx 
            logger.info(f"New tumor mask (s,r,c) created from contours. Shape: {self.tumor_mask.shape}, Sum: {np.sum(self.tumor_mask)}")
            if self.planner:
                logger.info("Updating planner with new tumor mask from contours...")
                # set_planner_tumor_mask takes self.tumor_mask (s,r,c) and passes it to planner.set_tumor_data
                # which expects tumor_mask_input as (c,r,s) after transpose.
                set_planner_success = self.set_planner_tumor_mask() 
                if not set_planner_success:
                    logger.error("Failed to update planner with new tumor mask from contours.")
            return True
        else:
            logger.error("Failed to create tumor mask from contours."); return False

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    dm = DataManager()
    # Setup dummy DICOM directory for testing, including RTSTRUCT
    # (Using code similar to dicom_utils.py's __main__ for setup)
    test_base_dir = tempfile.mkdtemp(prefix="qrad_dm_main_test_")
    ct_series_dir = os.path.join(test_base_dir, "CT_Series_DMTest")
    os.makedirs(ct_series_dir, exist_ok=True)
    logger.info(f"DataManager test: Dummy DICOMs in: {ct_series_dir}")

    try:
        # Create dummy CT (similar to dicom_utils example)
        num_test_slices, rows, cols = 3, 64, 64
        ct_datasets_for_rt = [] # To hold pydicom datasets of CT for RTStruct ref
        series_uid_ct = pydicom.uid.generate_uid()
        for i in range(num_test_slices):
            ct_slice = pydicom.Dataset()
            ct_slice.PatientName = "DMTest^Patient"
            ct_slice.Modality = "CT"; ct_slice.SOPClassUID = pydicom.uid.CTImageStorage
            ct_slice.SOPInstanceUID = pydicom.uid.generate_uid()
            ct_slice.SeriesInstanceUID = series_uid_ct
            ct_slice.InstanceNumber = i + 1; ct_slice.SliceLocation = float(i * 2.0)
            ct_slice.ImagePositionPatient = [-50.0, -50.0, float(i * 2.0 - 2.0)]
            ct_slice.ImageOrientationPatient = [1.0,0.0,0.0, 0.0,1.0,0.0]
            ct_slice.PixelSpacing = [0.5, 0.5]; ct_slice.SliceThickness = 2.0
            ct_slice.Rows = rows; ct_slice.Columns = cols
            ct_slice.BitsAllocated = 16; ct_slice.BitsStored = 12; ct_slice.HighBit = 11
            ct_slice.PixelRepresentation = 1; ct_slice.RescaleIntercept = -1024.0; ct_slice.RescaleSlope = 1.0
            pixel_arr = (np.arange(rows*cols, dtype=np.int16).reshape((rows,cols)) % 500) - 200 
            if i == 1: pixel_arr[28:36, 28:36] = 800 # Higher HU for potential tumor
            ct_slice.PixelData = pixel_arr.tobytes()
            ct_datasets_for_rt.append(ct_slice)
            pydicom.dcmwrite(os.path.join(ct_series_dir, f"ct_slice_{i+1:03d}.dcm"), ct_slice)
        
        # Create dummy RTStruct (simplified, focusing on OAR for this test)
        rtstruct_ds = pydicom.Dataset()
        rtstruct_ds.PatientName = ct_datasets_for_rt[0].PatientName
        rtstruct_ds.Modality = "RTSTRUCT"; rtstruct_ds.SOPClassUID = pydicom.uid.RTStructureSetStorage
        rtstruct_ds.SOPInstanceUID = pydicom.uid.generate_uid()
        rtstruct_ds.StructureSetROISequence = [pydicom.Dataset()]
        rtstruct_ds.StructureSetROISequence[0].ROINumber = 1
        rtstruct_ds.StructureSetROISequence[0].ROIName = "TestOAR_Kidney"
        rtstruct_ds.ROIContourSequence = [pydicom.Dataset()]
        rtstruct_ds.ROIContourSequence[0].ReferencedROINumber = 1
        rtstruct_ds.ROIContourSequence[0].ROIDisplayColor = [0,255,0]
        contour_on_slice = pydicom.Dataset()
        contour_on_slice.ContourGeometricType = "CLOSED_PLANAR"
        # Example contour points for middle slice (LPS) - simplified for test
        origin_mid_lps = np.array(ct_datasets_for_rt[1].ImagePositionPatient)
        ps_r, ps_c = ct_datasets_for_rt[1].PixelSpacing
        # A small square near image center for OAR
        oar_pts_lps = [
            origin_mid_lps[0]+10*ps_c, origin_mid_lps[1]+10*ps_r, origin_mid_lps[2],
            origin_mid_lps[0]+20*ps_c, origin_mid_lps[1]+10*ps_r, origin_mid_lps[2],
            origin_mid_lps[0]+20*ps_c, origin_mid_lps[1]+20*ps_r, origin_mid_lps[2],
            origin_mid_lps[0]+10*ps_c, origin_mid_lps[1]+20*ps_r, origin_mid_lps[2],
        ]
        contour_on_slice.NumberOfContourPoints = len(oar_pts_lps) // 3
        contour_on_slice.ContourData = oar_pts_lps
        contour_on_slice.ContourImageSequence = [pydicom.Dataset()]
        contour_on_slice.ContourImageSequence[0].ReferencedSOPInstanceUID = ct_datasets_for_rt[1].SOPInstanceUID
        rtstruct_ds.ROIContourSequence[0].ContourSequence = [contour_on_slice]
        pydicom.dcmwrite(os.path.join(test_base_dir, "rtstruct_dm_test.dcm"), rtstruct_ds)
        
        logger.info("--- Testing DataManager with CT and RTStruct ---")
        load_success_dm = dm.load_dicom_from_folder(test_base_dir) # Load from base_dir containing CT_Series and rtstruct
        logger.info(f"DataManager load_dicom_from_folder success: {load_success_dm}")

        if load_success_dm:
            logger.info(f"  DM Patient Name: {dm.patient_metadata.get('PatientName')}")
            logger.info(f"  DM Volume Shape (s,r,c): {dm.volume_data.shape if dm.volume_data is not None else 'None'}")
            if dm.tumor_mask is not None: logger.info(f"  DM Auto Tumor Mask (s,r,c): {dm.tumor_mask.shape}, Sum: {np.sum(dm.tumor_mask)}")
            if dm.oar_masks_from_rtstruct:
                logger.info(f"  DM Loaded OARs: {list(dm.oar_masks_from_rtstruct.keys())}")
                for name, mask in dm.oar_masks_from_rtstruct.items():
                    logger.info(f"    OAR '{name}' mask (s,r,c): {mask.shape}, Sum: {np.sum(mask)}")

            logger.info("\n--- Testing DM: initialize_planner ---")
            init_success_dm = dm.initialize_planner()
            logger.info(f"DM initialize_planner success: {init_success_dm}")
            if init_success_dm and dm.planner:
                logger.info(f"  DM Planner grid (c,r,s): {dm.planner.grid_size}")
                if dm.planner.tumor_mask is not None: logger.info(f"  DM Planner tumor mask (c,r,s) sum: {np.sum(dm.planner.tumor_mask)}")
                if dm.planner.oar_masks: logger.info(f"  DM Planner OARs (c,r,s): {list(dm.planner.oar_masks.keys())}")
                
                # Further tests can be added here for optimization, dose calc, etc.
    except Exception as e_main_dm:
        logger.error(f"Error in DataManager __main__ test: {e_main_dm}", exc_info=True)
    finally:
        if os.path.exists(test_base_dir):
            shutil.rmtree(test_base_dir)
            logger.info(f"Cleaned up DataManager test directory: {test_base_dir}")
