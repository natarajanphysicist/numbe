#--- START OF FILE dicom_utils.py ---

import os
import logging
import zipfile
import tempfile
from typing import List, Tuple, Dict, Optional, Any
import pydicom
import numpy as np
from skimage.draw import polygon2mask # Ensure this is imported

logger = logging.getLogger(__name__)

def find_dicom_files(directory: str) -> List[str]:
    """Recursively find all DICOM files in directory"""
    dicom_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.dcm'):
                dicom_files.append(os.path.join(root, file))
    return dicom_files

def process_zip_file(zip_file_path: str) -> Optional[str]:
    """Process uploaded ZIP file containing DICOM series and return the temp directory path."""
    try:
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        return temp_dir
    except Exception as e:
        logger.error(f"Error processing ZIP file {zip_file_path}: {str(e)}")
        return None

# _sort_dicom_files is not directly used by load_dicom_series_from_directory anymore,
# as sorting logic is now embedded within it. It can be kept as a standalone utility if needed.
def _sort_dicom_files(dicom_file_paths: List[str]) -> List[pydicom.Dataset]: # Takes paths now
    """Sort DICOM files by InstanceNumber, then SliceLocation, then filename."""
    
    def get_sort_key(ds):
        instance_number = getattr(ds, 'InstanceNumber', None)
        slice_location = getattr(ds, 'SliceLocation', None)
        # filename is part of the pydicom.FileDataset if read from file
        filename = getattr(ds, 'filename', '') 
        return (
            instance_number if instance_number is not None else float('inf'),
            slice_location if slice_location is not None else float('inf'),
            filename
        )

    datasets = []
    for f_path in dicom_file_paths:
        try:
            datasets.append(pydicom.dcmread(f_path))
        except Exception as e:
            logger.warning(f"Could not read DICOM file {f_path} for sorting: {e}")

    datasets.sort(key=get_sort_key)
    return datasets


def _rtstruct_roi_to_mask(roi_contour_sequence, structure_set_roi, ref_ct_series, volume_shape_zyx, image_properties):
    """
    Helper to convert a single ROI from RTStruct to a 3D boolean mask.
    volume_shape_zyx is (num_slices, num_rows, num_cols)
    image_properties are from the reference CT series.
    ref_ct_series is the list of pydicom datasets for the CT series.
    Returns a 3D numpy boolean mask (slices, rows, cols).
    """
    num_slices, num_rows, num_cols = volume_shape_zyx
    roi_mask_zyx = np.zeros(volume_shape_zyx, dtype=bool)
    
    sop_uid_to_index = {s.SOPInstanceUID: i for i, s in enumerate(ref_ct_series)}
    
    # Get geometry information from the first CT slice (assuming all slices are consistent)
    # This is a simplification; a robust solution would handle slice-by-slice variations if they exist.
    first_ct_slice = ref_ct_series[0]
    slice_origin_patient_ref = np.array(first_ct_slice.ImagePositionPatient, dtype=float)
    pixel_spacing_rc_ref = np.array(first_ct_slice.PixelSpacing, dtype=float) # [row_spacing, col_spacing]
    slice_thickness_ref = float(first_ct_slice.SliceThickness)
    iop_ref = np.array(first_ct_slice.ImageOrientationPatient, dtype=float) # [Rx,Ry,Rz, Cx,Cy,Cz]

    row_dir_pat_ref = iop_ref[0:3]
    col_dir_pat_ref = iop_ref[3:6]
    # Normal vector (slice direction) can be derived or taken from image_properties if pre-calculated
    # For simplicity here, assume it's orthogonal and can be derived from row/col
    # normal_dir_pat_ref = np.cross(row_dir_pat_ref, col_dir_pat_ref)

    logger.debug(f"Processing ROI: {structure_set_roi.ROIName} for RTStruct conversion.")
    
    for contour_item in roi_contour_sequence.ContourSequence: # Renamed from contour_sequence to avoid name clash
        contour_data_patient_lps = np.array(contour_item.ContourData).reshape(-1, 3) # (x,y,z) patient LPS
        
        slice_idx_contour = -1
        if hasattr(contour_item, 'ContourImageSequence') and contour_item.ContourImageSequence:
            ref_sop_uid = contour_item.ContourImageSequence[0].ReferencedSOPInstanceUID
            if ref_sop_uid in sop_uid_to_index:
                slice_idx_contour = sop_uid_to_index[ref_sop_uid]
            else: # SOP UID not found in reference CTs
                logger.warning(f"SOPInstanceUID {ref_sop_uid} from RTStruct contour for ROI {structure_set_roi.ROIName} not found in reference CT series. Attempting Z-match.")
        
        # Fallback or primary method: Z-coordinate matching
        if slice_idx_contour == -1:
            contour_z_lps = contour_data_patient_lps[0, 2] # Z-coordinate of the first point of the contour
            min_dist_z = float('inf')
            for i_s, ct_slice_ds in enumerate(ref_ct_series):
                # ImagePositionPatient is the (x,y,z) of the center of the first voxel transmitted
                slice_z_lps = float(ct_slice_ds.ImagePositionPatient[2])
                dist_z = abs(contour_z_lps - slice_z_lps)
                if dist_z < min_dist_z:
                    min_dist_z = dist_z
                    slice_idx_contour = i_s
            
            # Check if the closest slice is reasonably close (e.g., within half slice thickness)
            if min_dist_z > (slice_thickness_ref / 2.0 + 1e-3): # Add epsilon for float comparisons
                logger.warning(f"Contour Z={contour_z_lps:.2f} for ROI {structure_set_roi.ROIName} is far from any CT slice plane. "
                               f"Closest slice {slice_idx_contour} at Z={ref_ct_series[slice_idx_contour].ImagePositionPatient[2]:.2f} (dist {min_dist_z:.2f}). Skipping contour.")
                continue
        
        if not (0 <= slice_idx_contour < num_slices):
            logger.warning(f"Mapped slice index {slice_idx_contour} for ROI {structure_set_roi.ROIName} is out of bounds ({num_slices} slices). Skipping contour.")
            continue

        # Now, transform contour points from LPS to this slice's pixel coordinates (row, col)
        # Get specific geometry for this slice_idx_contour
        current_slice_ds = ref_ct_series[slice_idx_contour]
        slice_origin_lps = np.array(current_slice_ds.ImagePositionPatient, dtype=float)
        slice_iop = np.array(current_slice_ds.ImageOrientationPatient, dtype=float)
        slice_pixel_spacing_rc = np.array(current_slice_ds.PixelSpacing, dtype=float) # [row, col]

        row_dir = slice_iop[0:3]
        col_dir = slice_iop[3:6]

        # Transform: (Point_LPS - SliceOrigin_LPS) dotted with RowDir / RowSpacing = RowIndex
        #            (Point_LPS - SliceOrigin_LPS) dotted with ColDir / ColSpacing = ColIndex
        
        # Displacement vector from slice origin to each contour point, in LPS
        disp_vectors_lps = contour_data_patient_lps - slice_origin_lps
        
        # Project these displacement vectors onto the row and column direction vectors
        # Remember: pixel_spacing_rc is [row_spacing, col_spacing]
        row_indices_float = np.dot(disp_vectors_lps, row_dir) / slice_pixel_spacing_rc[0]
        col_indices_float = np.dot(disp_vectors_lps, col_dir) / slice_pixel_spacing_rc[1]
        
        # polygon2mask expects (row, col) pairs
        contour_rc_vox_coords = np.vstack((row_indices_float, col_indices_float)).T
        
        if len(contour_rc_vox_coords) >= 3:
            try:
                # polygon2mask's shape is (num_rows, num_cols)
                poly_mask_on_slice_yx = polygon2mask((num_rows, num_cols), contour_rc_vox_coords)
                roi_mask_zyx[slice_idx_contour, :, :] |= poly_mask_on_slice_yx
            except Exception as e_poly:
                logger.error(f"Error rasterizing polygon for ROI {structure_set_roi.ROIName} on slice {slice_idx_contour}: {e_poly}")
        else:
            logger.debug(f"Skipping contour on slice {slice_idx_contour} for ROI {structure_set_roi.ROIName} with < 3 points after transform.")
            
    if np.any(roi_mask_zyx):
        logger.info(f"Successfully created mask for ROI '{structure_set_roi.ROIName}', sum: {np.sum(roi_mask_zyx)}")
    else:
        logger.warning(f"Mask for ROI '{structure_set_roi.ROIName}' is empty after processing all contours.")
    return roi_mask_zyx


def load_dicom_series_from_directory(directory_path: str) -> Tuple[Optional[List[pydicom.Dataset]], Optional[np.ndarray], Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, np.ndarray]]]:
    """
    Loads DICOM CT series and an optional RTStruct from a directory.
    Returns: (pydicom_datasets, volume_data_zyx, patient_meta, image_props, oar_masks_dict_zyx)
    volume_data_zyx: (slices, rows, cols)
    oar_masks_dict_zyx: Keys are OAR names, values are (slices,rows,cols) boolean masks.
    """
    dicom_file_paths = find_dicom_files(directory_path)
    if not dicom_file_paths:
        logger.error(f"No DICOM files found in directory: {directory_path}")
        return None, None, None, None, None

    datasets_to_sort_ct = []
    rtstruct_path: Optional[str] = None

    for f_path in dicom_file_paths:
        try:
            ds_peek = pydicom.dcmread(f_path, stop_before_pixels=True) # Read only headers
            if hasattr(ds_peek, 'SOPClassUID'):
                if ds_peek.SOPClassUID == pydicom.uid.RTStructureSetStorage:
                    if rtstruct_path is None: 
                        rtstruct_path = f_path
                    else:
                        logger.warning(f"Multiple RTStruct files found. Using first one: {rtstruct_path}. Ignoring: {f_path}")
                # Identify CT images more reliably by SOPClassUID or Modality
                elif ds_peek.SOPClassUID == pydicom.uid.CTImageStorage or \
                     (hasattr(ds_peek, 'Modality') and ds_peek.Modality.upper() == 'CT'):
                    datasets_to_sort_ct.append({
                        'path': f_path, 
                        'InstanceNumber': getattr(ds_peek, 'InstanceNumber', float('inf')),
                        'SliceLocation': getattr(ds_peek, 'SliceLocation', float('inf')),
                        'ImagePositionPatientZ': getattr(ds_peek, 'ImagePositionPatient', [0,0,float('inf')])[2] # Z-coord for tie-breaking
                    })
        except Exception as e:
            logger.warning(f"Could not peek DICOM file {f_path} for initial classification: {e}")

    # Sort CT image paths based on SliceLocation, then ImagePositionPatient Z, then InstanceNumber
    datasets_to_sort_ct.sort(key=lambda x: (x['SliceLocation'], x['ImagePositionPatientZ'], x['InstanceNumber']))
    sorted_ct_image_paths = [item['path'] for item in datasets_to_sort_ct]

    if not sorted_ct_image_paths:
        logger.error("No CT image files found for series construction.")
        return None, None, None, None, None
        
    pydicom_datasets_ct = []
    for f_path in sorted_ct_image_paths:
        try:
            pydicom_datasets_ct.append(pydicom.dcmread(f_path))
        except Exception as e:
            logger.error(f"Error reading sorted CT DICOM file {f_path}: {e}. Skipping.")
    
    if not pydicom_datasets_ct:
        logger.error("Failed to read any CT DICOM datasets after sorting.")
        return None, None, None, None, None

    first_slice = pydicom_datasets_ct[0]
    patient_metadata = _extract_metadata_from_pydicom(first_slice)
    
    pixel_spacing_rc_raw = getattr(first_slice, 'PixelSpacing', [1.0, 1.0]) # [row_spacing, col_spacing]
    slice_thickness_val_raw = getattr(first_slice, 'SliceThickness', 1.0)
    image_position_pat_raw = getattr(first_slice, 'ImagePositionPatient', [0.0, 0.0, 0.0])
    image_orientation_pat_raw = getattr(first_slice, 'ImageOrientationPatient', [1.0,0.0,0.0,0.0,1.0,0.0]) # Ensure float
    
    row_cosines = np.array(image_orientation_pat_raw[:3], dtype=float)
    col_cosines = np.array(image_orientation_pat_raw[3:], dtype=float)
    z_cosines = np.cross(row_cosines, col_cosines)
    orientation_matrix_3x3 = np.array([row_cosines, col_cosines, z_cosines], dtype=float).T 

    image_properties = {
        'pixel_spacing': [float(ps) for ps in pixel_spacing_rc_raw], # [row_spacing, col_spacing]
        'slice_thickness': float(slice_thickness_val_raw),
        'origin': [float(c) for c in image_position_pat_raw], # LPS of first voxel's center
        'orientation_matrix_3x3': orientation_matrix_3x3.tolist(), # Columns are patient axes for image X,Y,Z
        'dimensions': [ # (cols, rows, slices) - Planner's preferred XYZ interpretation
            int(getattr(first_slice, 'Columns', 0)),
            int(getattr(first_slice, 'Rows', 0)),
            len(pydicom_datasets_ct)
        ]
    }
    
    num_slices_ct = image_properties['dimensions'][2]
    num_rows_ct = image_properties['dimensions'][1]
    num_cols_ct = image_properties['dimensions'][0]
    
    volume_data_zyx = np.zeros((num_slices_ct, num_rows_ct, num_cols_ct), dtype=np.float32)

    for i, ds_ct in enumerate(pydicom_datasets_ct):
        if hasattr(ds_ct, 'pixel_array') and ds_ct.pixel_array.shape == (num_rows_ct, num_cols_ct):
            pixel_array = ds_ct.pixel_array.astype(np.float32)
            slope = float(getattr(ds_ct, 'RescaleSlope', 1.0))
            intercept = float(getattr(ds_ct, 'RescaleIntercept', 0.0))
            volume_data_zyx[i, :, :] = pixel_array * slope + intercept
        else:
            logger.warning(f"CT Slice {i} has inconsistent dimensions or missing pixel_array. Filling with zeros.")

    oar_masks_dict_zyx: Dict[str, np.ndarray] = {}
    if rtstruct_path:
        logger.info(f"Attempting to load OARs from RTStruct: {rtstruct_path}")
        try:
            rt_struct_ds = pydicom.dcmread(rtstruct_path)
            if hasattr(rt_struct_ds, 'StructureSetROISequence') and \
               hasattr(rt_struct_ds, 'ROIContourSequence'):
                
                roi_map_by_number = {roi.ROINumber: roi for roi in rt_struct_ds.StructureSetROISequence}

                for roi_contour_item in rt_struct_ds.ROIContourSequence:
                    roi_number_ref = roi_contour_item.ReferencedROINumber
                    if roi_number_ref in roi_map_by_number:
                        structure_set_roi_info = roi_map_by_number[roi_number_ref]
                        roi_name_rt = structure_set_roi_info.ROIName
                        
                        # Crude check for target volumes to exclude them from OARs
                        # This should be made more robust or configurable
                        is_target_volume = False
                        target_keywords = ["tumor", "gtv", "ptv", "ctv", "target", "lesion"]
                        for keyword in target_keywords:
                            if keyword in roi_name_rt.lower():
                                is_target_volume = True
                                break
                        if is_target_volume:
                            logger.info(f"Skipping ROI '{roi_name_rt}' as it appears to be a target volume.")
                            continue

                        logger.info(f"Processing OAR from RTStruct: {roi_name_rt}")
                        oar_mask_single_zyx = _rtstruct_roi_to_mask(
                            roi_contour_item, 
                            structure_set_roi_info, 
                            pydicom_datasets_ct, # Pass the list of CT pydicom datasets
                            volume_data_zyx.shape, # (num_slices, num_rows, num_cols)
                            image_properties # Pass the derived image properties of the CT
                        )
                        if oar_mask_single_zyx is not None and np.any(oar_mask_single_zyx):
                            oar_masks_dict_zyx[roi_name_rt] = oar_mask_single_zyx
                        else:
                            logger.warning(f"Generated mask for OAR '{roi_name_rt}' was empty or generation failed.")
            else:
                logger.warning(f"RTStruct file {rtstruct_path} is missing StructureSetROISequence or ROIContourSequence.")
        except Exception as e_rt:
            logger.error(f"Failed to load or parse RTStruct {rtstruct_path}: {e_rt}", exc_info=True)

    return pydicom_datasets_ct, volume_data_zyx, patient_metadata, image_properties, oar_masks_dict_zyx


def _extract_metadata_from_pydicom(ds: pydicom.Dataset) -> Dict:
    """Helper to extract common patient metadata from a pydicom dataset."""
    return {
        'PatientName': str(getattr(ds, 'PatientName', 'N/A')),
        'PatientID': str(getattr(ds, 'PatientID', 'N/A')),
        'StudyDate': str(getattr(ds, 'StudyDate', 'N/A')),
        'Modality': str(getattr(ds, 'Modality', 'N/A')),
    }

def _normalize_volume(volume: np.ndarray, new_min: float = 0.0, new_max: float = 1.0) -> np.ndarray:
    """Normalize volume data to a new min-max range."""
    min_val = np.min(volume)
    max_val = np.max(volume)
    if max_val > min_val:
        return new_min + ((volume - min_val) / (max_val - min_val)) * (new_max - new_min)
    elif max_val == min_val: 
        return np.full_like(volume, new_min if min_val < (new_min + new_max) / 2 else new_max)
    return volume 

if __name__ == '__main__':
    # Example usage for testing this module independently
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    
    # Create a dummy DICOM directory with CT and RTSTRUCT for testing
    test_base_dir = tempfile.mkdtemp(prefix="qrad_dicom_utils_test_")
    ct_series_dir = os.path.join(test_base_dir, "CT_Series")
    os.makedirs(ct_series_dir, exist_ok=True)
    
    logger.info(f"Test DICOMs will be created in: {ct_series_dir}")

    try:
        # Create dummy CT slices
        num_test_slices = 3
        rows, cols = 64, 64 # Small but usable for testing polygon2mask
        ct_datasets_for_rt = []
        for i in range(num_test_slices):
            ct_slice = pydicom.Dataset()
            ct_slice.PatientName = "Test^Patient^DICOMUtils"
            ct_slice.PatientID = "DU_001"
            ct_slice.StudyDate = "20240315"
            ct_slice.Modality = "CT"
            ct_slice.SOPClassUID = pydicom.uid.CTImageStorage
            ct_slice.SOPInstanceUID = pydicom.uid.generate_uid()
            ct_slice.SeriesInstanceUID = "1.2.3.4.5.6.7.8.9.0" # Same for all CT slices
            ct_slice.InstanceNumber = i + 1
            ct_slice.SliceLocation = float(i * 2.5) 
            ct_slice.ImagePositionPatient = [-50.0, -50.0, float(i * 2.5 - 2.5)] # Example origin
            ct_slice.ImageOrientationPatient = [1.0,0.0,0.0, 0.0,1.0,0.0]
            ct_slice.PixelSpacing = [0.5, 0.5] # row, col
            ct_slice.SliceThickness = 2.5
            ct_slice.Rows = rows
            ct_slice.Columns = cols
            ct_slice.BitsAllocated = 16
            ct_slice.BitsStored = 12
            ct_slice.HighBit = 11
            ct_slice.PixelRepresentation = 1 # Signed
            ct_slice.RescaleIntercept = -1024.0
            ct_slice.RescaleSlope = 1.0
            
            # Simple pixel data (e.g., a ramp)
            pixel_arr = (np.arange(rows*cols, dtype=np.int16).reshape((rows,cols)) % 500) - 200 
            ct_slice.PixelData = pixel_arr.tobytes()
            ct_datasets_for_rt.append(ct_slice)
            pydicom.dcmwrite(os.path.join(ct_series_dir, f"ct_slice_{i+1:03d}.dcm"), ct_slice)

        # Create a dummy RTStruct file
        rtstruct_ds = pydicom.Dataset()
        rtstruct_ds.PatientName = ct_datasets_for_rt[0].PatientName
        rtstruct_ds.PatientID = ct_datasets_for_rt[0].PatientID
        rtstruct_ds.Modality = "RTSTRUCT"
        rtstruct_ds.SOPClassUID = pydicom.uid.RTStructureSetStorage
        rtstruct_ds.SOPInstanceUID = pydicom.uid.generate_uid()
        rtstruct_ds.StructureSetLabel = "TestStructureSet"
        rtstruct_ds.StructureSetName = "TestStructureSet"
        rtstruct_ds.StructureSetDate = "20240315"
        rtstruct_ds.StructureSetTime = "120000"
        
        # Referenced Frame of Reference Sequence
        ref_frame_of_ref = pydicom.Dataset()
        ref_frame_of_ref.FrameOfReferenceUID = getattr(ct_datasets_for_rt[0], 'FrameOfReferenceUID', pydicom.uid.generate_uid()) # Use CT's FoR if available
        # ... (other necessary tags for ReferencedFrameOfReferenceSequence)
        rtstruct_ds.ReferencedFrameOfReferenceSequence = [ref_frame_of_ref]


        # Structure Set ROI Sequence
        structure_set_roi_seq = []
        roi_1 = pydicom.Dataset()
        roi_1.ROINumber = 1
        roi_1.ReferencedFrameOfReferenceUID = ref_frame_of_ref.FrameOfReferenceUID
        roi_1.ROIName = "OAR_Test_Kidney"
        roi_1.ROIGenerationAlgorithm = "MANUAL"
        structure_set_roi_seq.append(roi_1)
        rtstruct_ds.StructureSetROISequence = structure_set_roi_seq

        # ROI Contour Sequence
        roi_contour_seq = []
        contour_1 = pydicom.Dataset()
        contour_1.ReferencedROINumber = 1
        contour_1.ROIDisplayColor = [255,0,0] # Red

        # Example contour on the middle CT slice (index 1)
        middle_slice_idx = 1 
        ref_sop_uid_middle_slice = ct_datasets_for_rt[middle_slice_idx].SOPInstanceUID
        
        # Define contour points in patient LPS - this needs to be realistic for the geometry
        # For a 64x64 image with 0.5mm pixels, centered, slice origin is (-15.75, -15.75, Z)
        # A 10x10mm box (20x20 pixels) centered in slice (at pixel 32,32)
        # Origin: ImagePositionPatient of middle slice
        origin_middle_slice_lps = np.array(ct_datasets_for_rt[middle_slice_idx].ImagePositionPatient)
        ps_row, ps_col = ct_datasets_for_rt[middle_slice_idx].PixelSpacing
        
        # Define a simple square contour (approx 10mm x 10mm) in the center of the slice plane
        # Assuming standard axial orientation [1,0,0,0,1,0]
        # Points (x,y,z) in LPS:
        # Center of image plane: origin_x + (cols/2)*ps_col, origin_y + (rows/2)*ps_row
        center_x_lps = origin_middle_slice_lps[0] + (cols/2 - 0.5) * ps_col # -0.5 for voxel center convention
        center_y_lps = origin_middle_slice_lps[1] + (rows/2 - 0.5) * ps_row
        z_lps_contour = origin_middle_slice_lps[2]

        contour_points_lps = [
            center_x_lps - 5, center_y_lps - 5, z_lps_contour, # Top-left
            center_x_lps + 5, center_y_lps - 5, z_lps_contour, # Top-right
            center_x_lps + 5, center_y_lps + 5, z_lps_contour, # Bottom-right
            center_x_lps - 5, center_y_lps + 5, z_lps_contour, # Bottom-left
        ]
        
        contour_on_slice_item = pydicom.Dataset()
        contour_on_slice_item.ContourGeometricType = "CLOSED_PLANAR"
        contour_on_slice_item.NumberOfContourPoints = len(contour_points_lps) // 3
        contour_on_slice_item.ContourData = contour_points_lps
        
        # ContourImageSequence linking to the specific CT slice
        contour_img_seq_item = pydicom.Dataset()
        contour_img_seq_item.ReferencedSOPClassUID = pydicom.uid.CTImageStorage
        contour_img_seq_item.ReferencedSOPInstanceUID = ref_sop_uid_middle_slice
        contour_on_slice_item.ContourImageSequence = [contour_img_seq_item]
        
        contour_1.ContourSequence = [contour_on_slice_item]
        roi_contour_seq.append(contour_1)
        rtstruct_ds.ROIContourSequence = roi_contour_seq
        
        pydicom.dcmwrite(os.path.join(test_base_dir, "rtstruct.dcm"), rtstruct_ds)
        logger.info(f"Dummy RTStruct created in: {test_base_dir}")

        # Test loading
        datasets, volume, patient_meta, image_props, oar_masks = load_dicom_series_from_directory(test_base_dir)
        
        if datasets:
            logger.info("DICOM Series Loaded Successfully.")
            logger.info(f"Patient Metadata: {patient_meta}")
            logger.info(f"Image Properties: {image_props}")
            logger.info(f"Volume shape (Z,Y,X): {volume.shape}")
            logger.info(f"Number of Pydicom CT datasets: {len(datasets)}")
            if oar_masks:
                logger.info(f"OARs loaded: {list(oar_masks.keys())}")
                for name, mask in oar_masks.items():
                    logger.info(f"  OAR '{name}' mask shape (Z,Y,X): {mask.shape}, Sum: {np.sum(mask)}")
            else:
                logger.info("No OARs loaded from RTStruct or RTStruct not found/parsed.")
        else:
            logger.error("Failed to load DICOM series from test directory.")
    
    except Exception as e_test:
        logger.error(f"Error during dicom_utils self-test: {e_test}", exc_info=True)
    finally:
        # Clean up dummy files and directory
        if os.path.exists(test_base_dir):
            shutil.rmtree(test_base_dir)
            logger.info(f"Cleaned up test directory: {test_base_dir}")
    pass
