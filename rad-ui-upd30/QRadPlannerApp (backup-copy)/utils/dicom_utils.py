import os
import logging
import zipfile
import tempfile
from typing import List, Tuple, Dict, Optional, Any
import pydicom
import numpy as np

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
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Extract ZIP contents
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        return temp_dir
    except Exception as e:
        logger.error(f"Error processing ZIP file {zip_file_path}: {str(e)}")
        return None

def _sort_dicom_files(dicom_files: List[pydicom.Dataset]) -> List[pydicom.Dataset]:
    """Sort DICOM files by InstanceNumber, then SliceLocation, then filename."""
    
    def get_sort_key(ds):
        instance_number = getattr(ds, 'InstanceNumber', None)
        slice_location = getattr(ds, 'SliceLocation', None)
        filename = getattr(ds, 'filename', '') # filename is part of the pydicom.FileDataset
        return (
            instance_number if instance_number is not None else float('inf'),
            slice_location if slice_location is not None else float('inf'),
            filename
        )

    # Read all datasets first to access sorting attributes
    datasets = []
    for f_path in dicom_files:
        try:
            datasets.append(pydicom.dcmread(f_path))
        except Exception as e:
            logger.warning(f"Could not read DICOM file {f_path} for sorting: {e}")
            # Optionally, you might want to skip files that can't be read or handle them differently

    # Sort the datasets
    datasets.sort(key=get_sort_key)
    return datasets


def load_dicom_series_from_directory(directory_path: str) -> Tuple[Optional[List[pydicom.Dataset]], Optional[np.ndarray], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Loads DICOM series from a directory.

    Args:
        directory_path: Path to the directory containing DICOM files.

    Returns:
        A tuple containing:
            - List of Pydicom datasets (sorted).
            - 3D NumPy array (volume_data).
            - Patient metadata dictionary.
            - Image properties dictionary.
        Returns (None, None, None, None) if loading fails.
    """
    dicom_file_paths = find_dicom_files(directory_path)
    if not dicom_file_paths:
        logger.error(f"No DICOM files found in directory: {directory_path}")
        return None, None, None, None

    # Read and sort DICOM files
    # First, read all files to access metadata for sorting
    datasets_for_sorting = []
    for f_path in dicom_file_paths:
        try:
            datasets_for_sorting.append(pydicom.dcmread(f_path, stop_before_pixels=True)) # stop_before_pixels for faster read if only sorting keys are needed
        except Exception as e:
            logger.warning(f"Could not read DICOM file {f_path} for sorting: {e}")
    
    # Sort based on InstanceNumber, SliceLocation, then filename
    def get_sort_key(ds):
        return (
            getattr(ds, 'InstanceNumber', float('inf')), 
            getattr(ds, 'SliceLocation', float('inf')), 
            ds.filename
        )

    try:
        # Now sort the paths based on the read metadata
        # We need to store the original path with the dataset for sorting
        path_ds_tuples = []
        for f_path in dicom_file_paths:
            try:
                ds = pydicom.dcmread(f_path, stop_before_pixels=True)
                path_ds_tuples.append((f_path, ds))
            except Exception as e:
                logger.warning(f"Could not read DICOM file {f_path} for sorting key extraction: {e}")
        
        path_ds_tuples.sort(key=lambda x: get_sort_key(x[1]))
        
        # Get the sorted list of file paths
        sorted_dicom_file_paths = [item[0] for item in path_ds_tuples]

        # Now read the full datasets in sorted order
        pydicom_datasets = [pydicom.dcmread(f_path) for f_path in sorted_dicom_file_paths]

    except Exception as e:
        logger.error(f"Error sorting DICOM files: {e}. Falling back to filename sort.")
        # Fallback to simple filename sort if complex sort fails
        dicom_file_paths.sort()
        pydicom_datasets = [pydicom.dcmread(f_path) for f_path in dicom_file_paths]


    if not pydicom_datasets:
        logger.error("Could not read any DICOM datasets.")
        return None, None, None, None

    # Extract metadata from the first slice
    first_slice = pydicom_datasets[0]
    
    patient_metadata = {
        'PatientName': str(getattr(first_slice, 'PatientName', 'N/A')),
        'PatientID': str(getattr(first_slice, 'PatientID', 'N/A')),
        'StudyDate': str(getattr(first_slice, 'StudyDate', 'N/A')),
        'Modality': str(getattr(first_slice, 'Modality', 'N/A')),
    }

    # Image properties
    pixel_spacing = getattr(first_slice, 'PixelSpacing', [1.0, 1.0])
    slice_thickness = getattr(first_slice, 'SliceThickness', 1.0)
    image_position = getattr(first_slice, 'ImagePositionPatient', [0.0, 0.0, 0.0])
    image_orientation_patient = getattr(first_slice, 'ImageOrientationPatient', [1,0,0,0,1,0]) # Default to axial
    
    # Construct 3x3 orientation matrix from ImageOrientationPatient (vector of 6)
    # Row cosines (Xx, Xy, Xz) then Column cosines (Yx, Yy, Yz)
    # Z-axis is cross product of X and Y axes
    row_cosines = np.array(image_orientation_patient[:3])
    col_cosines = np.array(image_orientation_patient[3:])
    z_cosines = np.cross(row_cosines, col_cosines)
    orientation_matrix_3x3 = np.array([row_cosines, col_cosines, z_cosines]).T # Transpose to match ITK/VTK convention

    image_properties = {
        'pixel_spacing': [float(ps) for ps in pixel_spacing],
        'slice_thickness': float(slice_thickness),
        'origin': [float(c) for c in image_position], # x, y, z
        'orientation_matrix_3x3': orientation_matrix_3x3.tolist(), # Store as list for easier serialization if needed
        'dimensions': [
            int(getattr(first_slice, 'Columns', 0)), # cols
            int(getattr(first_slice, 'Rows', 0)),    # rows
            len(pydicom_datasets)                    # slices
        ]
    }

    # Stack pixel arrays into a 3D NumPy volume
    # Ensure all slices have the same dimensions
    rows = image_properties['dimensions'][1]
    cols = image_properties['dimensions'][0]
    num_slices = image_properties['dimensions'][2]

    volume_data = np.zeros((num_slices, rows, cols), dtype=np.float32) # Slices, Rows, Cols

    for i, ds in enumerate(pydicom_datasets):
        if hasattr(ds, 'pixel_array') and ds.pixel_array.shape == (rows, cols):
            # Apply RescaleSlope and RescaleIntercept if present
            pixel_array = ds.pixel_array.astype(np.float32)
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                slope = float(getattr(ds, 'RescaleSlope', 1.0))
                intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
                pixel_array = pixel_array * slope + intercept
            volume_data[i, :, :] = pixel_array
        else:
            logger.warning(f"Slice {i} has inconsistent dimensions or missing pixel_array. Filling with zeros.")
            # Handle missing or inconsistent slices, e.g., fill with zeros or raise error

    return pydicom_datasets, volume_data, patient_metadata, image_properties

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
    elif max_val == min_val: # Handle case where all values are the same
        return np.full_like(volume, new_min if min_val < (new_min + new_max) / 2 else new_max)
    return volume # Should not happen if max_val > min_val is false and max_val == min_val is false

if __name__ == '__main__':
    # Example usage (requires dummy DICOM files)
    # Create a dummy DICOM directory for testing
    # Path: /tmp/dummy_dicom_dir/
    # Inside, create dummy_slice1.dcm, dummy_slice2.dcm, etc.
    # You'll need to create actual minimal DICOM files for this to run.
    
    # logger.setLevel(logging.INFO)
    # logging.basicConfig()
    
    # Create dummy files for testing
    # temp_dir_for_test = tempfile.mkdtemp()
    # for i in range(3):
    #     meta = pydicom.Dataset()
    #     meta.PatientName = "Test^Patient"
    #     meta.PatientID = "12345"
    #     meta.StudyDate = "20230101"
    #     meta.Modality = "CT"
    #     meta.InstanceNumber = i + 1
    #     meta.SliceLocation = i * 2.5
    #     meta.PixelSpacing = [0.5, 0.5]
    #     meta.SliceThickness = 2.5
    #     meta.ImagePositionPatient = [0,0, i * 2.5]
    #     meta.ImageOrientationPatient = [1,0,0,0,1,0]
    #     meta.Rows = 2 # Minimal size
    #     meta.Columns = 2
        
    #     # Create minimal pixel data
    #     pixel_data = np.arange(4, dtype=np.uint16).reshape((2,2)) * (i+1)
    #     meta.PixelData = pixel_data.tobytes()
    #     meta.is_little_endian = True
    #     meta.is_implicit_VR = True # Common for CT
        
    #     pydicom.dcmwrite(os.path.join(temp_dir_for_test, f"slice_{i+1}.dcm"), meta, write_like_original=False)

    # print(f"Dummy DICOM files created in: {temp_dir_for_test}")
    # datasets, volume, patient_meta, image_props = load_dicom_series_from_directory(temp_dir_for_test)
    
    # if datasets:
    #     print("DICOM Series Loaded Successfully.")
    #     print("Patient Metadata:", patient_meta)
    #     print("Image Properties:", image_props)
    #     print("Volume shape:", volume.shape)
    #     print("Number of Pydicom datasets:", len(datasets))
    #     print("First dataset filename:", datasets[0].filename)
    #     print("Last dataset filename:", datasets[-1].filename)
    # else:
    #     print("Failed to load DICOM series.")
    
    # # Clean up dummy files
    # import shutil
    # shutil.rmtree(temp_dir_for_test)
    # print(f"Cleaned up dummy directory: {temp_dir_for_test}")
    pass
