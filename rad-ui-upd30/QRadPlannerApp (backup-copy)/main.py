# --- START OF FILE rad-ui-full-02-main/QRadPlannerApp/main.py ---

import os
import sys # Added for QApplication
import logging
import tempfile
import shutil
import zipfile # Added for create_dummy_zip
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset # Added
from pydicom.uid import generate_uid # Added
import numpy as np
from typing import Optional # <-- ADD THIS LINE

from PyQt5.QtWidgets import QApplication # Added for GUI

from QRadPlannerApp.backend.data_manager import DataManager
from QRadPlannerApp.ui.main_window import MainWindow # Added for GUI
from QRadPlannerApp.utils import dicom_utils # Required for create_dummy_zip

# Setup basic logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - %(message)s')
logger = logging.getLogger("QRadPlannerApp") # Changed logger name for app context

def create_dummy_dicom_files(base_dir: str, series_instance_uid: str, num_slices: int = 5, rows: int = 64, cols: int = 64) -> str:
    """Creates dummy DICOM files in a subdirectory of base_dir."""
    dicom_dir = os.path.join(base_dir, f"dicom_series_{series_instance_uid}")
    os.makedirs(dicom_dir, exist_ok=True)
    logger.info(f"Creating dummy DICOM files in: {dicom_dir}")

    for i in range(num_slices):
        slice_pos = float(i * 5.0)
        sop_instance_uid = generate_uid()

        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
        file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
        file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        ds = Dataset()
        ds.file_meta = file_meta
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows = rows
        ds.Columns = cols
        ds.PixelSpacing = [0.5, 0.5]
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        pixel_array = np.arange(rows * cols, dtype=np.uint16).reshape((rows, cols))
        pixel_array += (i * 100)
        ds.PixelData = pixel_array.tobytes()
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = series_instance_uid
        ds.SOPInstanceUID = sop_instance_uid
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.Modality = "CT"
        ds.InstanceNumber = str(i + 1)
        ds.ImagePositionPatient = [-cols/2 * 0.5, -rows/2 * 0.5, slice_pos]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.SliceThickness = 5.0
        ds.SliceLocation = slice_pos
        ds.PatientName = "Test^Patient"
        ds.PatientID = "TestPID001_MainPy" # Differentiate from DataManager test PIDs
        ds.PatientBirthDate = "19700101"
        ds.PatientSex = "O"
        ds.StudyDate = "20240101"
        ds.StudyTime = "120000"
        ds.AccessionNumber = "12345"
        ds.ReferringPhysicianName = "Test^Doctor"
        ds.StudyID = "STUDY001_MainPy"
        ds.SeriesNumber = "1"
        ds.FrameOfReferenceUID = generate_uid()
        ds.PositionReferenceIndicator = ""
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        file_path = os.path.join(dicom_dir, f"ct_slice_{i+1}.dcm")
        try:
            pydicom.dcmwrite(file_path, ds, write_like_original=False)
            logger.debug(f"Successfully wrote dummy DICOM: {file_path}")
        except Exception as e:
            logger.error(f"Failed to write dummy DICOM {file_path}: {e}", exc_info=True)
            return ""
    return dicom_dir

def create_dummy_zip(base_dir: str, dicom_folder_name: str, zip_name: str) -> Optional[str]:
    dicom_folder_path = os.path.join(base_dir, dicom_folder_name)
    zip_file_path = os.path.join(base_dir, zip_name)

    if not os.path.isdir(dicom_folder_path):
        logger.error(f"DICOM folder {dicom_folder_path} does not exist for zipping.")
        return None

    try:
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(dicom_folder_path):
                for file in files:
                    if file.endswith('.dcm'):
                        full_path = os.path.join(root, file)
                        arcname = os.path.relpath(full_path, os.path.dirname(dicom_folder_path))
                        zipf.write(full_path, arcname=arcname)
        logger.info(f"Successfully created dummy ZIP file: {zip_file_path}")
        return zip_file_path
    except Exception as e:
        logger.error(f"Failed to create ZIP file {zip_file_path}: {e}", exc_info=True)
        return None

def run_cli_data_manager_tests():
    """
    Original command-line test workflow for DataManager.
    Can be called if needed, e.g., via a CLI argument.
    """
    logger.info("--- Starting QRadPlannerApp DataManager CLI Test Workflow ---")

    test_data_base_dir = "temp_qrad_cli_test_data"
    if os.path.exists(test_data_base_dir):
        shutil.rmtree(test_data_base_dir)
    os.makedirs(test_data_base_dir, exist_ok=True)
    logger.info(f"CLI Test data will be stored in: {os.path.abspath(test_data_base_dir)}")

    data_manager = DataManager()
    series_uid_cli = generate_uid()
    dummy_dicom_folder_cli = create_dummy_dicom_files(test_data_base_dir, series_uid_cli, num_slices=3, rows=32, cols=32) # Smaller for CLI

    if not dummy_dicom_folder_cli:
        logger.error("CLI: Failed to create dummy DICOM folder. Aborting.")
        return

    logger.info(f"\nCLI: --- Testing DICOM Loading from Folder: {dummy_dicom_folder_cli} ---")
    load_folder_success = data_manager.load_dicom_from_folder(dummy_dicom_folder_cli)
    if load_folder_success:
        logger.info(f"CLI Folder Load: Patient Meta: {data_manager.patient_metadata}")
        logger.info(f"CLI Folder Load: Volume data shape: {data_manager.volume_data.shape if data_manager.volume_data is not None else 'None'}")
        logger.info(f"CLI Folder Load: Tumor mask shape: {data_manager.tumor_mask.shape if data_manager.tumor_mask is not None else 'None'}, Sum: {np.sum(data_manager.tumor_mask) if data_manager.tumor_mask is not None else 'N/A'}")
    else:
        logger.error("CLI Folder Load: Failed.")

    dummy_zip_cli_name = f"dicom_series_cli_{series_uid_cli}.zip"
    dummy_zip_path_cli = create_dummy_zip(test_data_base_dir, os.path.basename(dummy_dicom_folder_cli), dummy_zip_cli_name)

    if dummy_zip_path_cli:
        logger.info(f"\nCLI: --- Testing DICOM Loading from ZIP: {dummy_zip_path_cli} ---")
        dm_zip_cli = DataManager() # Fresh instance
        load_zip_success_cli = dm_zip_cli.load_dicom_from_zip(dummy_zip_path_cli)
        if load_zip_success_cli:
            logger.info(f"CLI ZIP Load: Patient Meta: {dm_zip_cli.patient_metadata}")
        else:
            logger.error("CLI ZIP Load: Failed.")

    logger.info("\nCLI: --- Cleaning up CLI test data ---")
    try:
        if os.path.exists(test_data_base_dir):
            shutil.rmtree(test_data_base_dir)
            logger.info(f"CLI: Successfully removed test data directory: {test_data_base_dir}")
    except Exception as e:
        logger.error(f"CLI: Error during cleanup: {e}", exc_info=True)

    logger.info("--- QRadPlannerApp DataManager CLI Test Workflow Finished ---")


def run_gui_app():
    """Initializes and runs the PyQt5 GUI application."""
    logger.info("Starting QRadPlanner GUI application...")
    app = QApplication(sys.argv)

    # Initialize DataManager for the GUI
    # This instance will be shared across the application.
    data_manager_gui = DataManager()

    main_win = MainWindow(data_manager_gui)
    main_win.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    # Default action is to run the GUI.
    # To run CLI tests, you could add a command-line argument check, e.g.:
    # if len(sys.argv) > 1 and sys.argv[1] == '--cli-test':
    #     run_cli_data_manager_tests()
    # else:
    #     run_gui_app()

    run_gui_app()
