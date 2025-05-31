import os
import shutil
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian, CTImageStorage, PYDICOM_IMPLEMENTATION_UID
import numpy as np
import zipfile

print("Script starting...")

def create_single_dummy_dicom(file_path):
    print(f"Attempting to create: {file_path}...")
    try:
        # File meta information
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid() # Unique for this file
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID
        file_meta.ImplementationVersionName = "PYDICOM_TEST"

        # Main dataset
        ds = Dataset()
        ds.file_meta = file_meta

        ds.PatientName = "Test^Patient"
        ds.PatientID = "123456"

        ds.StudyInstanceUID = "1.2.3.4.5" # Fixed Study UID
        ds.SeriesInstanceUID = "1.2.3.4.5.6" # Fixed Series UID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = CTImageStorage

        ds.InstanceNumber = "1"
        ds.SliceLocation = "0.0"

        ds.Modality = "CT"
        ds.Rows = 2
        ds.Columns = 2

        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 1.0

        ds.ImagePositionPatient = [-0.5, -0.5, 0.0]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]

        ds.RescaleSlope = 1
        ds.RescaleIntercept = 0

        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15

        pixel_array = np.array([[0, 100], [200, 300]], dtype=np.uint16)
        ds.PixelData = pixel_array.tobytes()

        ds.is_little_endian = True
        ds.is_implicit_VR = False

        print(f"Dataset configured for {file_path}. Attempting dcmwrite...")
        pydicom.dcmwrite(file_path, ds, write_like_original=False)
        print(f"Successfully created: {file_path}")
        return True
    except Exception as e:
        print(f"Error writing {file_path}: {e}")
        return False

def create_zip_from_series(series_dir, zip_file_path):
    print(f"Attempting to create ZIP: {zip_file_path} from directory {series_dir}...")
    try:
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(series_dir):
                for file in files:
                    file_to_zip = os.path.join(root, file)
                    arcname = os.path.join(os.path.basename(series_dir), file)
                    print(f"Adding to ZIP: {file_to_zip} as {arcname}")
                    zf.write(file_to_zip, arcname=arcname)
        print(f"Successfully created ZIP: {zip_file_path}")
        return True
    except Exception as e:
        print(f"Error creating ZIP {zip_file_path}: {e}")
        return False

if __name__ == "__main__":
    base_dir = "temp_dicom_test"
    series_name = "series_1"
    series_dir = os.path.join(base_dir, series_name)

    print(f"Base directory: {base_dir}")
    print(f"Series directory: {series_dir}")

    if os.path.exists(base_dir):
        print(f"Removing existing base directory: {base_dir}...")
        shutil.rmtree(base_dir)
    print(f"Creating base directory: {base_dir}...")
    os.makedirs(base_dir)
    print(f"Creating series directory: {series_dir}...")
    os.makedirs(series_dir)

    # Create a couple of DICOM files
    file1_ok = create_single_dummy_dicom(os.path.join(series_dir, "slice_001.dcm"))
    if file1_ok:
        # Modify SOPInstanceUID and InstanceNumber for the second file
        ds_for_file2 = pydicom.dcmread(os.path.join(series_dir, "slice_001.dcm"))
        ds_for_file2.SOPInstanceUID = generate_uid()
        ds_for_file2.InstanceNumber = "2"
        ds_for_file2.SliceLocation = "1.0" # Update slice location
        ds_for_file2.ImagePositionPatient = [-0.5, -0.5, 1.0] # Update image position
        pixel_array2 = np.array([[400, 500], [600, 700]], dtype=np.uint16)
        ds_for_file2.PixelData = pixel_array2.tobytes()
        print(f"Dataset for slice_002.dcm configured. Attempting dcmwrite...")
        pydicom.dcmwrite(os.path.join(series_dir, "slice_002.dcm"), ds_for_file2, write_like_original=False)
        print("slice_002.dcm created.")

    zip_file_path = os.path.join(base_dir, f"{series_name}.zip")
    create_zip_from_series(series_dir, zip_file_path)

    print("\nScript finished.")
