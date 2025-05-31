# QRadPlannerApp

## Overview

QRadPlannerApp is a desktop application for medical imaging visualization and radiotherapy planning. It is built in Python using PyQt5 for the user interface and integrates custom planning logic from `radiotherapy_planner.py` (based on `qradplan-may27-no-quantum.py`). The application allows users to load DICOM series, view them in 2D and 3D, perform basic tumor detection, configure and run radiotherapy treatment planning simulations, and analyze results via metrics and (eventually) DVH plots.

## Features

*   **DICOM Loading:** Supports loading DICOM series from local folders and ZIP archives.
*   **2D Viewer:**
    *   Interactive slice-by-slice navigation.
    *   Window Center (WC) / Window Width (WW) adjustments.
    *   Overlay of detected tumor masks.
    *   Toggleable overlay of calculated dose distributions.
    *   Interactive drawing of points and lines on slices for temporary visual marking (finalization and storage of these contours as usable masks is not functional).
*   **3D Viewer:**
    *   VTK-based 3D rendering of the DICOM volume.
    *   Overlay of detected tumor masks as a 3D surface.
    *   The 3D viewer widget also includes backend capability to display dose distribution as isosurfaces.
*   **Tumor Detection:** Includes a basic tumor detection feature using scikit-image.
*   **Radiotherapy Planning Workflow:**
    *   Initializes the `QRadPlan3D` planning engine.
    *   Allows setting the detected tumor mask as the target for planning.
    *   Triggers beam angle optimization.
    *   Calculates 3D dose distributions.
    *   (Optional) Simulates a short fractionated treatment course.
*   **Results & Analysis:**
    *   Calculates and displays key plan metrics (e.g., tumor V95, mean doses, TCP, OAR max/mean doses, NTCP) in text format.
    *   Calculates Dose-Volume Histogram (DVH) data for tumor and OARs.

## Known Limitations

*   **DVH Plot UI:** The graphical DVH plot is not yet displayed in the UI's "Results & Analysis" tab. This was due to technical issues encountered with file modification tools during development, which prevented the final integration of the `DvhPlotWidget`. The DVH data *is* calculated and can be accessed programmatically if needed.
*   **Manual Contour Finalization:** While interactive drawing on 2D slices shows temporary points and connecting lines, the functionality to finalize these into stored contours (for subsequent mask generation and planning) was not completed due to persistent tool errors preventing necessary modifications to UI files.
*   **3D Dose Display UI Controls:** The `DicomViewer3DWidget` has the backend capability to display 3D dose isosurfaces. However, the UI controls within the main application window (to specify isovalues, trigger updates, or clear these isosurfaces) could not be implemented due to the same tool limitations affecting UI file modifications. Thus, this feature is not currently accessible to the end-user through the GUI.
*   **Tumor Detection:** The current tumor detection is based on classical image processing (scikit-image) and is considered basic. Planned enhancements using advanced deep learning models were not implemented due to "No space left on device" errors in the development environment, which prevented the installation of necessary packages like MONAI/PyTorch.
*   **Packaging:** The application is provided as a runnable Python project. It has not been packaged into a standalone executable due to the same environmental disk space limitations.
*   **Radiotherapy Planner Script:** The core `radiotherapy_planner.py` script was largely used as-is regarding its internal structure. New functionalities like metrics calculation were added via external utility functions due to difficulties in reliably modifying this large script with the available development tools.

## Project Structure

```
QRadPlannerApp/
├── main.py                     # Main application entry point
├── requirements.txt            # Python package dependencies
├── README.md                   # This file
│
├── backend/
│   ├── __init__.py
│   ├── data_manager.py         # Manages data, DICOM loading, planner interaction
│   └── radiotherapy_planner.py # Core RT planning engine (qradplan-may27-no-quantum.py)
│
├── features/
│   ├── __init__.py
│   └── tumor_detector.py       # Basic tumor detection logic
│
├── ui/
│   ├── __init__.py
│   ├── main_window.py          # Main application window (PyQt5)
│   ├── dicom_viewer_2d.py      # 2D DICOM slice viewer widget
│   ├── dicom_viewer_3d.py      # 3D DICOM volume viewer widget (VTK based)
│   └── dvh_plot_widget.py      # DVH plotting widget (created but not fully integrated)
│
└── utils/
    ├── __init__.py
    ├── dicom_utils.py          # DICOM file handling utilities
    └── plan_eval_utils.py      # Metrics and DVH calculation utilities
```

## Setup and Installation

1.  **Prerequisites:**
    *   Python 3.7+ is recommended.
    *   Ensure `pip` (Python package installer) is available.

2.  **Get the Code:**
    *   Clone the repository (if applicable) or download the `QRadPlannerApp` directory.

3.  **Create a Virtual Environment (Recommended):**
    *   Open a terminal or command prompt.
    *   Navigate to the directory *containing* the `QRadPlannerApp` folder (i.e., its parent directory).
    *   Create a virtual environment:
        ```bash
        python -m venv venv_qradplanner
        ```
    *   Activate the virtual environment:
        *   Windows: `venv_qradplanner\Scripts\activate`
        *   macOS/Linux: `source venv_qradplanner/bin/activate`

4.  **Install Dependencies:**
    *   Ensure your virtual environment is activated.
    *   Navigate into the `QRadPlannerApp` directory if you are not already there (the directory containing `requirements.txt`).
    *   Install the required packages:
        ```bash
        pip install -r requirements.txt
        ```

5.  **Note on `dose_kernel.npy`:**
    *   The `radiotherapy_planner.py` script (the planning engine) requires a dose kernel file named `dose_kernel.npy` in its working directory (which is `QRadPlannerApp/backend/`).
    *   If this file is not found, the script will attempt to generate it by calling functions from a `generate_dose_kernel.py` script.
    *   **Action Required:** Ensure `generate_dose_kernel.py` (from the original project source) is present in the `QRadPlannerApp/backend/` directory alongside `radiotherapy_planner.py`. If `dose_kernel.npy` already exists and is correct, `generate_dose_kernel.py` might not be strictly needed at runtime but is crucial if the kernel needs to be (re)generated.

## Running the Application

1.  Ensure your virtual environment (if created) is activated.
2.  Navigate to the `QRadPlannerApp` directory in your terminal.
3.  Run the application using:
    ```bash
    python main.py
    ```
    Alternatively, from the parent directory of `QRadPlannerApp`:
    ```bash
    python -m QRadPlannerApp.main
    ```

## Basic Usage Guide

1.  **Load DICOM Data:**
    *   Use the "File" menu in the application.
    *   Select "Open DICOM Folder..." to load an entire directory of DICOM (`.dcm`) files.
    *   Select "Open DICOM ZIP..." to load DICOM files from a ZIP archive.
    *   Status messages will indicate if loading was successful.

2.  **2D Viewer:**
    *   Once data is loaded, the "2D View" tab will display slices.
    *   Use the slider in the right-hand panel to navigate through slices.
    *   Adjust "Window Center" (WC) and "Window Width" (WW) inputs for contrast.
    *   Click "Detect Tumors" to run the basic tumor detection. The mask will overlay in red. You can also click on the image to draw temporary points/lines if drawing mode were fully enabled (currently, these drawings are not saved or used).
    *   If a dose distribution is calculated, check "Show Dose Overlay" to view it.

3.  **3D Viewer:**
    *   The "3D View" tab will display a 3D rendering of the loaded volume and any detected tumor mask. (Backend support for 3D dose isosurfaces was added to the widget, but GUI controls are not available).
    *   Use mouse controls (typically left-click-drag to rotate, right-click-drag or scroll to zoom, middle-click-drag to pan) to interact with the 3D scene.

4.  **Treatment Planning Tab:**
    *   Input parameters like "Number of Beams," "Number of Fractions," and "Target Prescription Dose."
    *   Click "Initialize Planner": This sets up the planning engine with the current data and tumor mask.
    *   Click "Run Beam Optimization": Calculates optimal beam weights/angles.
    *   Click "Calculate Dose Distribution": Computes the 3D dose based on the optimized plan.
    *   Status messages will appear in the text box on this tab.

5.  **Results & Analysis Tab:**
    *   After calculating a dose distribution, click "Calculate Metrics & DVH".
    *   Plan metrics (TCP, NTCP, dose statistics) will be displayed as text.
    *   The DVH plot is intended to appear here but is currently not functional in the UI, though DVH data is calculated.

## Key Dependencies

The application relies on several Python packages. Key dependencies include:

*   PyQt5 (for the GUI)
*   NumPy (for numerical operations)
*   SciPy (for scientific computing, used by planner)
*   Pydicom (for reading DICOM files)
*   Matplotlib (for 2D and DVH plotting)
*   VTK (for 3D rendering)
*   scikit-image (for image processing, used by basic tumor detector)
*   SimpleITK (for image processing utilities)
*   Numba (for performance optimization in the planner)

Refer to `requirements.txt` for a complete list.
