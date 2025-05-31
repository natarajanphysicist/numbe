import os
import sys
import gc
import time
import logging
import importlib.util
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import zipfile
import tempfile
import shutil
import io
import glob

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pydicom
import nibabel as nib
import cv2
import torch
import torch.nn as nn
from PIL import Image
from scipy.ndimage import gaussian_filter, label
from skimage.measure import regionprops
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import SimpleITK as sitk
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
current_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_KERNEL_PATH = 'dose_kernel.npy'
CHUNK_SIZE = 1024 * 1024  # 1MB chunks

# Configure Streamlit
st.set_page_config(
    page_title="StarViewer RT",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Increase message size limit
st.config.set_option('server.maxMessageSize', 1000)  # 1000MB limit

class TumorDetector:
    def __init__(self):
        self.model = None
        self.threshold = 0.5
        
    def detect_tumors(self, volume_data: np.ndarray) -> np.ndarray:
        """Detect tumors in the volume data"""
        try:
            # Convert to float32
            volume = volume_data.astype(np.float32)
            
            # Normalize
            volume = (volume - volume.min()) / (volume.max() - volume.min())
            
            # Apply Gaussian blur
            volume = gaussian_filter(volume, sigma=1)
            
            # Find local maxima
            coordinates = peak_local_max(volume, min_distance=20, threshold_abs=0.3)
            
            # Create mask
            mask = np.zeros_like(volume)
            for coord in coordinates:
                mask[coord[0], coord[1], coord[2]] = 1
            
            # Apply watershed
            distance = gaussian_filter(volume, sigma=2)
            labels = watershed(-distance, mask)
            
            # Filter regions
            regions = regionprops(labels)
            tumor_mask = np.zeros_like(volume)
            
            for region in regions:
                if region.area > 100:  # Minimum size threshold
                    tumor_mask[labels == region.label] = 1
            
            return tumor_mask
            
        except Exception as e:
            logger.error(f"Error in tumor detection: {str(e)}")
            return np.zeros_like(volume_data)

class DICOMViewer:
    def __init__(self):
        self.current_series = None
        self.current_slice = 0
        self.window_center = 40
        self.window_width = 400
        self.zoom_level = 1.0
        self.overlay_opacity = 0.5
        self.show_contours = True
        self.show_dose = True
        self.volume_data = None
        self.spacing = None
        self.origin = None
        self.tumor_detector = TumorDetector()
        self.tumor_mask = None
        
    def find_dicom_files(self, directory: str) -> List[str]:
        """Recursively find all DICOM files in directory"""
        dicom_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.dcm'):
                    dicom_files.append(os.path.join(root, file))
        return dicom_files
        
    def load_dicom_series(self, directory_path: str) -> bool:
        """Load a DICOM series from directory"""
        try:
            # Find all DICOM files
            dicom_files = self.find_dicom_files(directory_path)
            
            if not dicom_files:
                logger.error("No DICOM files found in directory")
                return False
            
            # Sort files by instance number
            try:
                dicom_files.sort(key=lambda x: pydicom.dcmread(x).InstanceNumber)
            except:
                # If InstanceNumber is not available, sort by filename
                dicom_files.sort()
            
            # Read first file to get metadata
            first_slice = pydicom.dcmread(dicom_files[0])
            self.metadata = self._extract_metadata(first_slice)
            
            # Get image dimensions
            rows = first_slice.Rows
            cols = first_slice.Columns
            slices = len(dicom_files)
            
            # Initialize volume array
            self.volume_data = np.zeros((slices, rows, cols), dtype=np.float32)
            
            # Get pixel spacing and slice thickness
            pixel_spacing = first_slice.PixelSpacing if hasattr(first_slice, 'PixelSpacing') else [1, 1]
            slice_thickness = first_slice.SliceThickness if hasattr(first_slice, 'SliceThickness') else 1.0
            
            # Set spacing
            self.spacing = (float(slice_thickness), float(pixel_spacing[0]), float(pixel_spacing[1]))
            
            # Get image position
            self.origin = first_slice.ImagePositionPatient if hasattr(first_slice, 'ImagePositionPatient') else [0, 0, 0]
            
            # Load all slices
            for i, file_path in enumerate(dicom_files):
                ds = pydicom.dcmread(file_path)
                self.volume_data[i, :, :] = ds.pixel_array.astype(np.float32)
            
            # Normalize pixel values
            self.volume_data = self._normalize_volume(self.volume_data)
            
            # Detect tumors
            self.tumor_mask = self.tumor_detector.detect_tumors(self.volume_data)
            
            # Create SimpleITK image
            self.current_series = sitk.GetImageFromArray(self.volume_data)
            self.current_series.SetSpacing(self.spacing)
            self.current_series.SetOrigin(self.origin)
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading DICOM series: {str(e)}")
            return False
            
    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Normalize volume data to 0-1 range"""
        min_val = np.min(volume)
        max_val = np.max(volume)
        if max_val > min_val:
            return (volume - min_val) / (max_val - min_val)
        return volume
            
    def _extract_metadata(self, ds: pydicom.Dataset) -> Dict:
        """Extract relevant metadata from DICOM dataset"""
        try:
            return {
                'patient_name': str(ds.PatientName),
                'patient_id': str(ds.PatientID),
                'study_date': str(ds.StudyDate),
                'modality': str(ds.Modality),
                'pixel_spacing': ds.PixelSpacing if hasattr(ds, 'PixelSpacing') else [1, 1],
                'slice_thickness': ds.SliceThickness if hasattr(ds, 'SliceThickness') else 1.0,
                'image_position': ds.ImagePositionPatient if hasattr(ds, 'ImagePositionPatient') else [0, 0, 0],
                'image_orientation': ds.ImageOrientationPatient if hasattr(ds, 'ImageOrientationPatient') else [1, 0, 0, 0, 1, 0]
            }
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            return {}
            
    def get_slice(self, slice_idx: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Get a specific slice from the series"""
        if self.volume_data is None:
            return None, None
            
        try:
            # Get slice data
            slice_data = self.volume_data[slice_idx]
            
            # Apply window/level
            min_val = self.window_center - self.window_width/2
            max_val = self.window_center + self.window_width/2
            slice_data = np.clip(slice_data, min_val, max_val)
            slice_data = (slice_data - min_val) / (max_val - min_val)
            
            # Get tumor mask slice if available
            tumor_slice = None
            if self.tumor_mask is not None:
                tumor_slice = self.tumor_mask[slice_idx]
            
            return slice_data, tumor_slice
        except Exception as e:
            logger.error(f"Error getting slice: {str(e)}")
            return None, None
            
    def get_3d_view(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Get 3D volume data"""
        if self.volume_data is None:
            return None, None
            
        try:
            return self.volume_data, self.tumor_mask
        except Exception as e:
            logger.error(f"Error getting 3D view: {str(e)}")
            return None, None

class TreatmentPlanner:
    def __init__(self):
        self.planner = None
        self.current_plan = None
        self.dose_distribution = None
        self.beam_weights = None
        
    def initialize_planner(self, grid_size: Tuple[int, int, int], num_beams: int = 8):
        """Initialize the treatment planner"""
        try:
            # Import QRadPlan3D
            spec = importlib.util.spec_from_file_location(
                "qradplan",
                os.path.join(current_dir, "qradplan-may27-no-quantum.py")
            )
            qradplan_module = importlib.util.module_from_spec(spec)
            sys.modules["qradplan"] = qradplan_module
            spec.loader.exec_module(qradplan_module)
            
            self.planner = qradplan_module.QRadPlan3D(
                grid_size=grid_size,
                num_beams=num_beams,
                kernel_path=os.path.join(current_dir, DEFAULT_KERNEL_PATH)
            )
            return True
        except Exception as e:
            logger.error(f"Error initializing planner: {str(e)}")
            return False
            
    def set_tumor_data(self, tumor_mask: np.ndarray):
        """Set tumor data for planning"""
        if self.planner is None:
            return False
            
        try:
            return self.planner.set_tumor_data(tumor_mask_input=tumor_mask)
        except Exception as e:
            logger.error(f"Error setting tumor data: {str(e)}")
            return False
            
    def optimize_plan(self, plan_params: Dict):
        """Optimize treatment plan"""
        if self.planner is None:
            return None
            
        try:
            # Set optimization parameters
            self.planner.optimize_oars = plan_params.get('optimize_oars', True)
            self.planner.motion_compensation = plan_params.get('motion_compensation', False)
            
            # Run optimization
            self.beam_weights = self.planner.optimize_beams()
            
            # Calculate dose distribution
            self.dose_distribution = self.planner.calculate_dose(self.beam_weights)
            
            # Calculate metrics
            metrics = self.planner.calculate_plan_metrics(self.beam_weights)
            
            # Store complete plan
            self.current_plan = {
                'beam_weights': self.beam_weights,
                'dose_distribution': self.dose_distribution,
                'metrics': metrics
            }
            
            return self.current_plan
        except Exception as e:
            logger.error(f"Error optimizing plan: {str(e)}")
            return None

def process_zip_file(zip_file) -> Optional[str]:
    """Process uploaded ZIP file containing DICOM series"""
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Extract ZIP contents
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        return temp_dir
    except Exception as e:
        logger.error(f"Error processing ZIP file: {str(e)}")
        return None

def main():
    # Initialize session state
    if 'viewer' not in st.session_state:
        st.session_state.viewer = DICOMViewer()
    if 'planner' not in st.session_state:
        st.session_state.planner = TreatmentPlanner()
    if 'current_series' not in st.session_state:
        st.session_state.current_series = None
    if 'current_slice' not in st.session_state:
        st.session_state.current_slice = 0
    if 'window_center' not in st.session_state:
        st.session_state.window_center = 40
    if 'window_width' not in st.session_state:
        st.session_state.window_width = 400
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = None
        
    # Main interface
    st.title("StarViewer RT - Medical Imaging & Treatment Planning")
    
    # Sidebar
    with st.sidebar:
        st.header("File Operations")
        
        # DICOM series upload
        uploaded_file = st.file_uploader(
            "Upload DICOM Series (ZIP or DICOM files)",
            type=['zip', 'dcm'],
            accept_multiple_files=True
        )
        
        if uploaded_file:
            try:
                # Clean up previous temporary directory if it exists
                if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
                    shutil.rmtree(st.session_state.temp_dir)
                
                # Create new temporary directory
                temp_dir = tempfile.mkdtemp()
                st.session_state.temp_dir = temp_dir
                
                # Process uploaded files
                if isinstance(uploaded_file, list):
                    # Multiple DICOM files
                    progress_bar = st.progress(0)
                    total_files = len(uploaded_file)
                    
                    for i, file in enumerate(uploaded_file):
                        with open(os.path.join(temp_dir, file.name), 'wb') as f:
                            f.write(file.getvalue())
                        progress_bar.progress((i + 1) / total_files)
                    
                    dicom_dir = temp_dir
                else:
                    # Single ZIP file
                    with st.spinner("Processing ZIP file..."):
                        dicom_dir = process_zip_file(uploaded_file)
                
                if dicom_dir:
                    # Load DICOM series
                    with st.spinner("Loading DICOM series..."):
                        if st.session_state.viewer.load_dicom_series(dicom_dir):
                            st.session_state.current_series = st.session_state.viewer.current_series
                            
                            # Display metadata
                            if hasattr(st.session_state.viewer, 'metadata'):
                                metadata = st.session_state.viewer.metadata
                                st.success("DICOM series loaded successfully")
                                st.markdown("### Patient Information")
                                st.markdown(f"**Name:** {metadata.get('patient_name', 'N/A')}")
                                st.markdown(f"**ID:** {metadata.get('patient_id', 'N/A')}")
                                st.markdown(f"**Study Date:** {metadata.get('study_date', 'N/A')}")
                                st.markdown(f"**Modality:** {metadata.get('modality', 'N/A')}")
                        else:
                            st.error("Failed to load DICOM series")
                else:
                    st.error("No valid DICOM files found in the uploaded content")
                    
            except Exception as e:
                st.error(f"Error processing uploaded files: {str(e)}")
                logger.error(f"Error processing uploaded files: {str(e)}")
            finally:
                # Clean up temporary files
                gc.collect()
        
        # Display controls
        st.header("Display Controls")
        st.session_state.window_center = st.slider(
            "Window Center",
            min_value=-1000,
            max_value=3000,
            value=st.session_state.window_center
        )
        st.session_state.window_width = st.slider(
            "Window Width",
            min_value=1,
            max_value=4000,
            value=st.session_state.window_width
        )
        
        # Treatment planning controls
        st.header("Treatment Planning")
        if st.button("Initialize Planner"):
            if st.session_state.current_series is not None:
                grid_size = st.session_state.current_series.GetSize()
                if st.session_state.planner.initialize_planner(grid_size):
                    st.success("Treatment planner initialized")
                else:
                    st.error("Failed to initialize treatment planner")
            else:
                st.warning("Please load a DICOM series first")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs([
        "2D View",
        "3D View",
        "Treatment Planning",
        "Results & Analysis"
    ])
    
    # Tab 1: 2D View
    with tab1:
        if st.session_state.current_series is not None:
            # Slice navigation
            num_slices = st.session_state.current_series.GetSize()[2]
            st.session_state.current_slice = st.slider(
                "Slice",
                0,
                num_slices - 1,
                st.session_state.current_slice
            )
            
            # Get current slice
            slice_data, tumor_slice = st.session_state.viewer.get_slice(st.session_state.current_slice)
            
            if slice_data is not None:
                # Create figure with subplots
                fig = make_subplots(rows=1, cols=2)
                
                # Original slice
                fig.add_trace(
                    go.Heatmap(
                        z=slice_data,
                        colorscale='Gray',
                        showscale=False
                    ),
                    row=1, col=1
                )
                
                # Add tumor overlay if available
                if tumor_slice is not None:
                    fig.add_trace(
                        go.Heatmap(
                            z=tumor_slice,
                            colorscale='Reds',
                            opacity=0.3,
                            showscale=False
                        ),
                        row=1, col=1
                    )
                
                # Update layout
                fig.update_layout(
                    title="2D View",
                    height=600,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please upload a DICOM series to begin")
    
    # Tab 2: 3D View
    with tab2:
        if st.session_state.current_series is not None:
            # Get 3D volume data
            volume_data, tumor_mask = st.session_state.viewer.get_3d_view()
            
            if volume_data is not None:
                # Create 3D visualization
                fig = go.Figure()
                
                # Add volume
                fig.add_trace(go.Volume(
                    x=np.arange(volume_data.shape[2]),
                    y=np.arange(volume_data.shape[1]),
                    z=np.arange(volume_data.shape[0]),
                    value=volume_data.flatten(),
                    isomin=0.1,
                    isomax=0.9,
                    opacity=0.1,
                    surface_count=20,
                    colorscale='Gray',
                    showscale=False
                ))
                
                # Add tumor mask if available
                if tumor_mask is not None:
                    fig.add_trace(go.Volume(
                        x=np.arange(tumor_mask.shape[2]),
                        y=np.arange(tumor_mask.shape[1]),
                        z=np.arange(tumor_mask.shape[0]),
                        value=tumor_mask.flatten(),
                        isomin=0.5,
                        isomax=1.0,
                        opacity=0.3,
                        surface_count=20,
                        colorscale='Reds',
                        showscale=False
                    ))
                
                # Update layout
                fig.update_layout(
                    title="3D View",
                    scene=dict(
                        xaxis_title="X",
                        yaxis_title="Y",
                        zaxis_title="Z",
                        aspectmode='data'
                    ),
                    height=800,
                    margin=dict(l=0, r=0, b=0, t=30)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add 3D view controls
                st.sidebar.header("3D View Controls")
                opacity = st.sidebar.slider("Volume Opacity", 0.0, 1.0, 0.1)
                surface_count = st.sidebar.slider("Surface Count", 1, 50, 20)
                isomin = st.sidebar.slider("Minimum Intensity", 0.0, 1.0, 0.1)
                isomax = st.sidebar.slider("Maximum Intensity", 0.0, 1.0, 0.9)
                
                # Update visualization with new parameters
                fig.update_traces(
                    opacity=opacity,
                    surface_count=surface_count,
                    isomin=isomin,
                    isomax=isomax
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please upload a DICOM series to begin")
    
    # Tab 3: Treatment Planning
    with tab3:
        if st.session_state.current_series is not None:
            st.header("Treatment Plan Parameters")
            
            with st.form("planning_parameters"):
                cols = st.columns([1, 1, 1])
                
                # Column 1: Dose settings
                with cols[0]:
                    st.markdown("##### Radiation Settings")
                    prescribed_dose = st.number_input(
                        "Total Dose (Gy)",
                        min_value=1.0,
                        max_value=80.0,
                        value=60.0
                    )
                    
                    num_fractions = st.number_input(
                        "Number of Fractions",
                        min_value=1,
                        max_value=35,
                        value=30
                    )
                    
                    beam_energy = st.selectbox(
                        "Beam Energy (MV)",
                        options=[6, 10, 15],
                        index=0
                    )
                
                # Column 2: Planning strategy
                with cols[1]:
                    st.markdown("##### Planning Strategy")
                    motion_comp = st.checkbox(
                        "Motion Compensation",
                        help="Enable respiratory motion compensation"
                    )
                    
                    optimize_oars = st.checkbox(
                        "Optimize OARs",
                        value=True,
                        help="Consider organs at risk in optimization"
                    )
                    
                    num_beams = st.slider(
                        "Number of Beams",
                        min_value=4,
                        max_value=12,
                        value=8
                    )
                
                # Column 3: Optimization weights
                with cols[2]:
                    st.markdown("##### Optimization Weights")
                    tumor_weight = st.slider(
                        "Tumor Coverage",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.7
                    )
                    
                    healthy_tissue_weight = st.slider(
                        "Healthy Tissue Sparing",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.3
                    )
                
                # Submit button
                plan_submitted = st.form_submit_button("Generate Plan")
                
                if plan_submitted:
                    with st.spinner("Generating treatment plan..."):
                        # Set tumor data
                        if st.session_state.viewer.tumor_mask is not None:
                            st.session_state.planner.set_tumor_data(st.session_state.viewer.tumor_mask)
                        
                        # Prepare plan parameters
                        plan_params = {
                            'total_dose': prescribed_dose,
                            'num_fractions': num_fractions,
                            'beam_energy': beam_energy,
                            'num_beams': num_beams,
                            'motion_compensation': motion_comp,
                            'optimize_oars': optimize_oars,
                            'tumor_weight': tumor_weight,
                            'healthy_tissue_weight': healthy_tissue_weight
                        }
                        
                        # Generate plan
                        plan_results = st.session_state.planner.optimize_plan(plan_params)
                        
                        if plan_results:
                            st.success("Treatment plan generated successfully")
                            
                            # Store results in session state
                            st.session_state.dose_distribution = plan_results['dose_distribution']
                            st.session_state.plan_metrics = plan_results['metrics']
                        else:
                            st.error("Failed to generate treatment plan")
        else:
            st.info("Please upload a DICOM series to begin")
    
    # Tab 4: Results & Analysis
    with tab4:
        if hasattr(st.session_state, 'plan_metrics'):
            st.header("Treatment Plan Analysis")
            
            # Display metrics in columns
            metric_cols = st.columns(3)
            
            # Column 1: Tumor metrics
            with metric_cols[0]:
                st.markdown("##### 🎯 Tumor Coverage")
                if 'tumor' in st.session_state.plan_metrics:
                    tumor = st.session_state.plan_metrics['tumor']
                    if 'V95' in tumor:
                        st.metric(
                            "V95 Coverage",
                            f"{tumor['V95']:.1f}%",
                            delta=f"{tumor['V95']-95:.1f}%"
                        )
                    if 'TCP' in tumor:
                        st.metric("TCP", f"{tumor['TCP']:.1f}%")
            
            # Column 2: OAR metrics
            with metric_cols[1]:
                st.markdown("##### 🫁 Organs at Risk")
                if 'lung' in st.session_state.plan_metrics:
                    lung = st.session_state.plan_metrics['lung']
                    if 'mean_dose' in lung:
                        st.metric(
                            "Mean Lung Dose",
                            f"{lung['mean_dose']:.1f} Gy",
                            delta=f"{20-lung['mean_dose']:.1f} Gy",
                            delta_color="inverse"
                        )
                    if 'V20' in lung:
                        st.metric(
                            "Lung V20",
                            f"{lung['V20']:.1f}%",
                            delta=f"{35-lung['V20']:.1f}%",
                            delta_color="inverse"
                        )
            
            # Column 3: Treatment course
            with metric_cols[2]:
                st.markdown("##### 📅 Treatment Course")
                if 'course' in st.session_state.plan_metrics:
                    course = st.session_state.plan_metrics['course']
                    if 'total_dose' in course:
                        st.metric("Total Dose", f"{course['total_dose']:.1f} Gy")
                    if 'num_fractions' in course:
                        st.metric("Fractions", str(course['num_fractions']))
            
            # DVH plot
            if hasattr(st.session_state.planner, 'plot_dvh'):
                st.markdown("### Dose Volume Histogram")
                fig, ax = plt.subplots(figsize=(10, 6))
                st.session_state.planner.planner.plot_dvh(ax=ax)
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("Please generate a treatment plan first")

if __name__ == "__main__":
    main() 