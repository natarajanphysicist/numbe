import sys
import os
import logging
import numpy as np 

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QAction, 
    QFileDialog, QMessageBox, QStatusBar, QLabel, QApplication, QTextEdit,
    QSlider, QLineEdit, QFormLayout, QTabWidget, QPushButton, QCheckBox # Added QCheckBox
)
from PyQt5.QtCore import Qt

from QRadPlannerApp.backend.data_manager import DataManager
from QRadPlannerApp.ui.dicom_viewer_2d import DicomViewer2DWidget
from QRadPlannerApp.ui.dicom_viewer_3d import DicomViewer3DWidget # Corrected: Ensure single import
from QRadPlannerApp.ui.dvh_plot_widget import DvhPlotWidget # Ensure single import

logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    def __init__(self, data_manager: DataManager):
        super().__init__()
        self.data_manager = data_manager
        
        self.setWindowTitle("QRadPlanner - Radiotherapy Planning Tool")
        self.setGeometry(100, 100, 1800, 1000)  # Adjusted size for potentially larger 3D view

        self._create_menu_bar()
        self._create_status_bar()
        self._create_viewer_controls() 
        self._init_central_widget() # This will now setup tabs
        
        logger.info("MainWindow initialized.")
        self._update_displayed_slice() 
        self._update_3d_viewer() # Initial call for 3D view

    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        
        # File Menu
        file_menu = menu_bar.addMenu("&File")
        
        open_folder_action = QAction("Open DICOM Folder...", self)
        open_folder_action.setStatusTip("Open a folder containing DICOM files")
        open_folder_action.triggered.connect(self._open_dicom_folder)
        file_menu.addAction(open_folder_action)
        
        open_zip_action = QAction("Open DICOM ZIP...", self)
        open_zip_action.setStatusTip("Open a ZIP archive containing DICOM files")
        open_zip_action.triggered.connect(self._open_dicom_zip)
        file_menu.addAction(open_zip_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("&Exit", self)
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(self.close) # QMainWindow.close()
        file_menu.addAction(exit_action)

    def _create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready", 3000) # Message, timeout in ms

    def _init_central_widget(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        
        layout = QVBoxLayout(self.central_widget)
        
        # Placeholder label
        
        
        # Create Tab Widget
        self.tabs = QTabWidget()
        
        # --- 2D View Tab ---
        self.view_2d_tab_content = QWidget()
        view_2d_main_layout = QHBoxLayout(self.view_2d_tab_content)
        
        self.viewer_2d = DicomViewer2DWidget()
        view_2d_main_layout.addWidget(self.viewer_2d, 3) # Viewer takes 3/4 of space
        view_2d_main_layout.addWidget(self.viewer_controls_widget, 1) # Controls take 1/4 of space
        self.view_2d_tab_content.setLayout(view_2d_main_layout)
        
        self.tabs.addTab(self.view_2d_tab_content, "2D View")

        # --- 3D View Tab ---
        self.viewer_3d = DicomViewer3DWidget()
        self.tabs.addTab(self.viewer_3d, "3D View")

        # --- Treatment Planning Tab ---
        self.planning_tab_widget = self._create_planning_tab()
        self.tabs.addTab(self.planning_tab_widget, "Treatment Planning")
        
        # --- Results & Analysis Tab ---
        self.results_tab_widget = self._create_results_tab()
        self.tabs.addTab(self.results_tab_widget, "Results & Analysis")
        
        self.setCentralWidget(self.tabs)

    def _create_viewer_controls(self):
        self.viewer_controls_widget = QWidget()
        controls_layout = QVBoxLayout() # Main layout for controls

        # Slice navigation
        slice_nav_layout = QFormLayout()
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0) # Will be updated on data load
        self.slice_slider.setValue(0)
        self.slice_slider.valueChanged.connect(self._update_displayed_slice)
        
        self.current_slice_label = QLabel("Slice: 0 / 0")
        slice_nav_layout.addRow("Navigate Slices:", self.slice_slider)
        slice_nav_layout.addRow(self.current_slice_label)
        controls_layout.addLayout(slice_nav_layout)

        controls_layout.addStretch(1) # Spacer

        # Window/Level controls
        wl_layout = QFormLayout()
        self.wc_input = QLineEdit("40") # Default WC
        self.ww_input = QLineEdit("400") # Default WW
        
        self.wc_input.editingFinished.connect(self._update_displayed_slice)
        self.ww_input.editingFinished.connect(self._update_displayed_slice)
        # Could also use valueChanged signal with debounce if performance is an issue

        wl_layout.addRow("Window Center (WC):", self.wc_input)
        wl_layout.addRow("Window Width (WW):", self.ww_input)
        controls_layout.addLayout(wl_layout)
        
        # Dose Overlay Checkbox
        self.show_dose_checkbox = QCheckBox("Show Dose Overlay")
        self.show_dose_checkbox.setChecked(True) # Default to show dose if available
        self.show_dose_checkbox.stateChanged.connect(self._update_displayed_slice)
        self.show_dose_checkbox.setStatusTip("Toggle visibility of dose overlay on 2D view.")
        controls_layout.addWidget(self.show_dose_checkbox)

        controls_layout.addStretch(1) 

        # Tumor Detection Button
        self.detect_tumors_button = QPushButton("Detect Tumors")
        self.detect_tumors_button.clicked.connect(self._run_tumor_detection)
        self.detect_tumors_button.setStatusTip("Run automatic tumor detection on the loaded volume.")
        controls_layout.addWidget(self.detect_tumors_button)
        
        controls_layout.addStretch(5) 

        self.viewer_controls_widget.setLayout(controls_layout)

    def _run_tumor_detection(self):
        logger.info("Attempting to run tumor detection...")
        if self.data_manager.volume_data is None:
            QMessageBox.warning(self, "No Data", "Please load a DICOM volume before running tumor detection.")
            logger.warning("Tumor detection skipped: No volume data.")
            return

        try:
            self.status_bar.showMessage("Running tumor detection...", 3000) # Temporary message
            QApplication.processEvents()
            # The detect_tumors method is part of TumorDetector, accessed via DataManager
            # And it should store the mask back into data_manager.tumor_mask
            # DataManager's load_dicom_from_folder/zip already calls this.
            # This button provides a way to re-run or run if it wasn't run initially.
            
            # For this UI action, we assume DataManager has a method that can trigger re-detection
            # or that TumorDetector is accessible. Let's assume DataManager handles it.
            # A direct call as below implies TumorDetector is part of DataManager and its detect_tumors
            # method takes volume and returns a mask.
            
            # If DataManager.tumor_detector is public:
            if hasattr(self.data_manager, 'tumor_detector') and self.data_manager.tumor_detector is not None:
                logger.debug(f"Using DataManager's tumor_detector. Original mask sum: {np.sum(self.data_manager.tumor_mask) if self.data_manager.tumor_mask is not None else 'None'}")
                # The TumorDetector in DataManager is already instantiated.
                # Its detect_tumors method takes volume_data and returns the mask.
                # We need to ensure DataManager updates its own self.tumor_mask.
                # A wrapper method in DataManager would be cleaner: e.g., data_manager.run_tumor_detection()
                
                # For now, let's call the detector directly and update DM's mask
                new_tumor_mask = self.data_manager.tumor_detector.detect_tumors(self.data_manager.volume_data)
                if new_tumor_mask is not None:
                    self.data_manager.tumor_mask = new_tumor_mask # Update DataManager's mask
                    logger.info(f"Tumor detection complete. New mask sum: {np.sum(new_tumor_mask)}")
                    QMessageBox.information(self, "Tumor Detection", "Tumor detection complete. Mask updated.")
                else:
                    logger.error("Tumor detection returned None.")
                    QMessageBox.warning(self, "Tumor Detection", "Tumor detection failed to produce a mask.")

            else: # Fallback if direct access isn't set up as assumed
                 logger.error("DataManager does not have an accessible 'tumor_detector' or it's None.")
                 QMessageBox.critical(self, "Error", "Tumor detector not available via DataManager.")
                 return

            self._update_displayed_slice()
            self._update_3d_viewer()
            self.status_bar.showMessage("Tumor detection finished.", 3000)

        except Exception as e:
            logger.error(f"Error during tumor detection: {e}", exc_info=True)
            QMessageBox.critical(self, "Tumor Detection Error", f"An error occurred: {e}")
            self.status_bar.showMessage("Tumor detection error.", 3000)

    def _update_displayed_slice(self):
        if self.data_manager.volume_data is None:
            self.viewer_2d.clear_view()
            self.current_slice_label.setText("Slice: N/A")
            self.slice_slider.setEnabled(False)
            return

        self.slice_slider.setEnabled(True)
        slice_idx = self.slice_slider.value()
        
        # Update slice label
        num_slices = self.data_manager.volume_data.shape[0] # Assuming (slices, rows, cols)
        self.current_slice_label.setText(f"Slice: {slice_idx + 1} / {num_slices}")

        # Get WC/WW values
        try:
            wc = float(self.wc_input.text())
            ww = float(self.ww_input.text())
            if ww <= 0: # Window width must be positive
                logger.warning("Window width must be positive. Using default WW=400.")
                ww = 400.0
                self.ww_input.setText(str(ww))
        except ValueError:
            logger.error("Invalid WC/WW input. Using defaults.")
            wc, ww = 40, 400 # Fallback to defaults
            self.wc_input.setText(str(wc))
            self.ww_input.setText(str(ww))

        # Get raw slice data - assuming DataManager stores volume as (slices, rows, cols)
        # This needs to be consistent with how DataManager loads/stores the volume.
        # If DataManager stores as (cols, rows, slices) i.e. (x,y,z), then access is different.
        # Based on DataManager's load_dicom_series_from_directory, volume_data is (num_slices, rows, cols)
        raw_slice = self.data_manager.volume_data[slice_idx, :, :]

        # Apply window/level
        lower_bound = wc - (ww / 2.0)
        upper_bound = wc + (ww / 2.0)
        
        display_slice = np.clip(raw_slice, lower_bound, upper_bound)
        
        # Normalize for display [0,1]
        # Add epsilon to ww to avoid division by zero if ww is very small (though we check ww > 0)
        display_slice_normalized = (display_slice - lower_bound) / (ww + 1e-9)
        display_slice_normalized = np.clip(display_slice_normalized, 0.0, 1.0)


        # Get tumor mask slice
        tumor_slice = None
        if self.data_manager.tumor_mask is not None:
            if slice_idx < self.data_manager.tumor_mask.shape[0]: # Check bounds
                tumor_slice = self.data_manager.tumor_mask[slice_idx, :, :]
            else:
                logger.warning(f"Slice index {slice_idx} out of bounds for tumor mask shape {self.data_manager.tumor_mask.shape}")
        
        
        dose_slice_to_display = None
        if self.show_dose_checkbox.isChecked() and self.data_manager.dose_distribution is not None:
            # Assuming dose_distribution has same (slices, rows, cols) as volume_data
            if slice_idx < self.data_manager.dose_distribution.shape[0]:
                dose_slice_to_display = self.data_manager.dose_distribution[slice_idx, :, :]
            else:
                logger.warning(f"Slice index {slice_idx} out of bounds for dose distribution shape {self.data_manager.dose_distribution.shape}")
        
        #self.viewer_2d.update_slice(display_slice_normalized, tumor_slice, dose_slice_to_display)
        self.viewer_2d.update_slice(slice_idx, display_slice_normalized, tumor_slice, dose_slice_to_display) # <-- CORRECTED CALL

    def _open_dicom_folder(self):
        # Use the user's home directory or last opened directory as a starting point
        start_dir = os.path.expanduser("~") 
        folder_path = QFileDialog.getExistingDirectory(
            self, 
            "Open DICOM Folder",
            start_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if folder_path:
            if hasattr(self, 'dvh_plot_widget') and self.dvh_plot_widget: # Added
                self.dvh_plot_widget.clear_plot() # Added
            logger.info(f"Selected DICOM folder: {folder_path}")
            self.status_bar.showMessage(f"Loading DICOM folder: {folder_path}...")
            QApplication.processEvents() 

            load_success = self.data_manager.load_dicom_from_folder(folder_path)
            
            if load_success and self.data_manager.volume_data is not None:
                patient_name = self.data_manager.patient_metadata.get('PatientName', 'Unknown Patient')
                # Assuming volume_data is (slices, rows, cols) from DataManager
                num_slices = self.data_manager.volume_data.shape[0]
                
                self.slice_slider.setMaximum(num_slices - 1 if num_slices > 0 else 0)
                middle_slice = (num_slices - 1) // 2 if num_slices > 0 else 0
                self.slice_slider.setValue(middle_slice)
                
                # Set default WC/WW or derive from data (e.g. CT HU range)
                # For now, using fixed defaults.
                self.wc_input.setText("40") 
                self.ww_input.setText("400")

                self._update_displayed_slice() 
                self._update_3d_viewer() # Update 3D viewer as well

                status_msg = f"Loaded: {patient_name} - {num_slices} slices. Volume shape: {self.data_manager.volume_data.shape}"
                self.status_bar.showMessage(status_msg, 5000)
                logger.info(status_msg)
            else:
                self.viewer_2d.clear_view()
                self.viewer_3d.clear_view() 
                if hasattr(self, 'dvh_plot_widget'): self.dvh_plot_widget.clear_plot() # Added
                self.slice_slider.setMaximum(0)
                self.current_slice_label.setText("Slice: N/A")
                error_msg = f"Failed to load DICOM series from folder: {folder_path}. Check logs for details."
                self.status_bar.showMessage("Error loading DICOM folder.", 5000)
                logger.error(error_msg)
                QMessageBox.critical(self, "Load Error", error_msg)

    def _open_dicom_zip(self):
        start_dir = os.path.expanduser("~")
        zip_file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open DICOM ZIP Archive",
            start_dir,
            "ZIP Files (*.zip);;All Files (*)"
        )
        
        if zip_file_path:
            if hasattr(self, 'dvh_plot_widget') and self.dvh_plot_widget: # Added
                self.dvh_plot_widget.clear_plot() # Added
            logger.info(f"Selected DICOM ZIP file: {zip_file_path}")
            self.status_bar.showMessage(f"Loading DICOM ZIP: {zip_file_path}...")
            QApplication.processEvents()

            load_success = self.data_manager.load_dicom_from_zip(zip_file_path)
            
            if load_success and self.data_manager.volume_data is not None:
                patient_name = self.data_manager.patient_metadata.get('PatientName', 'Unknown Patient')
                num_slices = self.data_manager.volume_data.shape[0] # Assuming (slices, rows, cols)

                self.slice_slider.setMaximum(num_slices - 1 if num_slices > 0 else 0)
                middle_slice = (num_slices - 1) // 2 if num_slices > 0 else 0
                self.slice_slider.setValue(middle_slice)

                self.wc_input.setText("40")
                self.ww_input.setText("400")
                
                self._update_displayed_slice()
                self._update_3d_viewer() # Update 3D viewer as well

                status_msg = f"Loaded: {patient_name} - {num_slices} slices. Volume shape: {self.data_manager.volume_data.shape}"
                self.status_bar.showMessage(status_msg, 5000)
                logger.info(status_msg)
            else:
                self.viewer_2d.clear_view()
                self.viewer_3d.clear_view() 
                if hasattr(self, 'dvh_plot_widget'): self.dvh_plot_widget.clear_plot() # Added
                self.slice_slider.setMaximum(0)
                self.current_slice_label.setText("Slice: N/A")
                error_msg = f"Failed to load DICOM series from ZIP: {zip_file_path}. Check logs for details."
                self.status_bar.showMessage("Error loading DICOM ZIP.", 5000)
                logger.error(error_msg)
                QMessageBox.critical(self, "Load Error", error_msg)

    # --- Corrected part of QRadPlannerApp/ui/main_window.py ---
    def _update_3d_viewer(self):
        logger.debug("Attempting to update 3D viewer.")
        if self.data_manager.volume_data is None or self.data_manager.image_properties is None:
            self.viewer_3d.clear_view()
            logger.debug("3D Viewer cleared as no volume data/properties are available.")
            return

        try:
            # Call update_volume on DicomViewer3DWidget WITHOUT dose_volume_full
            self.viewer_3d.update_volume(
                volume_data_full=self.data_manager.volume_data,
                image_properties=self.data_manager.image_properties,
                tumor_mask_full=self.data_manager.tumor_mask
                # dose_volume_full=self.data_manager.dose_distribution # <-- REMOVE THIS LINE
            )

            # If you intend to show dose isosurfaces immediately after loading data,
            # and assuming you have a default list of isovalues or UI controls for them,
            # you would call _update_dose_isosurfaces here.
            # For now, this is not implemented based on the README (UI controls missing).
            # Example of how it might be called if controls existed:
            # if self.data_manager.dose_distribution is not None:
            #     # Get isovalues from UI or use defaults
            #     isovalues = [10.0, 30.0, 50.0] # Example
            #     self.viewer_3d._update_dose_isosurfaces(
            #        dose_volume_full=self.data_manager.dose_distribution,
            #        image_properties=self.data_manager.image_properties,
            #        isovalues_list=isovalues
            #     )
            # else:
            #      self.viewer_3d._clear_dose_isosurfaces() # Clear if no dose

            logger.info("3D Viewer updated with new data.")
        except Exception as e:
            logger.error(f"Error updating 3D viewer: {e}", exc_info=True)
            QMessageBox.warning(self, "3D View Error", f"Could not update 3D view: {e}")
            if hasattr(self, 'viewer_3d'): self.viewer_3d.clear_view()
    

    def _create_planning_tab(self) -> QWidget:
        planning_widget = QWidget()
        main_layout = QVBoxLayout(planning_widget)
        
        form_layout = QFormLayout()
        
        self.num_beams_input = QLineEdit("8") # Changed to QLineEdit for now, can be QSpinBox
        form_layout.addRow("Number of Beams:", self.num_beams_input)
        
        self.num_fractions_input = QLineEdit("30") # Changed to QLineEdit
        form_layout.addRow("Number of Fractions:", self.num_fractions_input)
        
        self.target_dose_input = QLineEdit("60.0")
        form_layout.addRow("Target Prescription Dose (Gy):", self.target_dose_input)
        
        main_layout.addLayout(form_layout)
        
        self.init_planner_button = QPushButton("Initialize Planner")
        self.init_planner_button.clicked.connect(self._initialize_planner_from_ui)
        main_layout.addWidget(self.init_planner_button)
        
        self.run_optimization_button = QPushButton("Run Beam Optimization")
        self.run_optimization_button.clicked.connect(self._run_optimization_from_ui)
        main_layout.addWidget(self.run_optimization_button)
        
        self.calculate_dose_button = QPushButton("Calculate Dose Distribution")
        self.calculate_dose_button.clicked.connect(self._calculate_dose_from_ui)
        main_layout.addWidget(self.calculate_dose_button)
        
        self.planning_status_text = QTextEdit()
        self.planning_status_text.setReadOnly(True)
        main_layout.addWidget(self.planning_status_text)
        
        planning_widget.setLayout(main_layout)
        return planning_widget

    def _initialize_planner_from_ui(self):
        if self.data_manager.volume_data is None:
            QMessageBox.warning(self, "No Data", "Load DICOM data before initializing planner.")
            return
        try:
            num_beams = int(self.num_beams_input.text())
            if not (4 <= num_beams <= 12):
                QMessageBox.warning(self, "Input Error", "Number of beams must be between 4 and 12.")
                return

            # Grid size derived from data_manager.image_properties as in DataManager.initialize_planner
            init_success = self.data_manager.initialize_planner(num_beams_override=num_beams)
            
            if init_success:
                msg = "Planner initialized successfully."
                set_mask_success = self.data_manager.set_planner_tumor_mask()
                if set_mask_success:
                    msg += " Tumor mask set in planner."
                else:
                    msg += " Failed to set tumor mask in planner (mask may be missing or invalid)."
                self.planning_status_text.append(msg)
                logger.info(msg)
            else:
                self.planning_status_text.append("Planner initialization failed. Check logs.")
                logger.error("Planner initialization failed from UI.")
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Invalid number of beams.")
        except Exception as e:
            logger.error(f"Error initializing planner from UI: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Planner initialization error: {e}")

    def _run_optimization_from_ui(self):
        if not self.data_manager.planner:
            QMessageBox.warning(self, "Planner Not Ready", "Initialize planner first.")
            return
        try:
            self.planning_status_text.append("Running beam optimization...")
            QApplication.processEvents()
            opt_success = self.data_manager.run_beam_optimization()
            if opt_success:
                weights_str = np.array2string(self.data_manager.plan_results.get('beam_weights', np.array([])), precision=3)
                msg = f"Beam optimization complete. Weights: {weights_str}"
                self.planning_status_text.append(msg)
                logger.info(msg)
            else:
                self.planning_status_text.append("Beam optimization failed. Check logs.")
                logger.error("Beam optimization failed from UI.")
        except Exception as e:
            logger.error(f"Error running optimization from UI: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Optimization error: {e}")

    def _calculate_dose_from_ui(self):
        if not self.data_manager.plan_results.get('beam_weights') is not None:
            QMessageBox.warning(self, "Prerequisite Missing", "Run beam optimization first to get beam weights.")
            return
        try:
            self.planning_status_text.append("Calculating dose distribution...")
            QApplication.processEvents()
            dose_success = self.data_manager.calculate_dose_distribution()
            if dose_success:
                msg = "Dose calculation complete."
                if self.data_manager.dose_distribution is not None:
                    msg += f" Dose Min: {self.data_manager.dose_distribution.min():.2f}, Max: {self.data_manager.dose_distribution.max():.2f} Gy"
                self.planning_status_text.append(msg)
                logger.info(msg)
                self._update_displayed_slice() # Refresh views to show dose potentially
                self._update_3d_viewer()
            else:
                self.planning_status_text.append("Dose calculation failed. Check logs.")
                logger.error("Dose calculation failed from UI.")
        except Exception as e:
            logger.error(f"Error calculating dose from UI: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Dose calculation error: {e}")
            
    def _create_results_tab(self) -> QWidget:
        results_widget = QWidget()
        layout = QVBoxLayout(results_widget)
        
        self.calculate_metrics_button = QPushButton("Calculate Metrics & DVH")
        self.calculate_metrics_button.clicked.connect(self._calculate_metrics_dvh_from_ui)
        layout.addWidget(self.calculate_metrics_button)
        
        self.metrics_display_text = QTextEdit()
        self.metrics_display_text.setReadOnly(True)
        layout.addWidget(self.metrics_display_text)
        
        # DVH Plot Widget
        self.dvh_plot_widget = DvhPlotWidget()
        layout.addWidget(self.dvh_plot_widget) # Add DVH plot widget
        
        results_widget.setLayout(layout)
        return results_widget

    def _calculate_metrics_dvh_from_ui(self):
        if self.data_manager.dose_distribution is None or not self.data_manager.planner:
            QMessageBox.warning(self, "Prerequisites Missing", "Ensure dose distribution is calculated and planner is initialized.")
            return

        try:
            target_dose_str = self.target_dose_input.text()
            num_fractions_str = self.num_fractions_input.text()
            
            target_dose = float(target_dose_str)
            num_fractions = int(num_fractions_str)

            if num_fractions <= 0:
                QMessageBox.warning(self, "Input Error", "Number of fractions must be positive.")
                return

            self.planning_status_text.append(f"Calculating metrics & DVH for {target_dose}Gy in {num_fractions} fractions...")
            QApplication.processEvents()

            metrics_success = self.data_manager.get_plan_metrics(
                target_prescription_dose=target_dose,
                num_fractions_for_radiobio=num_fractions
            )
            dvh_success = self.data_manager.get_dvh_data()

            if metrics_success:
                metrics = self.data_manager.plan_results.get('metrics', {})
                # Pretty print metrics (basic example)
                metrics_str = "Plan Metrics:\n"
                if 'tumor' in metrics:
                    metrics_str += f"  Tumor:\n"
                    for k, v in metrics['tumor'].items(): metrics_str += f"    {k}: {v:.3f if isinstance(v, float) else v}\n"
                if 'oars' in metrics:
                    metrics_str += f"  OARs:\n"
                    for oar, oar_metrics in metrics['oars'].items():
                        metrics_str += f"    {oar}:\n"
                        for k, v in oar_metrics.items(): metrics_str += f"      {k}: {v:.3f if isinstance(v, float) else v}\n"
                self.metrics_display_text.setText(metrics_str)
                self.planning_status_text.append("Metrics calculated.")
            else:
                self.metrics_display_text.setText("Failed to calculate metrics.")
                self.planning_status_text.append("Metrics calculation failed.")
                # Also clear DVH plot if metrics calculation failed, as DVH might be stale
                if hasattr(self, 'dvh_plot_widget'): self.dvh_plot_widget.clear_plot()


            if dvh_success:
                if self.data_manager.plan_results and 'dvh_data' in self.data_manager.plan_results:
                    self.dvh_plot_widget.plot_dvh(self.data_manager.plan_results['dvh_data'])
                    self.planning_status_text.append("DVH plot updated with new data.")
                    logger.info(f"DVH Data plotted for ROIs: {list(self.data_manager.plan_results['dvh_data'].keys())}")
                else:
                    self.dvh_plot_widget.clear_plot()
                    self.planning_status_text.append("DVH data was not available to plot even if generation step reported success.")
            else:
                self.planning_status_text.append("DVH data generation failed.")
                if hasattr(self, 'dvh_plot_widget'): self.dvh_plot_widget.clear_plot()

        except ValueError:
            QMessageBox.warning(self, "Input Error", "Invalid target dose or number of fractions.")
        except Exception as e:
            logger.error(f"Error calculating metrics/DVH from UI: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Metrics/DVH calculation error: {e}")

if __name__ == '__main__':
    # This is for testing the MainWindow directly if needed
    # Normally, it would be run from the main QRadPlannerApp/main.py
    app = QApplication(sys.argv)
    
    # Create a dummy DataManager for testing
    # In real use, this comes from main.py
    class DummyDataManager:
        def __init__(self):
            self.patient_metadata = {}
            self.volume_data = None
        def load_dicom_from_folder(self, folder_path):
            print(f"Dummy load from folder: {folder_path}")
            # self.volume_data = np.random.rand(10, 100, 100) # Example data
            # self.patient_metadata = {'PatientName': 'Dummy Patient Folder'}
            # return True
            return False # Simulate failure
        def load_dicom_from_zip(self, zip_path):
            print(f"Dummy load from zip: {zip_path}")
            # self.volume_data = np.random.rand(10, 100, 100)
            # self.patient_metadata = {'PatientName': 'Dummy Patient ZIP'}
            # return True
            return False


    # Configure logging for standalone test
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # dummy_dm = DummyDataManager() # Use this for quick UI test without full backend
    # For more integrated test, ensure DataManager and its dependencies can be imported
    # You might need to adjust PYTHONPATH or run from the project root for this to work
    try:
        # This assumes you are running this file directly from QRadPlannerApp/ui/
        # Adjust path if running from project root
        # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        # sys.path.insert(0, project_root)
        from QRadPlannerApp.backend.data_manager import DataManager as RealDataManager
        dm_instance = RealDataManager()
    except ImportError as e:
        print(f"Could not import RealDataManager, using DummyDataManager: {e}")
        dm_instance = DummyDataManager()


    main_win = MainWindow(dm_instance)
    main_win.show()
    sys.exit(app.exec_())
