#--- START OF FILE main_window.py ---

import sys
import os
import logging
import numpy as np 

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QAction, 
    QFileDialog, QMessageBox, QStatusBar, QLabel, QApplication, QTextEdit,
    QSlider, QLineEdit, QFormLayout, QTabWidget, QPushButton, QCheckBox, QSizePolicy 
)
from PyQt5.QtCore import Qt

from QRadPlannerApp.backend.data_manager import DataManager
from QRadPlannerApp.ui.dicom_viewer_2d import DicomViewer2DWidget
from QRadPlannerApp.ui.dicom_viewer_3d import DicomViewer3DWidget
from QRadPlannerApp.ui.dvh_plot_widget import DvhPlotWidget

logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    def __init__(self, data_manager: DataManager):
        super().__init__()
        self.data_manager = data_manager
        
        self.setWindowTitle("MG Health Tech - QRadPlanner - Radiotherapy Planning Tool")
        self.setGeometry(100, 100, 1800, 1000)  

        self._create_menu_bar()
        self._create_status_bar()
        self._create_viewer_controls() 
        self._init_central_widget() 
        
        logger.info("MainWindow initialized.")
        self._update_ui_element_states() 
        self._update_displayed_slice() 
        self._update_3d_viewer() 

    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        open_folder_action = QAction("Open DICOM Folder...", self)
        open_folder_action.setStatusTip("Open a folder containing DICOM files")
        open_folder_action.triggered.connect(self._open_dicom_folder)
        file_menu.addAction(open_folder_action)
        open_zip_action = QAction("Open DICOM ZIP...", self)
        open_zip_action.setStatusTip("Open a ZIP archive containing DICOM files")
        open_zip_action.triggered.connect(self._open_dicom_zip)
        file_menu.addAction(open_zip_action)

        open_file_action = QAction("Open DICOM File...", self)
        open_file_action.setStatusTip("Open a single DICOM file (loads the series it belongs to)")
        open_file_action.triggered.connect(self._open_dicom_file)
        file_menu.addAction(open_file_action)

        file_menu.addSeparator()
        exit_action = QAction("&Exit", self)
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready", 3000)

    def _init_central_widget(self):
        self.tabs = QTabWidget()
        self.view_2d_tab_content = QWidget()
        view_2d_main_layout = QHBoxLayout(self.view_2d_tab_content)
        self.viewer_2d = DicomViewer2DWidget()
        view_2d_main_layout.addWidget(self.viewer_2d, 3) 
        view_2d_main_layout.addWidget(self.viewer_controls_widget, 1) 
        self.tabs.addTab(self.view_2d_tab_content, "2D View")

        # --- 3D View Tab ---
        self.view_3d_tab_content = QWidget()
        view_3d_main_layout = QVBoxLayout(self.view_3d_tab_content)
        self.viewer_3d = DicomViewer3DWidget()
        view_3d_main_layout.addWidget(self.viewer_3d, 1)

        # --- Controls for 3D Display ---
        display_3d_controls_widget = QWidget()
        display_3d_controls_layout = QHBoxLayout(display_3d_controls_widget)
        
        # Dose Isovalues
        display_3d_controls_layout.addWidget(QLabel("Dose Isovalues (Gy):"))
        self.dose_isovalues_input = QLineEdit("10,30,50")
        self.dose_isovalues_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        display_3d_controls_layout.addWidget(self.dose_isovalues_input)
        
        self.update_3d_dose_button = QPushButton("Update 3D Dose") 
        self.update_3d_dose_button.clicked.connect(self._update_3d_dose_display)
        display_3d_controls_layout.addWidget(self.update_3d_dose_button)
        
        self.clear_3d_dose_button = QPushButton("Clear 3D Dose") 
        self.clear_3d_dose_button.clicked.connect(self._clear_3d_dose_display)
        display_3d_controls_layout.addWidget(self.clear_3d_dose_button)

        # Beam Visualization Toggle
        self.show_beams_checkbox = QCheckBox("Show Beams")
        self.show_beams_checkbox.toggled.connect(self._toggle_beam_visualization)
        display_3d_controls_layout.addWidget(self.show_beams_checkbox)
        
        view_3d_main_layout.addWidget(display_3d_controls_widget)
        self.tabs.addTab(self.view_3d_tab_content, "3D View")

        self.planning_tab_widget = self._create_planning_tab()
        self.tabs.addTab(self.planning_tab_widget, "Treatment Planning")
        self.results_tab_widget = self._create_results_tab()
        self.tabs.addTab(self.results_tab_widget, "Results & Analysis")
        self.setCentralWidget(self.tabs)

    def _create_viewer_controls(self): 
        self.viewer_controls_widget = QWidget()
        controls_layout = QVBoxLayout(self.viewer_controls_widget) 
        slice_nav_layout = QFormLayout()
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0); self.slice_slider.setMaximum(0); self.slice_slider.setValue(0)
        self.slice_slider.valueChanged.connect(self._update_displayed_slice)
        self.current_slice_label = QLabel("Slice: 0 / 0")
        slice_nav_layout.addRow("Navigate Slices:", self.slice_slider)
        slice_nav_layout.addRow(self.current_slice_label)
        controls_layout.addLayout(slice_nav_layout)

        wl_layout = QFormLayout()
        self.wc_input = QLineEdit("40"); self.ww_input = QLineEdit("400")
        self.wc_input.editingFinished.connect(self._update_displayed_slice)
        self.ww_input.editingFinished.connect(self._update_displayed_slice)
        wl_layout.addRow("Window Center (WC):", self.wc_input)
        wl_layout.addRow("Window Width (WW):", self.ww_input)
        controls_layout.addLayout(wl_layout)
        
        self.show_dose_checkbox = QCheckBox("Show Dose Overlay")
        self.show_dose_checkbox.setChecked(True) 
        self.show_dose_checkbox.stateChanged.connect(self._update_displayed_slice)
        self.show_dose_checkbox.setStatusTip("Toggle visibility of dose overlay on 2D view.")
        controls_layout.addWidget(self.show_dose_checkbox)

        self.detect_tumors_button = QPushButton("Detect Tumors")
        self.detect_tumors_button.clicked.connect(self._run_tumor_detection)
        self.detect_tumors_button.setStatusTip("Run automatic tumor detection on the loaded volume.")
        controls_layout.addWidget(self.detect_tumors_button)
        
        controls_layout.addWidget(QLabel("ROI Drawing (Current Slice):"))
        drawing_buttons_group = QWidget()
        drawing_buttons_layout = QHBoxLayout(drawing_buttons_group)
        drawing_buttons_layout.setContentsMargins(0,0,0,0)
        self.start_draw_button = QPushButton("Start Drawing")
        self.start_draw_button.setCheckable(True) 
        self.start_draw_button.toggled.connect(self._toggle_drawing_mode_2d)
        drawing_buttons_layout.addWidget(self.start_draw_button)
        self.finalize_contour_button = QPushButton("Finalize Contour")
        self.finalize_contour_button.clicked.connect(self._finalize_contour_2d)
        drawing_buttons_layout.addWidget(self.finalize_contour_button)
        controls_layout.addWidget(drawing_buttons_group)

        clear_buttons_group = QWidget()
        clear_buttons_layout = QHBoxLayout(clear_buttons_group)
        clear_buttons_layout.setContentsMargins(0,0,0,0)
        self.clear_current_draw_button = QPushButton("Clear Current Points")
        self.clear_current_draw_button.clicked.connect(self._clear_current_drawing_2d)
        clear_buttons_layout.addWidget(self.clear_current_draw_button)
        self.clear_slice_contours_button = QPushButton("Clear Slice's Contours")
        self.clear_slice_contours_button.clicked.connect(self._clear_slice_contours_2d)
        clear_buttons_layout.addWidget(self.clear_slice_contours_button)
        controls_layout.addWidget(clear_buttons_group)
        
        self.use_contours_as_tumor_button = QPushButton("Use Drawn Contours as Tumor")
        self.use_contours_as_tumor_button.clicked.connect(self._use_drawn_contours_as_tumor)
        self.use_contours_as_tumor_button.setStatusTip("Generate a 3D tumor mask from all finalized contours and set it for planning.")
        controls_layout.addWidget(self.use_contours_as_tumor_button)
        controls_layout.addStretch(1) 

    def _update_ui_element_states(self):
        dicom_loaded = self.data_manager.volume_data is not None
        planner_initialized = self.data_manager.planner is not None
        tumor_data_in_planner = planner_initialized and \
                                self.data_manager.planner.tumor_mask is not None and \
                                np.any(self.data_manager.planner.tumor_mask)
        beam_weights_calculated = planner_initialized and \
                                  self.data_manager.plan_results.get('beam_weights') is not None
        dose_calculated = planner_initialized and \
                          self.data_manager.dose_distribution is not None

        self.slice_slider.setEnabled(dicom_loaded)
        self.wc_input.setEnabled(dicom_loaded)
        self.ww_input.setEnabled(dicom_loaded)
        self.show_dose_checkbox.setEnabled(dicom_loaded and dose_calculated) 
        self.detect_tumors_button.setEnabled(dicom_loaded)
        self.start_draw_button.setEnabled(dicom_loaded)
        
        # More precise enabling for drawing buttons
        can_finalize_or_clear_current = dicom_loaded and self.viewer_2d.is_drawing_mode and bool(self.viewer_2d.current_slice_contour_points)
        self.finalize_contour_button.setEnabled(can_finalize_or_clear_current and len(self.viewer_2d.current_slice_contour_points) >=3)
        self.clear_current_draw_button.setEnabled(can_finalize_or_clear_current)
        
        # Enable if any contours exist on any slice
        has_any_finalized_contours = dicom_loaded and bool(self.viewer_2d.slice_contours) and \
                                     any(v for v_list in self.viewer_2d.slice_contours.values() for v in v_list) # Check for non-empty contour lists
        self.clear_slice_contours_button.setEnabled(dicom_loaded and bool(self.viewer_2d.slice_contours.get(self.slice_slider.value()))) # Only if current slice has contours
        self.use_contours_as_tumor_button.setEnabled(has_any_finalized_contours)

        self.dose_isovalues_input.setEnabled(dicom_loaded and dose_calculated)
        if hasattr(self, 'update_3d_dose_button'): 
            self.update_3d_dose_button.setEnabled(dicom_loaded and dose_calculated) 
            self.clear_3d_dose_button.setEnabled(dicom_loaded and dose_calculated)  
        if hasattr(self, 'show_beams_checkbox'):
            self.show_beams_checkbox.setEnabled(planner_initialized and beam_weights_calculated)


        self.init_planner_button.setEnabled(dicom_loaded)
        self.run_optimization_button.setEnabled(planner_initialized and tumor_data_in_planner)
        self.calculate_dose_button.setEnabled(planner_initialized and beam_weights_calculated)
        
        # Enable results tab and its button if plan is complete
        if hasattr(self, 'results_tab_widget'): # Check if tab exists
            self.results_tab_widget.setEnabled(planner_initialized and dose_calculated)
        self.calculate_metrics_button.setEnabled(planner_initialized and dose_calculated)

        logger.debug(f"UI States Updated: DICOM Loaded={dicom_loaded}, Planner Init={planner_initialized}, "
                     f"Tumor in Planner={tumor_data_in_planner}, Weights Calc={beam_weights_calculated}, Dose Calc={dose_calculated}")

    def _clear_planning_and_results(self):
        logger.info("Clearing previous planning and results data.")
        if self.data_manager: 
            self.data_manager.planner = None 
            self.data_manager.dose_distribution = None
            self.data_manager.plan_results = {} 
        if hasattr(self, 'planning_status_text'): self.planning_status_text.clear()
        if hasattr(self, 'metrics_display_text'): self.metrics_display_text.clear()
        if hasattr(self, 'dvh_plot_widget') and self.dvh_plot_widget: self.dvh_plot_widget.clear_plot()
        
        self._clear_3d_dose_display() # This calls _update_ui_element_states
        if hasattr(self, 'show_beams_checkbox'): 
            if self.show_beams_checkbox.isChecked(): # Only toggle if it's checked to avoid double toggle
                self.show_beams_checkbox.setChecked(False) # This will trigger _toggle_beam_visualization to clear beams
        elif hasattr(self, 'viewer_3d') and hasattr(self.viewer_3d, '_clear_beam_visualization'): 
             self.viewer_3d._clear_beam_visualization() 
             if self.viewer_3d.vtkWidget.GetRenderWindow(): self.viewer_3d.vtkWidget.GetRenderWindow().Render()


        self._update_ui_element_states() # Ensure final state consistency

    def _toggle_drawing_mode_2d(self, checked):
        if self.viewer_2d:
            if checked:
                self.viewer_2d.start_drawing_mode()
                self.start_draw_button.setText("Stop Drawing")
                logger.debug("2D drawing mode started.")
            else:
                self.viewer_2d.stop_drawing_mode()
                self.start_draw_button.setText("Start Drawing")
                logger.debug("2D drawing mode stopped.")
        self._update_ui_element_states()

    def _finalize_contour_2d(self):
        if self.viewer_2d:
            self.viewer_2d.finalize_contour_on_slice()
            self._update_displayed_slice() 
            logger.debug("2D contour finalized.")
        self._update_ui_element_states()

    def _clear_current_drawing_2d(self):
        if self.viewer_2d:
            self.viewer_2d.clear_current_drawing_on_slice()
            logger.debug("Current 2D drawing cleared.")
        self._update_ui_element_states()

    def _clear_slice_contours_2d(self):
        if self.viewer_2d:
            current_slice_for_clear = self.slice_slider.value()
            self.viewer_2d.clear_all_contours_on_slice(current_slice_for_clear) 
            self._update_displayed_slice() 
            logger.debug(f"All contours on 2D slice {current_slice_for_clear} cleared.")
        self._update_ui_element_states()

    def _update_3d_dose_display(self):
        logger.info("Attempting to update 3D dose display.")
        if self.data_manager.dose_distribution is None:
            QMessageBox.warning(self, "No Dose Data", "Dose distribution not calculated yet.")
            return
        if self.data_manager.image_properties is None: 
            QMessageBox.warning(self, "No Image Properties", "DICOM image properties are missing. 3D dose might be misaligned.")
        isovalues_str = self.dose_isovalues_input.text()
        try:
            isovalues_list = [float(val.strip()) for val in isovalues_str.split(',') if val.strip()]
            if not isovalues_list:
                QMessageBox.information(self, "Input Info", "No isovalues entered. Clearing 3D dose.")
                self._clear_3d_dose_display()
                return
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Invalid isovalues. Please enter comma-separated numbers.")
            return
        self.status_bar.showMessage("Updating 3D dose isosurfaces...", 2000)
        QApplication.processEvents()
        try:
            self.viewer_3d._update_dose_isosurfaces(
                dose_volume_full_crs=self.data_manager.dose_distribution, 
                image_properties=self.data_manager.image_properties,
                isovalues_list=isovalues_list
            )
            self.status_bar.showMessage("3D dose display updated.", 3000)
        except Exception as e:
            logger.error(f"Error calling _update_dose_isosurfaces: {e}", exc_info=True)
            QMessageBox.critical(self, "3D Dose Error", f"Failed to update 3D dose display: {e}")
            self.status_bar.showMessage("Error updating 3D dose display.", 3000)
        # self._update_ui_element_states() # Already called by _clear_3d_dose_display if triggered

    def _clear_3d_dose_display(self):
        logger.info("Clearing 3D dose display.")
        if hasattr(self.viewer_3d, '_clear_dose_isosurfaces'):
            self.viewer_3d._clear_dose_isosurfaces()
            if hasattr(self.viewer_3d, 'vtkWidget') and self.viewer_3d.vtkWidget.GetRenderWindow():
                 self.viewer_3d.vtkWidget.GetRenderWindow().Render()
            self.status_bar.showMessage("3D dose display cleared.", 3000)
        else:
            logger.warning("DicomViewer3DWidget does not have _clear_dose_isosurfaces method.")
        self._update_ui_element_states()
        
    def _toggle_beam_visualization(self, checked: bool):
        if not self.data_manager.planner or self.data_manager.plan_results.get('beam_weights') is None:
            if checked: 
                QMessageBox.information(self, "No Plan", "Optimize a plan first to visualize beams.")
                self.show_beams_checkbox.setChecked(False) 
            else: 
                if hasattr(self.viewer_3d, '_clear_beam_visualization'):
                    self.viewer_3d._clear_beam_visualization()
                    if self.viewer_3d.vtkWidget.GetRenderWindow(): self.viewer_3d.vtkWidget.GetRenderWindow().Render()
            self._update_ui_element_states() # Ensure checkbox state is reflected if disabled
            return

        if checked:
            self.status_bar.showMessage("Visualizing beams...", 2000)
            QApplication.processEvents()
            beam_viz_data = self.data_manager.planner.get_beam_visualization_data()
            if beam_viz_data:
                self.viewer_3d.display_beams(beam_viz_data)
                self.status_bar.showMessage("Beams displayed.", 3000)
            else:
                QMessageBox.warning(self, "Beam Data Error", "Could not retrieve beam visualization data from planner.")
                self.show_beams_checkbox.setChecked(False)
                self.status_bar.showMessage("Error displaying beams.", 3000)
        else:
            if hasattr(self.viewer_3d, '_clear_beam_visualization'):
                self.viewer_3d._clear_beam_visualization()
                if self.viewer_3d.vtkWidget.GetRenderWindow(): self.viewer_3d.vtkWidget.GetRenderWindow().Render()
                self.status_bar.showMessage("Beam visualization cleared.", 3000)
        
        self._update_ui_element_states()


    def _run_tumor_detection(self):
        logger.info("Attempting to run tumor detection...")
        if self.data_manager.volume_data is None:
            QMessageBox.warning(self, "No Data", "Please load a DICOM volume before running tumor detection."); return
        try:
            self.status_bar.showMessage("Running tumor detection...", 3000) 
            QApplication.processEvents()
            if hasattr(self.data_manager, 'tumor_detector') and self.data_manager.tumor_detector is not None:
                new_tumor_mask = self.data_manager.tumor_detector.detect_tumors(self.data_manager.volume_data) 
                if new_tumor_mask is not None:
                    self.data_manager.tumor_mask = new_tumor_mask 
                    logger.info(f"Tumor detection complete. New mask sum: {np.sum(new_tumor_mask)}")
                    QMessageBox.information(self, "Tumor Detection", "Tumor detection complete. Mask updated.")
                else:
                    logger.error("Tumor detection returned None.")
                    QMessageBox.warning(self, "Tumor Detection", "Tumor detection failed to produce a mask.")
            else: 
                 logger.error("DataManager's tumor_detector not accessible/None.")
                 QMessageBox.critical(self, "Error", "Tumor detector not available via DataManager.")
                 self._update_ui_element_states(); return
            self._update_displayed_slice(); self._update_3d_viewer() 
            self.status_bar.showMessage("Tumor detection finished.", 3000)
        except Exception as e:
            logger.error(f"Error during tumor detection: {e}", exc_info=True)
            QMessageBox.critical(self, "Tumor Detection Error", f"An error occurred: {e}")
            self.status_bar.showMessage("Tumor detection error.", 3000)
        self._update_ui_element_states()

    def _update_displayed_slice(self):
        if self.data_manager.volume_data is None:
            self.viewer_2d.clear_view()
            self.current_slice_label.setText("Slice: N/A")
            self._update_ui_element_states(); return # Update states and exit if no data
        
        slice_idx = self.slice_slider.value()
        num_slices = self.data_manager.volume_data.shape[0] 
        self.current_slice_label.setText(f"Slice: {slice_idx + 1} / {num_slices}")
        try:
            wc = float(self.wc_input.text()); ww = float(self.ww_input.text())
            if ww <= 0: ww = 400.0; self.ww_input.setText(str(ww))
        except ValueError:
            wc, ww = 40, 400; self.wc_input.setText(str(wc)); self.ww_input.setText(str(ww))
        raw_slice = self.data_manager.volume_data[slice_idx, :, :] 
        lower_bound = wc - (ww / 2.0); upper_bound = wc + (ww / 2.0)
        display_slice = np.clip(raw_slice, lower_bound, upper_bound)
        display_slice_normalized = np.clip((display_slice - lower_bound) / (ww + 1e-9), 0.0, 1.0)
        tumor_slice = None
        if self.data_manager.tumor_mask is not None and slice_idx < self.data_manager.tumor_mask.shape[0]:
            tumor_slice = self.data_manager.tumor_mask[slice_idx, :, :]
        dose_slice_to_display = None
        if self.show_dose_checkbox.isChecked() and self.data_manager.dose_distribution is not None:
            if slice_idx < self.data_manager.dose_distribution.shape[2]: 
                dose_slice_raw = self.data_manager.dose_distribution[:, :, slice_idx] 
                dose_slice_to_display = dose_slice_raw.T 
        self.viewer_2d.update_slice(slice_idx, display_slice_normalized, tumor_slice, dose_slice_to_display)
        self._update_ui_element_states()


    def _open_dicom_folder(self):
        start_dir = os.path.expanduser("~") 
        folder_path = QFileDialog.getExistingDirectory(self, "Open DICOM Folder", start_dir, QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        if folder_path:
            self._clear_planning_and_results() 
            logger.info(f"Selected DICOM folder: {folder_path}")
            self.status_bar.showMessage(f"Loading DICOM folder: {folder_path}...")
            QApplication.processEvents() 
            load_success = self.data_manager.load_dicom_from_folder(folder_path)
            if load_success and self.data_manager.volume_data is not None:
                patient_name = self.data_manager.patient_metadata.get('PatientName', 'Unknown Patient')
                num_slices = self.data_manager.volume_data.shape[0] 
                self.slice_slider.setMaximum(num_slices - 1 if num_slices > 0 else 0)
                self.slice_slider.setValue((num_slices - 1) // 2 if num_slices > 0 else 0)
                self.wc_input.setText("40"); self.ww_input.setText("400")
                self._update_displayed_slice() 
                self._update_3d_viewer() 
                status_msg = f"Loaded: {patient_name} - {num_slices} slices."
                self.status_bar.showMessage(status_msg, 5000); logger.info(status_msg)
            else: 
                self.viewer_2d.clear_view(); 
                if hasattr(self, 'viewer_3d'): self.viewer_3d.clear_view() 
                self.slice_slider.setMaximum(0); self.current_slice_label.setText("Slice: N/A")
                error_msg = f"Failed to load DICOM series from folder: {folder_path}. Check logs."
                self.status_bar.showMessage(error_msg, 5000); logger.error(error_msg)
                QMessageBox.critical(self, "Load Error", error_msg)
            self._update_ui_element_states() 

    def _open_dicom_zip(self):
        start_dir = os.path.expanduser("~")
        zip_file_path, _ = QFileDialog.getOpenFileName(self, "Open DICOM ZIP Archive", start_dir, "ZIP Files (*.zip);;All Files (*)")
        if zip_file_path:
            self._clear_planning_and_results() 
            logger.info(f"Selected DICOM ZIP file: {zip_file_path}")
            self.status_bar.showMessage(f"Loading DICOM ZIP: {zip_file_path}...")
            QApplication.processEvents()
            load_success = self.data_manager.load_dicom_from_zip(zip_file_path)
            if load_success and self.data_manager.volume_data is not None:
                patient_name = self.data_manager.patient_metadata.get('PatientName', 'Unknown Patient')
                num_slices = self.data_manager.volume_data.shape[0]
                self.slice_slider.setMaximum(num_slices - 1 if num_slices > 0 else 0)
                self.slice_slider.setValue((num_slices - 1) // 2 if num_slices > 0 else 0)
                self.wc_input.setText("40"); self.ww_input.setText("400")
                self._update_displayed_slice()
                self._update_3d_viewer() 
                status_msg = f"Loaded: {patient_name} - {num_slices} slices."
                self.status_bar.showMessage(status_msg, 5000); logger.info(status_msg)
            else: 
                self.viewer_2d.clear_view(); 
                if hasattr(self, 'viewer_3d'): self.viewer_3d.clear_view()
                self.slice_slider.setMaximum(0); self.current_slice_label.setText("Slice: N/A")
                error_msg = f"Failed to load DICOM series from ZIP: {zip_file_path}. Check logs."
                self.status_bar.showMessage(error_msg, 5000); logger.error(error_msg)
                QMessageBox.critical(self, "Load Error", error_msg)
            self._update_ui_element_states() 

    def _open_dicom_file(self):
        start_dir = os.path.expanduser("~")
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Single DICOM File", start_dir, "DICOM Files (*.dcm);;All Files (*)")
        if file_path:
            directory_path = os.path.dirname(file_path)
            self._clear_planning_and_results()
            logger.info(f"Selected DICOM file: {file_path}, loading from directory: {directory_path}")
            self.status_bar.showMessage(f"Loading DICOM series from file's folder: {directory_path}...")
            QApplication.processEvents()
            load_success = self.data_manager.load_dicom_from_folder(directory_path)
            if load_success and self.data_manager.volume_data is not None:
                patient_name = self.data_manager.patient_metadata.get('PatientName', 'Unknown Patient')
                num_slices = self.data_manager.volume_data.shape[0]
                self.slice_slider.setMaximum(num_slices - 1 if num_slices > 0 else 0)
                self.slice_slider.setValue((num_slices - 1) // 2 if num_slices > 0 else 0)
                self.wc_input.setText("40"); self.ww_input.setText("400")
                self._update_displayed_slice()
                self._update_3d_viewer()
                status_msg = f"Loaded: {patient_name} - {num_slices} slices from directory of {os.path.basename(file_path)}."
                self.status_bar.showMessage(status_msg, 5000); logger.info(status_msg)
            else:
                self.viewer_2d.clear_view()
                if hasattr(self, 'viewer_3d'): self.viewer_3d.clear_view()
                self.slice_slider.setMaximum(0); self.current_slice_label.setText("Slice: N/A")
                error_msg = f"Failed to load DICOM series from directory of file: {file_path}. Check logs."
                self.status_bar.showMessage(error_msg, 5000); logger.error(error_msg)
                QMessageBox.critical(self, "Load Error", error_msg)
            self._update_ui_element_states()

    def _update_3d_viewer(self):
        logger.debug("Attempting to update 3D viewer (volume, tumor, OARs).")
        if self.data_manager.volume_data is None or self.data_manager.image_properties is None:
            if hasattr(self, 'viewer_3d'): self.viewer_3d.clear_view()
            logger.debug("3D Viewer cleared as no volume data/properties are available.")
        else:
            try:
                self.viewer_3d.update_volume(
                    volume_data_full_zyx=self.data_manager.volume_data,    
                    image_properties=self.data_manager.image_properties,
                    tumor_mask_full_zyx=self.data_manager.tumor_mask,     
                    oar_masks_full_zyx=self.data_manager.oar_masks_from_rtstruct 
                )
                logger.info("3D Viewer updated with new volume, tumor, and OAR data.")
            except Exception as e:
                logger.error(f"Error updating 3D viewer: {e}", exc_info=True)
                QMessageBox.warning(self, "3D View Error", f"Could not update 3D view: {e}")
                if hasattr(self, 'viewer_3d'): self.viewer_3d.clear_view()
        self._update_ui_element_states() 
    
    def _create_planning_tab(self) -> QWidget:
        planning_widget = QWidget()
        main_layout = QVBoxLayout(planning_widget)
        form_layout = QFormLayout()
        self.num_beams_input = QLineEdit("8")
        form_layout.addRow("Number of Beams:", self.num_beams_input)
        self.num_fractions_input = QLineEdit("30")
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
        return planning_widget

    def _initialize_planner_from_ui(self):
        if self.data_manager.volume_data is None:
            QMessageBox.warning(self, "No Data", "Load DICOM data before initializing planner."); return
        if self.data_manager.planner is not None:
            reply = QMessageBox.question(self, 'Re-initialize Planner?',
                                         "Planner already initialized. Re-initialize with current settings? Clears existing plan.",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No: self._update_ui_element_states(); return
            else: self._clear_planning_and_results()
        self.planning_status_text.append("Initializing planner...")
        QApplication.processEvents()
        try:
            num_beams = int(self.num_beams_input.text())
            if not (4 <= num_beams <= 20):
                QMessageBox.warning(self, "Input Error", "Number of beams must be between 4 and 20.")
                self.planning_status_text.append("Init failed: Invalid number of beams."); self._update_ui_element_states(); return
            init_success = self.data_manager.initialize_planner(num_beams_override=num_beams)
            if init_success and self.data_manager.planner is not None:
                msg = "Planner initialized."
                if self.data_manager.planner.tumor_mask is not None and np.any(self.data_manager.planner.tumor_mask):
                    msg += f" Using tumor mask (Volume: {np.sum(self.data_manager.planner.tumor_mask)} voxels)."
                else: msg += " WARNING: No valid tumor mask in planner; may use default tumor."
                self.planning_status_text.append(msg); logger.info(msg)
            else:
                error_msg = "Planner initialization failed. Check logs."
                self.planning_status_text.append(error_msg); logger.error(error_msg)
                QMessageBox.critical(self, "Planner Error", error_msg)
        except ValueError:
            self.planning_status_text.append("Init failed: Invalid number for beams.")
            QMessageBox.warning(self, "Input Error", "Invalid number of beams.")
        except Exception as e:
            error_msg = f"Planner initialization error: {e}"
            self.planning_status_text.append(error_msg); logger.error(f"Error init planner UI: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", error_msg)
        self._update_ui_element_states()

    def _run_optimization_from_ui(self):
        if not self.data_manager.planner: QMessageBox.warning(self, "Planner Not Ready", "Initialize planner first."); return
        if self.data_manager.planner.tumor_mask is None or not np.any(self.data_manager.planner.tumor_mask):
            QMessageBox.warning(self, "No Target", "No tumor target in planner. Optimization cannot proceed.")
            self.planning_status_text.append("Optimization aborted: No tumor target."); return
        self.planning_status_text.append("Running beam optimization..."); self.status_bar.showMessage("Optimizing beams...", 0)
        QApplication.processEvents()
        try:
            opt_success = self.data_manager.run_beam_optimization()
            if opt_success:
                weights_str = np.array2string(self.data_manager.plan_results.get('beam_weights', np.array([])), precision=3)
                msg = f"Beam optimization complete. Weights: {weights_str}"
                self.planning_status_text.append(msg); logger.info(msg); self.status_bar.showMessage("Optimization finished.", 5000)
            else:
                self.planning_status_text.append("Optimization failed. Check logs."); logger.error("Optimization failed UI.")
                self.status_bar.showMessage("Optimization failed.", 5000)
        except Exception as e:
            error_msg = f"Optimization error: {e}"
            self.planning_status_text.append(error_msg); logger.error(f"Error running optimization UI: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", error_msg); self.status_bar.showMessage("Optimization error.", 5000)
        self._update_ui_element_states()

    def _calculate_dose_from_ui(self):
        if self.data_manager.planner is None or self.data_manager.plan_results.get('beam_weights') is None:
            QMessageBox.warning(self, "Prerequisite Missing", "Initialize planner and run beam optimization first."); return
        self.planning_status_text.append("Calculating dose distribution..."); self.status_bar.showMessage("Calculating dose...", 0)
        QApplication.processEvents()
        try:
            dose_success = self.data_manager.calculate_dose_distribution()
            if dose_success:
                msg = "Dose calculation complete."
                if self.data_manager.dose_distribution is not None: 
                    msg += f" Dose Min: {self.data_manager.dose_distribution.min():.2f}, Max: {self.data_manager.dose_distribution.max():.2f} Gy"
                self.planning_status_text.append(msg); logger.info(msg); self.status_bar.showMessage("Dose calculation finished.", 5000)
                self._update_displayed_slice() 
            else:
                self.planning_status_text.append("Dose calculation failed. Check logs."); logger.error("Dose calculation failed UI.")
                self.status_bar.showMessage("Dose calculation failed.", 5000)
        except Exception as e:
            error_msg = f"Dose calculation error: {e}"
            self.planning_status_text.append(error_msg); logger.error(f"Error calculating dose UI: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", error_msg); self.status_bar.showMessage("Dose calculation error.", 5000)
        self._update_ui_element_states()
            
    def _create_results_tab(self) -> QWidget:
        results_widget = QWidget()
        layout = QVBoxLayout(results_widget)
        self.calculate_metrics_button = QPushButton("Calculate Metrics & DVH")
        self.calculate_metrics_button.clicked.connect(self._calculate_metrics_dvh_from_ui)
        layout.addWidget(self.calculate_metrics_button)
        self.metrics_display_text = QTextEdit()
        self.metrics_display_text.setReadOnly(True)
        layout.addWidget(self.metrics_display_text)
        self.dvh_plot_widget = DvhPlotWidget()
        layout.addWidget(self.dvh_plot_widget)
        return results_widget

    def _calculate_metrics_dvh_from_ui(self):
        if self.data_manager.dose_distribution is None or not self.data_manager.planner:
            QMessageBox.warning(self, "Prerequisites Missing", "Ensure dose is calculated and planner is initialized."); return
        self.planning_status_text.append("Calculating metrics & DVH..."); self.status_bar.showMessage("Calculating metrics & DVH...", 0)
        QApplication.processEvents()
        try:
            target_dose = float(self.target_dose_input.text()); num_fractions = int(self.num_fractions_input.text())
            if num_fractions <= 0:
                QMessageBox.warning(self, "Input Error", "Number of fractions must be positive.")
                self.planning_status_text.append("Metrics/DVH failed: Invalid num fractions."); self.status_bar.showMessage("Invalid input.", 5000); return
            self.planning_status_text.append(f"Using prescription: {target_dose}Gy in {num_fractions} fracs for eval.")
            metrics_success = self.data_manager.get_plan_metrics(target_prescription_dose=target_dose, num_fractions_for_radiobio=num_fractions)
            dvh_success = self.data_manager.get_dvh_data()
            if metrics_success:
                metrics = self.data_manager.plan_results.get('metrics', {})
                metrics_str = "Plan Metrics:\n"
                if 'tumor' in metrics: metrics_str += f"  Tumor:\n"; [metrics_str := metrics_str + f"    {k}: {f'{v_obj:.3f}' if isinstance(v_obj, float) else str(v_obj)}\n" for k, v_obj in metrics['tumor'].items()]
                if 'oars' in metrics: metrics_str += f"  OARs:\n"; [metrics_str := metrics_str + f"    {oar}:\n" + "".join([f"      {k}: {f'{v_obj:.3f}' if isinstance(v_obj, float) else str(v_obj)}\n" for k, v_obj in oar_metrics.items()]) for oar, oar_metrics in metrics['oars'].items()]
                self.metrics_display_text.setText(metrics_str); self.planning_status_text.append("Metrics calculated.")
            else: self.metrics_display_text.setText("Failed to calculate metrics."); self.planning_status_text.append("Metrics calculation failed."); self.dvh_plot_widget.clear_plot()
            if dvh_success:
                dvh_data_from_dm = self.data_manager.plan_results.get('dvh_data')
                if dvh_data_from_dm: self.dvh_plot_widget.plot_dvh(dvh_data_from_dm); self.planning_status_text.append("DVH plot updated."); logger.info(f"DVH plotted for: {list(dvh_data_from_dm.keys())}")
                else: self.dvh_plot_widget.clear_plot(); self.planning_status_text.append("DVH data not available to plot.")
            else: self.planning_status_text.append("DVH generation failed."); self.dvh_plot_widget.clear_plot()
            self.status_bar.showMessage("Metrics & DVH calculation finished.", 5000)
        except ValueError:
            self.planning_status_text.append("Metrics/DVH failed: Invalid target dose or num fractions.")
            QMessageBox.warning(self, "Input Error", "Invalid target dose or number of fractions."); self.status_bar.showMessage("Invalid input.", 5000)
        except Exception as e:
            error_msg = f"Metrics/DVH calculation error: {e}"
            self.planning_status_text.append(error_msg); logger.error(f"Error calculating metrics/DVH UI: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", error_msg); self.status_bar.showMessage("Metrics/DVH error.", 5000)
        self._update_ui_element_states()

    def _use_drawn_contours_as_tumor(self):
        if not hasattr(self, 'viewer_2d') or self.viewer_2d is None: QMessageBox.critical(self, "Error", "2D Viewer not available."); return
        if not self.viewer_2d.slice_contours: QMessageBox.information(self, "No Contours", "No contours finalized."); return
        if not any(contours_on_slice for contours_on_slice in self.viewer_2d.slice_contours.values()):
            QMessageBox.information(self, "No Contours", "Finalized contours are empty."); return
        if self.data_manager.volume_data is None: QMessageBox.warning(self, "No Volume Data", "Cannot process contours without loaded DICOM."); return
        all_slice_contours = self.viewer_2d.slice_contours 
        self.status_bar.showMessage("Processing drawn contours into tumor mask...", 3000); QApplication.processEvents()
        success = self.data_manager.set_tumor_mask_from_contours(all_slice_contours)
        if success:
            QMessageBox.information(self, "Success", "Tumor mask updated from drawn contours.\nPlanner also updated if initialized.")
            self.status_bar.showMessage("Tumor mask updated from contours.", 5000)
            self._update_displayed_slice(); self._update_3d_viewer() 
        else:
            QMessageBox.critical(self, "Error", "Failed to create or set tumor mask from contours. Check logs.")
            self.status_bar.showMessage("Error processing contours.", 5000)
        self._update_ui_element_states()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    try:
        from QRadPlannerApp.backend.data_manager import DataManager as RealDataManager
        dm_instance = RealDataManager(); logger.info("Using RealDataManager.")
    except ImportError as e:
        logger.warning(f"Could not import RealDataManager, using DummyDataManager: {e}")
        class DummyDataManager: 
            def __init__(self): self.patient_metadata = {}; self.volume_data = None; self.image_properties = None; self.tumor_mask = None; self.dose_distribution = None; self.plan_results = {}; self.planner = None; self.oar_masks_from_rtstruct = {}
            def load_dicom_from_folder(self,p): return False
            def load_dicom_from_zip(self,p): return False
            def initialize_planner(self,num_beams_override=None): return False
            def run_beam_optimization(self): return False
            def calculate_dose_distribution(self): return False
            def get_plan_metrics(self,t,n): return False
            def get_dvh_data(self): return False
            def set_tumor_mask_from_contours(self,contours): logger.info("Dummy: set_tumor_mask_from_contours"); return True
        dm_instance = DummyDataManager()
    main_win = MainWindow(dm_instance)
    main_win.show()
    sys.exit(app.exec_())
