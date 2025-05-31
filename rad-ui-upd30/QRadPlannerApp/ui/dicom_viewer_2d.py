#--- START OF FILE dicom_viewer_2d.py ---

import logging
import numpy as np
from typing import Optional, List, Dict # Added List, Dict

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication # Added QApplication for main
from PyQt5.QtCore import Qt

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.patches import Polygon # For drawing finalized contours
import matplotlib.pyplot as plt # For plt.Line2D type hint

logger = logging.getLogger(__name__)

class DicomViewer2DWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.main_layout = QVBoxLayout(self)

        self.figure = Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.main_layout.addWidget(self.toolbar)

        self.main_layout.addWidget(self.canvas)

        self.axes = self.figure.add_subplot(111)
        self.current_slice_cax = None
        self.current_dose_cax = None

        # Drawing-related attributes
        self.is_drawing_mode = False
        self.current_slice_idx_for_drawing = -1 # Initialize to an invalid index
        self.current_slice_contour_points: List[tuple] = [] # Explicitly list of tuples
        self.slice_contours: Dict[int, List[List[tuple]]] = {}  # {slice_idx: [contour1_pts, contour2_pts, ...]}
        
        self.drawing_aid_lines: List[plt.Line2D] = [] # Store matplotlib line objects
        # self.contour_patches = {} # Removed, as axes.clear() and re-adding Polygon is simpler

        # self.setLayout(self.main_layout) # Already done by QVBoxLayout constructor
        # self.logger = logging.getLogger(__name__) # logger is module-level

        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)

        logger.info("DicomViewer2DWidget initialized with drawing attributes.")

    def _on_mouse_press(self, event):
        if not self.is_drawing_mode or event.inaxes != self.axes or event.button != 1: # Only left-clicks in drawing mode
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None: # Click was outside axes data range
            return

        self.current_slice_contour_points.append((x, y))
        logger.debug(f"Added point ({x:.2f}, {y:.2f}). Total points: {len(self.current_slice_contour_points)}")

        # Clear previous temporary drawing aid lines
        self._clear_drawing_aids()

        # Draw new aid lines/points
        if len(self.current_slice_contour_points) == 1:
            # Plot just the first point
            pt, = self.axes.plot(self.current_slice_contour_points[0][0], self.current_slice_contour_points[0][1], 
                                  'co-', markersize=4, linewidth=0.8, alpha=0.9) # Cyan for drawing
            self.drawing_aid_lines.append(pt)
        elif len(self.current_slice_contour_points) > 1:
            # Plot lines connecting all current points
            xs, ys = zip(*self.current_slice_contour_points)
            line, = self.axes.plot(xs, ys, 'co-', markersize=4, linewidth=0.8, alpha=0.9)
            self.drawing_aid_lines.append(line)
            # Optionally, close the loop visually if more than 2 points
            if len(self.current_slice_contour_points) > 2:
                closed_loop_xs = list(xs) + [xs[0]]
                closed_loop_ys = list(ys) + [ys[0]]
                loop_line, = self.axes.plot(closed_loop_xs, closed_loop_ys, 'c--', linewidth=0.7, alpha=0.7)
                self.drawing_aid_lines.append(loop_line)

        self.canvas.draw_idle()

    def start_drawing_mode(self):
        self.is_drawing_mode = True
        # Clear any points from a previous drawing session *before* activating for the current slice
        self.current_slice_contour_points.clear()
        self._clear_drawing_aids() # Clear old visual aids
        logger.info(f"Drawing mode activated for slice {self.current_slice_idx_for_drawing}.")
        self.canvas.draw_idle() # Redraw to clear old aids if any

    def stop_drawing_mode(self):
        self.is_drawing_mode = False
        # Current points are typically finalized or cleared before stopping mode.
        # For safety, clear them here too if not finalized, along with aids.
        self.current_slice_contour_points.clear()
        self._clear_drawing_aids()
        logger.info("Drawing mode deactivated.")
        self.canvas.draw_idle() # Redraw to clear aids

    def _clear_drawing_aids(self):
        for line_obj in self.drawing_aid_lines:
            if line_obj in self.axes.lines: # Keep this check as it's good practice
                try:
                    line_obj.remove() # Changed line
                except Exception as e: # More general catch, though remove() is usually safe
                    logger.debug(f"Error removing line_obj {line_obj} using .remove(): {e}")
        self.drawing_aid_lines.clear()

    def clear_current_drawing_on_slice(self):
        self.current_slice_contour_points.clear()
        self._clear_drawing_aids()
        logger.info("Cleared current interactive drawing points and aids.")
        self.canvas.draw_idle()

    def finalize_contour_on_slice(self):
        if not self.current_slice_contour_points or len(self.current_slice_contour_points) < 3:
            logger.info("Not enough points to finalize contour (minimum 3 required). Current points cleared.")
            self.clear_current_drawing_on_slice() # Clear points and aids
            return

        slice_idx = self.current_slice_idx_for_drawing 
        if slice_idx < 0: 
            logger.warning("Cannot finalize contour: current slice index is invalid.")
            self.clear_current_drawing_on_slice()
            return

        finalized_points = list(self.current_slice_contour_points) # Make a copy

        if slice_idx not in self.slice_contours:
            self.slice_contours[slice_idx] = []
        self.slice_contours[slice_idx].append(finalized_points)
        logger.info(f"Contour with {len(finalized_points)} points finalized and stored for slice {slice_idx}.")
        
        self.clear_current_drawing_on_slice() # Clear interactive points and aids for next drawing
        # The main update_slice will handle drawing the newly finalized contour by re-reading self.slice_contours

    def clear_all_contours_on_slice(self, slice_idx: Optional[int] = None): 
        target_slice_idx = slice_idx if slice_idx is not None else self.current_slice_idx_for_drawing
        if target_slice_idx < 0:
            logger.warning("Cannot clear contours: current slice index is invalid.")
            return

        cleared_data = False
        if target_slice_idx in self.slice_contours:
            del self.slice_contours[target_slice_idx]
            logger.info(f"Cleared all finalized contours data for slice {target_slice_idx}.")
            cleared_data = True
        else:
            logger.info(f"No finalized contours to clear for slice {target_slice_idx}.")

        # Since update_slice redraws from scratch, we don't need to manually remove patches here.
        # The call to _update_displayed_slice (or equivalent) from MainWindow will refresh the view.
        # If this method is called and it's the current slice, MainWindow needs to refresh.

    def update_slice(self, slice_idx: int, slice_data_normalized: Optional[np.ndarray], tumor_mask_slice=None, dose_slice=None):
        # If slice changes, and we are in drawing mode, stop drawing and clear current points for the *old* slice.
        if self.is_drawing_mode and self.current_slice_idx_for_drawing != slice_idx and self.current_slice_idx_for_drawing != -1:
            logger.info(f"Slice changed from {self.current_slice_idx_for_drawing} to {slice_idx} while drawing. Clearing temp points from old slice.")
            self.current_slice_contour_points.clear() 
            self._clear_drawing_aids()
            # Drawing mode remains active, but points are for the new slice now.

        self.current_slice_idx_for_drawing = slice_idx

        # If drawing mode is NOT active, ensure any leftover drawing aids are gone
        if not self.is_drawing_mode:
            self._clear_drawing_aids() # This ensures aids are cleared if mode was toggled off and slice then changed.

        self.axes.clear() # Clears everything: image, contours, patches, lines

        # Remove old colorbars if they exist (axes.clear() might not handle figure-level colorbars well)
        if self.current_slice_cax:
            try: self.figure.delaxes(self.current_slice_cax)
            except Exception: pass
            self.current_slice_cax = None
        if self.current_dose_cax:
            try: self.figure.delaxes(self.current_dose_cax)
            except Exception: pass
            self.current_dose_cax = None

        if slice_data_normalized is None:
            self.axes.text(0.5, 0.5, 'No DICOM Data Loaded', horizontalalignment='center', verticalalignment='center', transform=self.axes.transAxes)
            self.axes.axis('off')
            self.canvas.draw()
            return

        # Display base DICOM slice
        self.axes.imshow(slice_data_normalized, cmap='gray', aspect='equal')

        # Overlay tumor contour (from automatic detection or RTStruct)
        if tumor_mask_slice is not None and np.any(tumor_mask_slice):
            self.axes.contour(tumor_mask_slice.astype(float), levels=[0.5], colors='r', linewidths=0.7, alpha=0.8)

        # Overlay dose distribution
        if dose_slice is not None and np.any(dose_slice):
            img_dose = self.axes.imshow(dose_slice, cmap='hot', alpha=0.3, aspect='equal', interpolation='bilinear')
            try: # Make colorbar creation more robust
                self.current_dose_cax = self.figure.colorbar(img_dose, ax=self.axes, fraction=0.046, pad=0.04, label='Dose (Gy)')
            except Exception as e_cb:
                logger.warning(f"Could not create dose colorbar: {e_cb}")
                self.current_dose_cax = None
        
        # --- Draw Finalized Contours for the current slice ---
        if slice_idx in self.slice_contours:
            for contour_points_list in self.slice_contours[slice_idx]:
                if len(contour_points_list) >= 3:
                    polygon = Polygon(contour_points_list, closed=True, fill=False, edgecolor='cyan', linewidth=1.2, alpha=0.9)
                    self.axes.add_patch(polygon)
        
        # --- Re-draw current interactive drawing aids if any (because axes.clear() removed them) ---
        if self.is_drawing_mode and self.current_slice_contour_points:
            # Store points, clear current list and aids, then re-add points one by one
            # to correctly reconstruct the visual aids using the _on_mouse_press logic.
            points_to_re_add = list(self.current_slice_contour_points)
            self.current_slice_contour_points.clear()
            self._clear_drawing_aids() # Make sure previous aid objects are gone

            for x_pt, y_pt in points_to_re_add:
                # Simulate adding points to redraw aids as if user clicked
                self.current_slice_contour_points.append((x_pt, y_pt))
                if len(self.current_slice_contour_points) == 1:
                    pt, = self.axes.plot(self.current_slice_contour_points[0][0], self.current_slice_contour_points[0][1], 
                                          'co-', markersize=4, linewidth=0.8, alpha=0.9)
                    self.drawing_aid_lines.append(pt)
                elif len(self.current_slice_contour_points) > 1:
                    # Remove the single point marker if it was the only aid
                    if len(self.drawing_aid_lines) == 1 and len(self.current_slice_contour_points) == 2:
                         # Check if the first aid line is just a point (no xdata list)
                         if not hasattr(self.drawing_aid_lines[0], 'get_xdata') or len(self.drawing_aid_lines[0].get_xdata()) <= 1:
                            try:
                                self.drawing_aid_lines[0].remove()
                            except Exception as e:
                                logger.debug(f"Error removing single point marker: {e}")
                            self.drawing_aid_lines.clear()
                    
                    xs, ys = zip(*self.current_slice_contour_points)
                    line, = self.axes.plot(xs, ys, 'co-', markersize=4, linewidth=0.8, alpha=0.9)
                    # Ensure previous aid line (if any) is removed before adding new one for this sequence
                    if self.drawing_aid_lines and self.drawing_aid_lines[-1] is not line and self.drawing_aid_lines[-1] in self.axes.lines:
                        self.axes.lines.remove(self.drawing_aid_lines[-1])
                        self.drawing_aid_lines[-1] = line # Replace with the new line
                    elif not self.drawing_aid_lines:
                        self.drawing_aid_lines.append(line)
                    else: # If it's the same line object being updated, that's fine
                        pass


                    if len(self.current_slice_contour_points) > 2:
                        # Remove previous dashed loop if it exists
                        if len(self.drawing_aid_lines) > 0 and self.drawing_aid_lines[-1].get_linestyle() == '--':
                            if self.drawing_aid_lines[-1] in self.axes.lines: self.axes.lines.remove(self.drawing_aid_lines[-1])
                            self.drawing_aid_lines.pop()
                        
                        closed_loop_xs = list(xs) + [xs[0]]
                        closed_loop_ys = list(ys) + [ys[0]]
                        loop_line, = self.axes.plot(closed_loop_xs, closed_loop_ys, 'c--', linewidth=0.7, alpha=0.7)
                        self.drawing_aid_lines.append(loop_line) # Add the new dashed loop

        self.axes.axis('off')
        try:
            if self.figure.get_axes(): # Check if figure has any axes
                self.figure.tight_layout()
        except Exception as e:
            logger.warning(f"Error during tight_layout: {e}. This can sometimes happen with dynamic plot changes.")
        self.canvas.draw()
        logger.debug(f"Slice {slice_idx} updated and redrawn.")

    def clear_view(self):
        logger.info("Clearing 2D viewer.")
        self.axes.clear()
        if self.current_slice_cax:
            try: self.figure.delaxes(self.current_slice_cax)
            except Exception: pass 
            self.current_slice_cax = None
        if self.current_dose_cax:
            try: self.figure.delaxes(self.current_dose_cax)
            except Exception: pass 
            self.current_dose_cax = None
        
        # Clear drawing state
        self.is_drawing_mode = False
        self.current_slice_idx_for_drawing = -1
        self.current_slice_contour_points.clear()
        self.slice_contours.clear()
        self.drawing_aid_lines.clear() # Should be empty, but for safety

        self.axes.text(0.5, 0.5, 'Viewer Cleared', horizontalalignment='center', verticalalignment='center', transform=self.axes.transAxes)
        self.axes.axis('off')
        self.canvas.draw()

if __name__ == '__main__':
    # Example usage for testing this widget independently
    # sys module is needed for QApplication
    import sys 
    app = QApplication(sys.argv)
    logging.basicConfig(level=logging.DEBUG)

    viewer = DicomViewer2DWidget()

    # Create dummy data for testing
    dummy_slice_data = np.random.rand(100, 100)
    dummy_tumor_mask = np.zeros((100, 100), dtype=bool)
    dummy_tumor_mask[40:60, 40:60] = True
    dummy_dose_data = np.zeros((100, 100))
    dummy_dose_data[20:80, 20:80] = np.random.rand(60,60) * 50 # Example dose up to 50 Gy

    viewer.update_slice(0, dummy_slice_data, dummy_tumor_mask, dummy_dose_data)
    
    viewer.show()
    sys.exit(app.exec_())
