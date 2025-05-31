# --- START OF FILE rad-ui-full-02-main/QRadPlannerApp/ui/dicom_viewer_2d.py ---

import logging
import numpy as np
from typing import Optional # <-- ADD THIS LINE

from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import Qt

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.patches import Polygon # For drawing finalized contours

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
        self.current_slice_contour_points = []
        self.slice_contours = {}  # {slice_idx: [[(x,y),...], ...]}
        self.drawing_aid_lines = []
        self.contour_patches = {} # {slice_idx: [PolygonPatch,...]}

        self.setLayout(self.main_layout)
        self.logger = logging.getLogger(__name__) # Ensure logger instance is set

        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)

        self.logger.info("DicomViewer2DWidget initialized with drawing attributes.")

    def _on_mouse_press(self, event):
        if not self.is_drawing_mode or event.inaxes != self.axes or event.button != 1: # Only left-clicks in drawing mode
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None: # Click was outside axes data range
            return

        self.current_slice_contour_points.append((x, y))
        self.logger.debug(f"Added point ({x:.2f}, {y:.2f}). Total points: {len(self.current_slice_contour_points)}")

        # Clear previous temporary drawing aid lines
        for line_obj in self.drawing_aid_lines:
            try:
                line_obj.remove()
            except ValueError: # Already removed or not in axes
                pass
        self.drawing_aid_lines.clear()

        # Draw new aid lines/points
        if len(self.current_slice_contour_points) == 1:
            # Plot just the first point
            pt, = self.axes.plot(self.current_slice_contour_points[0][0], self.current_slice_contour_points[0][1], 'ro-', markersize=3, linewidth=0.7, alpha=0.7) # Matched alpha
            self.drawing_aid_lines.append(pt)
        elif len(self.current_slice_contour_points) > 1:
            # Plot lines connecting all current points
            xs, ys = zip(*self.current_slice_contour_points)
            line, = self.axes.plot(xs, ys, 'ro-', markersize=3, linewidth=0.7, alpha=0.7) # Matched alpha
            self.drawing_aid_lines.append(line)

        self.canvas.draw_idle()

    def start_drawing_mode(self):
        self.is_drawing_mode = True
        # Clear any points from a previous drawing session on this slice or another
        self.current_slice_contour_points.clear()
        for line_obj in self.drawing_aid_lines: # Clear old visual aids
            try:
                line_obj.remove()
            except ValueError:
                pass
        self.drawing_aid_lines.clear()
        self.logger.info("Drawing mode activated. Click to add contour points.")
        self.canvas.draw_idle() # Redraw to clear old aids

    def stop_drawing_mode(self):
        self.is_drawing_mode = False
        # Current points are typically finalized or cleared before stopping mode.
        # For safety, clear them here too if not finalized.
        self.current_slice_contour_points.clear()
        for line_obj in self.drawing_aid_lines:
            try:
                line_obj.remove()
            except ValueError:
                pass
        self.drawing_aid_lines.clear()
        self.logger.info("Drawing mode deactivated.")
        self.canvas.draw_idle() # Redraw to clear aids

    def clear_current_drawing_on_slice(self):
        self.current_slice_contour_points.clear()
        for line_obj in self.drawing_aid_lines:
            try:
                line_obj.remove()
            except ValueError:
                pass
        self.drawing_aid_lines.clear()
        self.logger.info("Cleared current interactive drawing.")
        self.canvas.draw_idle()

    def finalize_contour_on_slice(self):
        if not self.current_slice_contour_points or len(self.current_slice_contour_points) < 3:
            self.logger.info("Not enough points to finalize contour (minimum 3 required).")
            # Still clear current interactive drawing aids and points
            self.current_slice_contour_points.clear()
            for line_obj in self.drawing_aid_lines:
                try: line_obj.remove()
                except ValueError: pass
            self.drawing_aid_lines.clear()
            self.canvas.draw_idle()
            return

        slice_idx = self.current_slice_idx_for_drawing # Assumes this is set by update_slice
        if slice_idx < 0: # Check if slice_idx is valid, was if slice_idx is None
            self.logger.warning("Cannot finalize contour: current slice index is invalid or not set.")
            return

        finalized_points = list(self.current_slice_contour_points) # Make a copy

        if slice_idx not in self.slice_contours:
            self.slice_contours[slice_idx] = []
        self.slice_contours[slice_idx].append(finalized_points)
        self.logger.info(f"Contour points stored for slice {slice_idx} with {len(finalized_points)} points. Persistent display deferred.")

        # Clear current interactive drawing aids and points
        self.current_slice_contour_points.clear()
        for line_obj in self.drawing_aid_lines:
            try: line_obj.remove()
            except ValueError: pass
        self.drawing_aid_lines.clear()

        self.canvas.draw_idle() # Redraw to remove aid lines

    def clear_all_contours_on_slice(self, slice_idx: Optional[int] = None): # Made slice_idx optional
        target_slice_idx = slice_idx if slice_idx is not None else self.current_slice_idx_for_drawing
        if target_slice_idx < 0:
            self.logger.warning("Cannot clear contours: current slice index is invalid.")
            return

        cleared_data = False
        if target_slice_idx in self.slice_contours:
            del self.slice_contours[target_slice_idx]
            self.logger.info(f"Cleared all finalized contours data for slice {target_slice_idx}.")
            cleared_data = True

        cleared_patches = False
        if target_slice_idx in self.contour_patches:
            for patch in self.contour_patches[target_slice_idx]:
                try:
                    patch.remove()
                except ValueError:
                    pass
            del self.contour_patches[target_slice_idx]
            self.logger.info(f"Removed contour patches from display for slice {target_slice_idx}.")
            cleared_patches = True

        if cleared_data or cleared_patches:
             # If this is the current slice, we need to redraw it.
            if target_slice_idx == self.current_slice_idx_for_drawing:
                # Need a way to signal the main window to redraw the current slice
                # If this widget is parented directly in a layout managed by MainWindow,
                # self.parent() might give access, but a signal is cleaner.
                # For now, assuming the parent has an _update_displayed_slice method accessible or signal connection exists.
                # If self.parent() is not MainWindow, this will fail.
                # A proper solution would involve emitting a signal here.
                # For now, try accessing parent method (assuming direct parentage in MainWindow layout)
                if hasattr(self.parent(), '_update_displayed_slice'):
                     self.parent()._update_displayed_slice() # Trigger MainWindow to refresh
                else:
                    # If parent doesn't have the method, just redraw canvas to remove aids
                    self.canvas.draw_idle()
                    self.logger.warning("Parent does not have _update_displayed_slice method. Cannot trigger full slice refresh.")
            else: # If clearing a non-current slice, just ensure canvas is up-to-date if it happens to be visible (though unlikely)
                self.canvas.draw_idle()
            self.logger.info(f"All contours cleared for slice {target_slice_idx+1}.")


    def update_slice(self, slice_idx: int, slice_data_normalized: Optional[np.ndarray], tumor_mask_slice=None, dose_slice=None):
        self.current_slice_idx_for_drawing = slice_idx # ENSURE THIS IS PRESENT AND FIRST

        # If changing slice, clear any temporary user drawing from the previous slice
        # This logic should be here, before self.axes.clear()
        if hasattr(self, '_last_drawn_slice_idx') and self._last_drawn_slice_idx != slice_idx:
            self.current_slice_contour_points.clear()
            for line in self.drawing_aid_lines:
                try: line.remove()
                except ValueError: pass
            self.drawing_aid_lines.clear()
        self._last_drawn_slice_idx = slice_idx # Store for next update

        self.axes.clear()

        # Remove old colorbars if they exist
        if hasattr(self, 'current_slice_cax') and self.current_slice_cax:
            try:
                self.figure.delaxes(self.current_slice_cax) # Use delaxes for proper removal
            except Exception as e:
                logger.debug(f"Error removing current_slice_cax: {e}")
            self.current_slice_cax = None

        if hasattr(self, 'current_dose_cax') and self.current_dose_cax:
            try:
                self.figure.delaxes(self.current_dose_cax) # Use delaxes for proper removal
            except Exception as e:
                logger.debug(f"Error removing current_dose_cax: {e}")
            self.current_dose_cax = None

        if slice_data_normalized is None:
            self.axes.text(0.5, 0.5, 'No DICOM Data Loaded', horizontalalignment='center', verticalalignment='center', transform=self.axes.transAxes)
            self.axes.axis('off')
            self.canvas.draw()
            return

        # Display base DICOM slice
        img_main = self.axes.imshow(slice_data_normalized, cmap='gray', aspect='equal')
        # self.current_slice_cax = self.figure.colorbar(img_main, ax=self.axes, fraction=0.046, pad=0.04, label='Image Intensity')


        # Overlay tumor contour
        if tumor_mask_slice is not None and np.any(tumor_mask_slice):
            # Ensure tumor_mask_slice is boolean or binary
            contour_mask = tumor_mask_slice.astype(float) # contour works better with float levels
            self.axes.contour(contour_mask, levels=[0.5], colors='r', linewidths=0.5, alpha=0.8) # levels=[0.5] for binary mask
            logger.debug("Tumor mask overlay added.")

        # Overlay dose distribution
        if dose_slice is not None and np.any(dose_slice):
            # Use a different colormap for dose, e.g., 'hot' or 'jet'
            # Alpha blending allows underlying image to be visible
            # Use a consistent vmin/vmax for dose if desired, e.g., from overall dose range
            # For now, default scaling of dose_slice is used by imshow
            img_dose = self.axes.imshow(dose_slice, cmap='hot', alpha=0.3, aspect='equal', interpolation='bilinear') # vmin=0, vmax=max_expected_dose
            self.current_dose_cax = self.figure.colorbar(img_dose, ax=self.axes, fraction=0.046, pad=0.04, label='Dose (Gy)')
            logger.debug("Dose slice overlay added.")
        elif hasattr(self, 'current_dose_cax') and self.current_dose_cax: # Ensure dose colorbar is removed if no dose slice
            try:
                self.figure.delaxes(self.current_dose_cax)
            except Exception as e:
                logger.debug(f"Error removing current_dose_cax when no dose_slice: {e}")
            self.current_dose_cax = None

        self.axes.axis('off')

        try:
            # Check if figure has any axes before calling tight_layout
            if self.figure.get_axes():
                self.figure.tight_layout()
        except Exception as e:
            logger.warning(f"Error during tight_layout: {e}. This can sometimes happen with dynamic plot changes.")

        self.canvas.draw()
        logger.debug("Slice updated and redrawn.")

    def clear_view(self):
        logger.info("Clearing 2D viewer.")
        self.axes.clear()
        if hasattr(self, 'current_slice_cax') and self.current_slice_cax:
            try: self.figure.delaxes(self.current_slice_cax)
            except Exception: pass # Ignore if already removed
            self.current_slice_cax = None
        if hasattr(self, 'current_dose_cax') and self.current_dose_cax:
            try: self.figure.delaxes(self.current_dose_cax)
            except Exception: pass # Ignore if already removed
            self.current_dose_cax = None
        self.axes.text(0.5, 0.5, 'Viewer Cleared', horizontalalignment='center', verticalalignment='center', transform=self.axes.transAxes)
        self.axes.axis('off')
        self.canvas.draw()

if __name__ == '__main__':
    # Example usage for testing this widget independently
    app = QApplication(sys.argv)
    logging.basicConfig(level=logging.DEBUG)

    viewer = DicomViewer2DWidget()

    # Create dummy data for testing
    dummy_slice = np.random.rand(100, 100)
    dummy_tumor = np.zeros((100, 100), dtype=bool)
    dummy_tumor[40:60, 40:60] = True
    dummy_dose = np.zeros((100, 100))
    dummy_dose[20:80, 20:80] = np.random.rand(60,60) * 50 # Example dose up to 50 Gy

    viewer.update_slice(0, dummy_slice, dummy_tumor, dummy_dose)
    # viewer.update_slice(dummy_slice, dummy_tumor)
    # viewer.update_slice(dummy_slice)
    # viewer.clear_view()

    viewer.show()
    sys.exit(app.exec_())
