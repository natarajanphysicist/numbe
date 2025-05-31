import logging
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

logger = logging.getLogger(__name__)

class DvhPlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0,0,0,0)
        
        self.figure = Figure(figsize=(6, 4), dpi=100) # Adjust as needed
        self.canvas = FigureCanvas(self.figure)
        self.main_layout.addWidget(self.canvas)
        
        self.axes = self.figure.add_subplot(111)
        
        self.setLayout(self.main_layout)
        self.clear_plot() # Initial empty state
        logger.info("DvhPlotWidget initialized.")

    def plot_dvh(self, dvh_data_dict: dict):
        self.axes.clear()
        
        if not dvh_data_dict:
            logger.info("DVH data is None or empty. Clearing plot.")
            self.axes.text(0.5, 0.5, 'No DVH Data to Display', 
                           horizontalalignment='center', 
                           verticalalignment='center', 
                           transform=self.axes.transAxes)
            self.axes.set_xlabel("Dose (Gy)")
            self.axes.set_ylabel("Volume (%)")
            self.axes.set_title("Dose-Volume Histogram")
            self.axes.grid(True)
            try:
                if self.figure.get_axes(): self.figure.tight_layout()
            except Exception as e: logger.warning(f"Tight_layout error: {e}")
            self.canvas.draw()
            return

        logger.info(f"Plotting DVH for {len(dvh_data_dict)} ROIs.")
        has_data_to_plot = False
        for roi_name, data in dvh_data_dict.items():
            bins = data.get('bins')
            volume_pct = data.get('volume_pct')
            
            if bins is not None and volume_pct is not None and len(bins) > 0 and len(volume_pct) > 0:
                # Ensure bins and volume_pct have compatible lengths for plotting
                # Typically, bins might be N+1 for N hist values, or N for N line points
                # If 'bins' are edges, plot against left edges. If 'bins' are centers, fine.
                # Assuming 'bins' are the left edges from np.histogram, so len(bins) == len(volume_pct)
                if len(bins) == len(volume_pct):
                    self.axes.plot(bins, volume_pct, label=roi_name)
                    has_data_to_plot = True
                else:
                    logger.warning(f"Skipping ROI '{roi_name}' due to mismatched bins ({len(bins)}) and volume_pct ({len(volume_pct)}) lengths.")
            else:
                logger.warning(f"Skipping ROI '{roi_name}' due to missing or empty bins/volume_pct data.")
        
        if not has_data_to_plot:
             self.axes.text(0.5, 0.5, 'No valid DVH data points for selected ROIs.', 
                           horizontalalignment='center', 
                           verticalalignment='center', 
                           transform=self.axes.transAxes)

        self.axes.set_title("Dose-Volume Histogram")
        self.axes.set_xlabel("Dose (Gy)")
        self.axes.set_ylabel("Volume (%)")
        if has_data_to_plot: self.axes.legend(loc='best') # Only show legend if there's something plotted
        self.axes.grid(True, linestyle=':')
        self.axes.set_ylim(0, 105) # Give a bit of space above 100%
        self.axes.set_xlim(left=0) # Ensure x-axis starts at 0

        try:
            if self.figure.get_axes(): self.figure.tight_layout()
        except Exception as e: logger.warning(f"Tight_layout error: {e}")
        
        self.canvas.draw()

    def clear_plot(self):
        self.axes.clear()
        self.axes.text(0.5, 0.5, 'DVH Plot Area - Calculate Metrics & DVH to Populate', 
                       horizontalalignment='center', 
                       verticalalignment='center', 
                       transform=self.axes.transAxes)
        self.axes.set_xlabel("Dose (Gy)")
        self.axes.set_ylabel("Volume (%)")
        self.axes.set_title("Dose-Volume Histogram")
        self.axes.grid(True)
        try:
            if self.figure.get_axes(): self.figure.tight_layout()
        except Exception as e: logger.warning(f"Tight_layout error: {e}")
        self.canvas.draw()
        logger.info("DVH plot cleared.")

if __name__ == '__main__':
    # Example usage for testing this widget independently
    import sys
    from PyQt5.QtWidgets import QApplication
    import numpy as np

    logging.basicConfig(level=logging.DEBUG)
    
    app = QApplication(sys.argv)
    dvh_widget = DvhPlotWidget()
    
    # Example DVH data
    dummy_dvh_data = {
        "Tumor": {
            "bins": np.linspace(0, 70, 50),
            "volume_pct": 100 * np.exp(-np.linspace(0, 70, 50) / 30) + np.random.rand(50)*5
        },
        "Lung_L": {
            "bins": np.linspace(0, 70, 50),
            "volume_pct": 80 * np.exp(-np.linspace(0, 70, 50) / 10) + np.random.rand(50)*3
        },
        "SpinalCord": {
            "bins": np.linspace(0, 70, 50),
            "volume_pct": 50 * np.exp(-np.linspace(0, 70, 50) / 5)
        },
        "EmptyROI": { # Test case for empty/invalid data
             "bins": np.array([]),
             "volume_pct": np.array([])
        }
    }
    
    # dvh_widget.plot_dvh(dummy_dvh_data)
    dvh_widget.plot_dvh({}) # Test empty dict
    # dvh_widget.clear_plot()
    
    dvh_widget.resize(600, 400)
    dvh_widget.show()
    sys.exit(app.exec_())
