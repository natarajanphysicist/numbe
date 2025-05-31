# --- START OF FILE rad-ui-full-02-main/QRadPlannerApp/ui/dicom_viewer_3d.py ---

import logging
import numpy as np

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication
from PyQt5.QtCore import Qt

# VTK imports
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPiecewiseFunction
from vtkmodules.vtkRenderingCore import (
    vtkRenderer,
    vtkVolume,
    vtkVolumeProperty,
    vtkColorTransferFunction,
    vtkPolyDataMapper,
    vtkActor,
)
from vtkmodules.vtkImagingCore import vtkImageShiftScale
from vtkmodules.vtkFiltersCore import vtkMarchingCubes
from vtkmodules.vtkFiltersGeneral import vtkDiscreteMarchingCubes

# from vtkmodules.vtkRenderingVolume import vtkSmartVolumeMapper # <-- Original location
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper # <-- Try this location

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util import numpy_support
from typing import Optional, Dict, List
import sys

logger = logging.getLogger(__name__)

class DicomViewer3DWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0,0,0,0) # Use full space

        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.main_layout.addWidget(self.vtkWidget)

        self.ren = vtkRenderer()
        self.ren.SetBackground(0.1, 0.2, 0.4)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)

        self.volume_actor: Optional[vtkVolume] = None
        self.tumor_actor: Optional[vtkActor] = None
        self.dose_isosurface_actors: List[vtkActor] = []

        self.setLayout(self.main_layout)

        # Initialize and start the interactor
        self.vtkWidget.Initialize()
        # self.vtkWidget.Start() # Start is usually called by the Qt event loop

        logger.info("DicomViewer3DWidget initialized.")

    def _numpy_to_vtkimage(self, np_array: np.ndarray, image_properties: Optional[Dict] = None) -> vtkImageData:
        """Converts a NumPy array to vtkImageData, sets spacing and origin if properties are provided."""
        vtk_image = vtkImageData()

        depth, height, width = np_array.shape
        vtk_image.SetDimensions(width, height, depth)

        if image_properties:
            spacing_x = image_properties.get('pixel_spacing', [1.0, 1.0])[0]
            spacing_y = image_properties.get('pixel_spacing', [1.0, 1.0])[1]
            spacing_z = image_properties.get('slice_thickness', 1.0)
            vtk_image.SetSpacing(spacing_x, spacing_y, spacing_z)

            origin = image_properties.get('origin', [0.0, 0.0, 0.0])
            vtk_image.SetOrigin(origin[0], origin[1], origin[2])
        else:
            vtk_image.SetSpacing(1, 1, 1)
            vtk_image.SetOrigin(0, 0, 0)

        vtk_array = numpy_support.numpy_to_vtk(num_array=np_array.ravel(order='F'), deep=True)
        vtk_image.GetPointData().SetScalars(vtk_array)

        return vtk_image

    def update_volume(self, volume_data_full: Optional[np.ndarray],
                      image_properties: Optional[Dict],
                      tumor_mask_full: Optional[np.ndarray] = None):

        logger.info("Updating 3D Viewer (Volume and Tumor Mask)...")
        if self.volume_actor is not None:
            self.ren.RemoveVolume(self.volume_actor)
            self.volume_actor = None
        if self.tumor_actor is not None:
            self.ren.RemoveActor(self.tumor_actor)
            self.tumor_actor = None

        if volume_data_full is None or image_properties is None:
            logger.info("No volume data or properties to display in 3D view.")
            self.ren.ResetCamera()
            self.vtkWidget.GetRenderWindow().Render()
            return

        try:
            logger.info(f"Volume data shape: {volume_data_full.shape}, dtype: {volume_data_full.dtype}")
            vtk_volume_image = self._numpy_to_vtkimage(volume_data_full.astype(np.float32), image_properties)

            color_func = vtkColorTransferFunction()
            color_func.AddRGBPoint(-500, 0.1, 0.1, 0.1)
            color_func.AddRGBPoint(0,    0.5, 0.5, 0.5)
            color_func.AddRGBPoint(400,  0.8, 0.8, 0.7)
            color_func.AddRGBPoint(1000, 0.9, 0.9, 0.9)
            color_func.AddRGBPoint(3000, 1.0, 1.0, 1.0)

            opacity_func = vtkPiecewiseFunction()
            opacity_func.AddPoint(-500, 0.0)
            opacity_func.AddPoint(0,    0.05)
            opacity_func.AddPoint(400,  0.2)
            opacity_func.AddPoint(1000, 0.5)
            opacity_func.AddPoint(3000, 0.8)

            self.volume_property = vtkVolumeProperty()
            self.volume_property.SetColor(color_func)
            self.volume_property.SetScalarOpacity(opacity_func)
            self.volume_property.SetInterpolationTypeToLinear()
            self.volume_property.ShadeOn()
            self.volume_property.SetAmbient(0.3)
            self.volume_property.SetDiffuse(0.7)
            self.volume_property.SetSpecular(0.2)
            self.volume_property.SetSpecularPower(10.0)

            volume_mapper = vtkSmartVolumeMapper()
            volume_mapper.SetInputData(vtk_volume_image)

            self.volume_actor = vtkVolume()
            self.volume_actor.SetMapper(volume_mapper)
            self.volume_actor.SetProperty(self.volume_property)
            self.ren.AddVolume(self.volume_actor)
            logger.info("DICOM volume actor created and added to renderer.")

        except Exception as e:
            logger.error(f"Error creating DICOM volume actor: {e}", exc_info=True)
            if self.volume_actor: self.ren.RemoveVolume(self.volume_actor); self.volume_actor = None

        if tumor_mask_full is not None and np.any(tumor_mask_full):
            try:
                logger.info(f"Tumor mask data shape: {tumor_mask_full.shape}, dtype: {tumor_mask_full.dtype}")
                tumor_mask_uint8 = tumor_mask_full.astype(np.uint8)
                vtk_tumor_image = self._numpy_to_vtkimage(tumor_mask_uint8, image_properties)

                mc = vtkDiscreteMarchingCubes()
                mc.SetInputData(vtk_tumor_image)
                mc.SetValue(0, 1)
                mc.Update()

                mc_mapper = vtkPolyDataMapper()
                mc_mapper.SetInputConnection(mc.GetOutputPort())
                mc_mapper.ScalarVisibilityOff()

                self.tumor_actor = vtkActor()
                self.tumor_actor.SetMapper(mc_mapper)
                self.tumor_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
                self.tumor_actor.GetProperty().SetOpacity(0.3)
                self.ren.AddActor(self.tumor_actor)
                logger.info("Tumor mask actor created and added to renderer.")

            except Exception as e:
                logger.error(f"Error creating tumor mask actor: {e}", exc_info=True)
                if self.tumor_actor: self.ren.RemoveActor(self.tumor_actor); self.tumor_actor = None

        self.ren.ResetCamera()
        self.ren.ResetCameraClippingRange()
        self.ren.ResetCamera()
        self.ren.ResetCameraClippingRange()
        self.vtkWidget.GetRenderWindow().Render()
        logger.info("3D View updated (Volume and Tumor).")

    def _clear_dose_isosurfaces(self):
        logger.debug(f"Clearing {len(self.dose_isosurface_actors)} dose isosurface actors.")
        for actor in self.dose_isosurface_actors:
            self.ren.RemoveActor(actor)
        self.dose_isosurface_actors.clear()

    def _update_dose_isosurfaces(self, dose_volume_full: Optional[np.ndarray],
                                 image_properties: Optional[Dict],
                                 isovalues_list: Optional[List[float]] = None):
        logger.info("Updating dose isosurfaces...")
        self._clear_dose_isosurfaces()

        if dose_volume_full is None or image_properties is None or not isovalues_list:
            logger.info("No dose volume, properties, or isovalues provided. Isosurfaces cleared or not generated.")
            self.vtkWidget.GetRenderWindow().Render()
            return

        try:
            logger.info(f"Dose volume for isosurfaces shape: {dose_volume_full.shape}, dtype: {dose_volume_full.dtype}")
            dose_vtk_image = self._numpy_to_vtkimage(dose_volume_full.astype(np.float32), image_properties)

            colors_vtk = [
                (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 0.5, 0)
            ]

            for i, value in enumerate(isovalues_list):
                if not isinstance(value, (int, float)):
                    logger.warning(f"Skipping invalid isovalue: {value} (type: {type(value)})")
                    continue

                logger.debug(f"Creating isosurface for value: {value} Gy")

                contour_filter = vtkMarchingCubes()
                contour_filter.SetInputData(dose_vtk_image)
                contour_filter.SetValue(0, value)
                contour_filter.Update()

                if contour_filter.GetOutput() is None or contour_filter.GetOutput().GetNumberOfPoints() == 0:
                    logger.info(f"No geometry found for isovalue {value} Gy. Skipping this isosurface.")
                    continue

                mapper = vtkPolyDataMapper()
                mapper.SetInputConnection(contour_filter.GetOutputPort())
                mapper.ScalarVisibilityOff()

                actor = vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(colors_vtk[i % len(colors_vtk)])
                actor.GetProperty().SetOpacity(0.3)

                self.ren.AddActor(actor)
                self.dose_isosurface_actors.append(actor)
                logger.info(f"Added isosurface actor for {value} Gy with color {colors_vtk[i % len(colors_vtk)]}")

        except Exception as e:
            logger.error(f"Error creating dose isosurfaces: {e}", exc_info=True)

        self.ren.ResetCameraClippingRange()
        self.vtkWidget.GetRenderWindow().Render()
        logger.info("Dose isosurfaces updated.")


    def clear_view(self):
        logger.info("Clearing 3D viewer (volume, tumor, and dose isosurfaces).")
        if self.volume_actor is not None:
            self.ren.RemoveVolume(self.volume_actor)
            self.volume_actor = None
        if self.tumor_actor is not None:
            self.ren.RemoveActor(self.tumor_actor)
            self.tumor_actor = None
        self._clear_dose_isosurfaces()

        self.vtkWidget.GetRenderWindow().Render()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    logging.basicConfig(level=logging.DEBUG)

    viewer3d = DicomViewer3DWidget()

    vol_shape = (50, 64, 64)
    dummy_volume_data = np.random.randint(-1000, 2000, size=vol_shape, dtype=np.int16)
    dummy_tumor_mask = np.zeros(vol_shape, dtype=bool)
    dummy_tumor_mask[20:30, 25:35, 25:35] = True
    dummy_image_properties = {
        'pixel_spacing': [0.8, 0.8],
        'slice_thickness': 2.5,
        'origin': [0, 0, 0]
    }
    viewer3d.update_volume(dummy_volume_data, dummy_image_properties, tumor_mask_full=dummy_tumor_mask)
    viewer3d.resize(800, 600)
    viewer3d.show()
    sys.exit(app.exec_())
