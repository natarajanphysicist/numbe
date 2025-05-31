#--- START OF FILE dicom_viewer_3d.py ---

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
from vtkmodules.vtkImagingCore import vtkImageShiftScale # Not used in current version but kept if needed later
from vtkmodules.vtkFiltersCore import vtkMarchingCubes
from vtkmodules.vtkFiltersGeneral import vtkDiscreteMarchingCubes # Good for binary masks
from vtkmodules.vtkFiltersSources import vtkLineSource, vtkConeSource, vtkCylinderSource # For beam viz
from vtkmodules.vtkCommonMath import vtkMatrix4x4 # For transformations (not directly used in current _planner_coords_to_patient_world_coords but good to have if needed)


from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper 

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util import numpy_support
from typing import Optional, Dict, List, Tuple 
import sys

logger = logging.getLogger(__name__)

class DicomViewer3DWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0,0,0,0) 

        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.main_layout.addWidget(self.vtkWidget)

        self.ren = vtkRenderer()
        self.ren.SetBackground(0.1, 0.2, 0.4) 
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)

        self.volume_actor: Optional[vtkVolume] =```python
--- START OF FILE dicom_viewer_3d.py ---

import logging None
        self.tumor_actor: Optional[vtkActor] = None
        self.oar_actors
import numpy as np

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication
from PyQt5.: Dict[str, vtkActor] = {} 
        self.dose_isosurface_actors: List[vtkQtCore import Qt

# VTK imports
from vtkmodules.vtkCommonDataModel import vtkImageData, vActor] = []
        self.beam_visualization_actors: List[vtkActor] = []
        self.imagetkPiecewiseFunction
from vtkmodules.vtkRenderingCore import (
    vtkRenderer,
    vtkVolume_properties_for_viz: Optional[Dict] = None # Store for transformations

        self.vtkWidget.,
    vtkVolumeProperty,
    vtkColorTransferFunction,
    vtkPolyDataMapper,
    vtkInitialize()
        logger.info("DicomViewer3DWidget initialized.")

    def _numpy_to_vtkimage(Actor,
)
from vtkmodules.vtkImagingCore import vtkImageShiftScale 
from vtkself, np_array_s_r_c: np.ndarray, image_properties: Optional[Dict]modules.vtkFiltersCore import vtkMarchingCubes
from vtkmodules.vtkFiltersGeneral import vtkDiscreteMarchingCubes 
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper  = None) -> vtkImageData:
        vtk_image = vtkImageData()
        depth_s, height_r, width_c = np_array_s_r_c.shape # slices, rows, cols
        
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vvtk_image.SetDimensions(width_c, height_r, depth_s) # VTK: width, height,tkmodules.util import numpy_support
from vtkmodules.vtkFiltersSources import vtkLineSource, vtk depth

        if image_properties:
            spacing_x_vtk = image_properties.get('pixel_spacing', [ConeSource, vtkCylinderSource # For beam viz
from vtkmodules.vtkCommonMath import vtk1.0, 1.0])[1] 
            spacing_y_vtk = image_properties.Matrix4x4 # For transformations (not directly used yet but good to have)


from typing import Optional, Dict, Listget('pixel_spacing', [1.0, 1.0])[0] 
            spacing_z, Tuple 
import sys

logger = logging.getLogger(__name__)

class DicomViewer3DWidget(QWidget):
_vtk = image_properties.get('slice_thickness', 1.0)
            vtk_image.SetSpacing(spacing_x_vtk, spacing_y_vtk, spacing_z_vtk)
            
    def __init__(self, parent=None):
        super().__init__(parent)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0,0,0            origin_pat = image_properties.get('origin', [0.0, 0.0, 0.,0) 

        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.0])
            vtk_image.SetOrigin(origin_pat[0], origin_pat[1], origin_pat[2])
        else: 
            vtk_image.SetSpacing(1.0,1main_layout.addWidget(self.vtkWidget)

        self.ren = vtkRenderer()
        self.ren.SetBackground(0.1, 0.2, 0.4) 
        self.0,1.0); vtk_image.SetOrigin(0.0,0.0,0.vtkWidget.GetRenderWindow().AddRenderer(self.ren)

        self.volume_actor: Optional.0)

        vtk_array = numpy_support.numpy_to_vtk(num_array=np[vtkVolume] = None
        self.tumor_actor: Optional[vtkActor] = None
        self_array_s_r_c.ravel(order='F'), deep=True, array_type=vtkImageData.Get.oar_actors: Dict[str, vtkActor] = {} 
        self.dose_isosScalarType())
        vtk_image.GetPointData().SetScalars(vtk_array)
        return vtk_imageurface_actors: List[vtkActor] = []
        self.beam_visualization_actors: List[vtk
    
    def _create_surface_actor_from_mask(self, mask_data_zyx:Actor] = []
        self.image_properties_for_viz: Optional[Dict] = None 

        self. np.ndarray, 
                                        image_properties: Dict, 
                                        color: Tuple[float, float, floatvtkWidget.Initialize()
        logger.info("DicomViewer3DWidget initialized.")

    def _numpy_to], 
                                        opacity: float = 0.3) -> Optional[vtkActor]:
        if mask_vtkimage(self, np_array_s_r_c: np.ndarray, image_properties:_data_zyx is None or not np.any(mask_data_zyx):
            return None Optional[Dict] = None) -> vtkImageData:
        """
        Converts a NumPy array (slices
        try:
            vtk_mask_image = self._numpy_to_vtkimage(mask_data_zyx, rows, cols) to vtkImageData.
        Sets spacing and origin if properties are provided.
        vtk.astype(np.uint8), image_properties)
            mc = vtkDiscreteMarchingCubes() ImageData expects dimensions (width, height, depth) i.e. (cols, rows, slices).
        np
            mc.SetInputData(vtk_mask_image)
            mc.SetValue(0, 1_array_s_r_c.ravel(order='F') should produce data in VTK's xyz order.) 
            mc.Update()
            if mc.GetOutput() is None or mc.GetOutput().
        """
        vtk_image = vtkImageData()
        depth_s, height_r, widthGetNumberOfPoints() == 0:
                logger.debug("No surface generated by marching cubes for a mask.")
                return_c = np_array_s_r_c.shape # slices, rows, cols
        vtk_image.Set None
            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(mc.GetOutputPort())
            Dimensions(width_c, height_r, depth_s) # VTK: width, height, depth

mapper.ScalarVisibilityOff() 
            actor = vtkActor()
            actor.SetMapper(mapper)        if image_properties:
            # image_properties['pixel_spacing'] is [row_spacing, col
            actor.GetProperty().SetColor(color[0], color[1], color[2])
            _spacing]
            # image_properties['slice_thickness'] is z_spacing
            # VTK spacingactor.GetProperty().SetOpacity(opacity)
            return actor
        except Exception as e:
            logger is (spacing_x, spacing_y, spacing_z) -> (col_spacing, row_spacing, slice_.error(f"Error creating surface actor from mask: {e}", exc_info=True)
            returnthk)
            spacing_x_vtk = image_properties.get('pixel_spacing', [1. None

    def update_volume(self, volume_data_full_zyx: Optional[np.ndarray0, 1.0])[1] # Col spacing
            spacing_y_vtk = image_properties.],    # (s,r,c)
                      image_properties: Optional[Dict],
                      tumor_maskget('pixel_spacing', [1.0, 1.0])[0] # Row spacing
            spacing_full_zyx: Optional[np.ndarray] = None,   # (s,r,c)_z_vtk = image_properties.get('slice_thickness', 1.0)
            vtk_image.Set
                      oar_masks_full_zyx: Optional[Dict[str, np.ndarray]] = NoneSpacing(spacing_x_vtk, spacing_y_vtk, spacing_z_vtk)
            
            ): # (s,r,c)
        
        self.image_properties_for_viz = image# Origin is (x,y,z) of the first voxel's corner (or center depending on DICOM interpretation_properties # Store for beam viz and coord transforms

        logger.info("Updating 3D Viewer (Volume, Tumor, OARs)...")
        if self.volume_actor is not None: self.ren.RemoveVolume()
            # For ITK/VTK, typically corner. DICOM ImagePositionPatient is center of first voxelself.volume_actor); self.volume_actor = None
        if self.tumor_actor is not None: self.ren.RemoveActor(self.tumor_actor); self.tumor_actor = None
        self.
            # This needs careful alignment if absolute patient coordinates are critical.
            # For now, assume image_properties['origin'] is directly usable.
            origin_pat = image_properties.get('origin', [0.0, 0.._clear_oar_actors()
        self._clear_beam_visualization() # Clear beams when volume changes

        if volume_data_full_zyx is None or image_properties is None:
            logger.info0, 0.0])
            vtk_image.SetOrigin(origin_pat[0], origin_("No volume data or properties to display in 3D view.")
            if self.vtkWidget.GetRenderWindowpat[1], origin_pat[2])
        else: 
            vtk_image.SetSpacing(1.0,1.0,1.0); vtk_image.SetOrigin(0.0,0.0,(): # Ensure render window exists
                self.ren.ResetCamera()
                self.vtkWidget.GetRenderWindow().0.0)

        vtk_array = numpy_support.numpy_to_vtk(num_array=np_Render()
            return

        try: 
            vtk_volume_image = self._numpy_to_vtkarray_s_r_c.ravel(order='F'), deep=True, array_type=vtkImageData.Getimage(volume_data_full_zyx.astype(np.float32), image_properties)
ScalarType())
        vtk_image.GetPointData().SetScalars(vtk_array)
        return v            color_func = vtkColorTransferFunction(); opacity_func = vtkPiecewiseFunction()
            colortk_image
    
    def _create_surface_actor_from_mask(self, mask_data_func.AddRGBPoint(-500,0.1,0.1,0.1);color_func.AddRGBPoint(0,0.5,0.5,0.5);color_func_zyx: np.ndarray, 
                                        image_properties: Dict, 
                                        color: Tuple[float.AddRGBPoint(400,0.8,0.8,0.7);color_func, float, float], 
                                        opacity: float = 0.3) -> Optional[vtkActor]:.AddRGBPoint(1000,0.9,0.9,0.9);color_
        if mask_data_zyx is None or not np.any(mask_data_zyx):
            returnfunc.AddRGBPoint(3000,1,1,1)
            opacity_func.Add None
        try:
            vtk_mask_image = self._numpy_to_vtkimage(mask_Point(-500,0);opacity_func.AddPoint(0,0.05);opacity_data_zyx.astype(np.uint8), image_properties)
            mc = vtkDiscreteMarchingCubesfunc.AddPoint(400,0.2);opacity_func.AddPoint(1000()
            mc.SetInputData(vtk_mask_image)
            mc.SetValue(0, 1) 
            mc.Update()
            if mc.GetOutput() is None or mc.GetOutput,0.5);opacity_func.AddPoint(3000,0.8)
            self.volume_property = vtkVolumeProperty(); self.volume_property.SetColor(color_func); self().GetNumberOfPoints() == 0:
                logger.debug("No surface generated by marching cubes for a mask.")
                .volume_property.SetScalarOpacity(opacity_func)
            self.volume_property.SetInterpolationreturn None
            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(mc.GetOutputTypeToLinear(); self.volume_property.ShadeOn(); self.volume_property.SetAmbient(0Port())
            mapper.ScalarVisibilityOff() 
            actor = vtkActor()
            actor.SetMapper(mapper.3); self.volume_property.SetDiffuse(0.7); self.volume_property.SetSpecular)
            actor.GetProperty().SetColor(color[0], color[1], color[2])
(0.2); self.volume_property.SetSpecularPower(10.0)
            volume_            actor.GetProperty().SetOpacity(opacity)
            return actor
        except Exception as e:
            mapper = vtkSmartVolumeMapper(); volume_mapper.SetInputData(vtk_volume_image)
            logger.error(f"Error creating surface actor from mask: {e}", exc_info=True)
            self.volume_actor = vtkVolume(); self.volume_actor.SetMapper(volume_mapper); selfreturn None

    def update_volume(self, volume_data_full_zyx: Optional[np..volume_actor.SetProperty(self.volume_property)
            self.ren.AddVolume(selfndarray],    # (s,r,c)
                      image_properties: Optional[Dict],
                      tumor_mask.volume_actor); logger.info("DICOM volume actor added.")
        except Exception as e_vol: _full_zyx: Optional[np.ndarray] = None,   # (s,r,c)
            logger.error(f"Error creating DICOM volume actor: {e_vol}", exc_info=True)
            if self.volume_actor: self.ren.RemoveVolume(self.volume_actor); self.
                      oar_masks_full_zyx: Optional[Dict[str, np.ndarray]] = None): # (s,r,c)
        
        self.image_properties_for_viz = imagevolume_actor = None


        if tumor_mask_full_zyx is not None and np.any(tumor__properties # Store for beam viz and other transforms

        logger.info("Updating 3D Viewer (Volume, Tumor,mask_full_zyx):
            self.tumor_actor = self._create_surface_actor_from OARs)...")
        if self.volume_actor is not None: self.ren.RemoveVolume(_mask(
                tumor_mask_full_zyx, image_properties, color=(1.0,self.volume_actor); self.volume_actor = None
        if self.tumor_actor is not None 0.0, 0.0), opacity=0.4 
            )
            if self.: self.ren.RemoveActor(self.tumor_actor); self.tumor_actor = None
        selftumor_actor: self.ren.AddActor(self.tumor_actor); logger.info("Tumor mask actor added.")

        if oar_masks_full_zyx:
            oar_colors = [(0._clear_oar_actors()
        self._clear_beam_visualization() # Clear beams when volume changes,1,0), (0,0,1), (1,1,0), (0,1,

        if volume_data_full_zyx is None or image_properties is None:
            logger.info1), (1,0,1), (0.5,0.5,0), (0,0("No volume data or properties to display in 3D view.")
            self.ren.ResetCamera(); 
            if self.vtkWidget.GetRenderWindow(): self.vtkWidget.GetRenderWindow().Render()
.5,0.5), (0.5,0,0.5)]
            color_idx = 0
            for name, oar_mask_zyx_data in oar_masks_full_zyx.items():
            return

        try:
            vtk_volume_image = self._numpy_to_vtkimage(volume_data_full_zyx.astype(np.float32), image_properties)
            color_func                if oar_mask_zyx_data is None or not np.any(oar_mask_zyx_data): continue
                oar_actor = self._create_surface_actor_from_mask(
 = vtkColorTransferFunction(); opacity_func = vtkPiecewiseFunction()
            color_func.AddRGBPoint(-500,0.1,0.1,0.1);color_func.Add                    oar_mask_zyx_data, image_properties, 
                    color=oar_colorsRGBPoint(0,0.5,0.5,0.5);color_func.AddRGBPoint[color_idx % len(oar_colors)], opacity=0.25
                )
                if(400,0.8,0.8,0.7);color_func.AddRGBPoint oar_actor: 
                    self.ren.AddActor(oar_actor); self.oar_actors(1000,0.9,0.9,0.9);color_func.AddRGB[name] = oar_actor
                    logger.info(f"OAR actor for '{name}' added.");Point(3000,1,1,1)
            opacity_func.AddPoint(-50 color_idx += 1
        
        if self.vtkWidget.GetRenderWindow():
            self.0,0);opacity_func.AddPoint(0,0.05);opacity_func.AddPointren.ResetCamera()
            self.ren.ResetCameraClippingRange()
            self.vtkWidget.(400,0.2);opacity_func.AddPoint(1000,0.5GetRenderWindow().Render()
        logger.info("3D View updated (Volume, Tumor, OARs)."));opacity_func.AddPoint(3000,0.8)
            self.volume_property

    def _clear_oar_actors(self):
        logger.debug(f"Clearing {len(self = vtkVolumeProperty(); self.volume_property.SetColor(color_func); self.volume_property.oar_actors)} OAR actors.")
        for actor in self.oar_actors.values():.SetScalarOpacity(opacity_func)
            self.volume_property.SetInterpolationTypeToLinear(); self.volume_property.ShadeOn(); self.volume_property.SetAmbient(0.3); self
            self.ren.RemoveActor(actor)
        self.oar_actors.clear()

    .volume_property.SetDiffuse(0.7); self.volume_property.SetSpecular(0.2def _clear_dose_isosurfaces(self):
        logger.debug(f"Clearing {len(self); self.volume_property.SetSpecularPower(10.0)
            volume_mapper = vtk.dose_isosurface_actors)} dose isosurface actors.")
        for actor in self.dose_isosurface_SmartVolumeMapper(); volume_mapper.SetInputData(vtk_volume_image)
            self.volume_actors:
            self.ren.RemoveActor(actor)
        self.dose_isosurface_actors.actor = vtkVolume(); self.volume_actor.SetMapper(volume_mapper); self.volume_actorclear()

    def _update_dose_isosurfaces(self, dose_volume_full_crs:.SetProperty(self.volume_property)
            self.ren.AddVolume(self.volume_actor Optional[np.ndarray], # Dose is (cols,rows,slices)
                                 image_properties: Optional[Dict],); logger.info("DICOM volume actor added.")
        except Exception as e_vol: logger.error(f"
                                 isovalues_list: Optional[List[float]] = None):
        logger.info("Error creating DICOM volume actor: {e_vol}", exc_info=True)

        if tumor_mask_full_Updating dose isosurfaces...")
        self._clear_dose_isosurfaces()

        if dose_volume_fullzyx is not None and np.any(tumor_mask_full_zyx):
            self.tumor_crs is None or image_properties is None or not isovalues_list:
            logger.info("_actor = self._create_surface_actor_from_mask(tumor_mask_full_zyx,No dose volume, properties, or isovalues provided. Isosurfaces cleared or not generated.")
            if self.v image_properties, color=(1.0, 0.0, 0.0), opacity=0.4)
            if self.tumor_actor: self.ren.AddActor(self.tumor_actor); logger.infotkWidget.GetRenderWindow(): self.vtkWidget.GetRenderWindow().Render()
            return

        try:
            logger.debug(f"Dose volume for isosurfaces shape (c,r,s): {dose("Tumor mask actor added.")

        if oar_masks_full_zyx:
            oar_colors = [(0,1,0), (0,0,1), (1,1,0), (_volume_full_crs.shape}, dtype: {dose_volume_full_crs.dtype}")
            dose_volume_full_zyx = np.transpose(dose_volume_full_crs, (2,0,1,1), (1,0,1), (0.5,0.5,0), 1, 0)).astype(np.float32) # Transpose to (s,r,c (0,0.5,0.5), (0.5,0,0.5), (0.8)
            dose_vtk_image = self._numpy_to_vtkimage(dose_volume_full_,0.6,0.4), (0.4,0.8,0.6)]
            zyx, image_properties)

            colors_vtk_dose = [
                (1,0,0, 0color_idx = 0
            for name, oar_mask_zyx_data in oar_masks_full_zyx.items():
                if oar_mask_zyx_data is None or not np.any(.3), (0,1,0, 0.3), (0,0,1, 0.3), (1,1,0, 0.3), 
                (0,1,1, 0.3oar_mask_zyx_data): continue
                oar_actor = self._create_surface_actor_from_mask(oar_mask_zyx_data, image_properties, color=oar_colors), (1,0,1, 0.3), (1,0.5,0, 0.[color_idx % len(oar_colors)], opacity=0.25)
                if oar3), (0.5,1,0, 0.3),
                (0,0.5,1, 0.3), (0.5,0,1, 0.3) 
_actor: self.ren.AddActor(oar_actor); self.oar_actors[name] = oar_actor; logger.info(f"OAR actor for '{name}' added."); color_idx += 1            ] 

            for i, value in enumerate(isovalues_list):
                if not isinstance(value, (int, float)): logger.warning(f"Skipping invalid isovalue: {value}"); continue
                logger.
        
        self.ren.ResetCamera(); self.ren.ResetCameraClippingRange()
        if self.vtkWidget.GetRenderWindow(): self.vtkWidget.GetRenderWindow().Render()
        logger.info("3debug(f"Creating isosurface for value: {value} Gy")
                contour_filter = vtkMarchingCubes()
                contour_filter.SetInputData(dose_vtk_image)
                contour_filter.SetValueD View updated (Volume, Tumor, OARs).")

    def _planner_coords_to_patient(0, value) 
                contour_filter.Update()
                if contour_filter.GetOutput() is None_world_coords(self, planner_coords_crs: np.ndarray) -> Optional[np.ndarray]:
        """
        Converts planner grid coordinates (cols,rows,slices indices) to patient world coordinates (LPS mm).
         or contour_filter.GetOutput().GetNumberOfPoints() == 0:
                    logger.info(f"No geometry for isovalue {value} Gy. Skipping."); continue
                mapper = vtkPolyDataMapper(); mapper.SetRequires self.image_properties_for_viz to be set.
        planner_coords_crs: (col_InputConnection(contour_filter.GetOutputPort()); mapper.ScalarVisibilityOff()
                actor = vtkActoridx, row_idx, slice_idx)
        """
        if self.image_properties_for_(); actor.SetMapper(mapper)
                color_def = colors_vtk_dose[i % len(colors_viz is None:
            logger.error("Cannot convert planner to world coords: image_properties_for_viz not setvtk_dose)]
                actor.GetProperty().SetColor(color_def[0], color_def[1], color.")
            return None
        
        col_idx, row_idx, slice_idx = planner_coords_def[2])
                actor.GetProperty().SetOpacity(color_def[3]) 
                _crs

        # Spacing: image_properties['pixel_spacing'] is [row_spacing, col_spacingself.ren.AddActor(actor); self.dose_isosurface_actors.append(actor)
                ]
        col_spacing = self.image_properties_for_viz.get('pixel_spacing', [logger.info(f"Added isosurface actor for {value} Gy.")
        except Exception as e:
1.0,1.0])[1]
        row_spacing = self.image_properties_for_            logger.error(f"Error creating dose isosurfaces: {e}", exc_info=True)

viz.get('pixel_spacing', [1.0,1.0])[0]
        slice_th        if self.vtkWidget.GetRenderWindow():
            self.ren.ResetCameraClippingRange()
k = self.image_properties_for_viz.get('slice_thickness', 1.0)
            self.vtkWidget.GetRenderWindow().Render()
        logger.info("Dose isosurfaces update process finished.")        
        # Image coordinate system for VTK/ITK: x=cols, y=rows, z=slices

    def _planner_coords_to_patient_world_coords(self, planner_coords_crs: np.ndarray
        # Scaled image coordinates (relative to image origin, distance along image axes)
        img_x_) -> Optional[np.ndarray]:
        """
        Converts planner grid coordinates (cols,rows,slices indicesdist = col_idx * col_spacing
        img_y_dist = row_idx * row_spacing
        img) to patient world coordinates (LPS mm).
        Requires self.image_properties_for_viz to be set._z_dist = slice_idx * slice_thk
        
        # Orientation matrix from image_properties
        """
        if self.image_properties_for_viz is None:
            logger.error("Cannot convert planner to world coords: image_properties_for_viz not set.")
            return None
        
: columns are patient axes for image X,Y,Z
        # P_patient = OrientationMatrix @ [img_x_dist, img_y_dist, img_z_dist]^T + Origin_patient
        orientation_matrix        col_idx, row_idx, slice_idx = planner_coords_crs # These are 0-based = np.array(self.image_properties_for_viz.get('orientation_matrix_3x3', np indices

        # Spacing from image_properties: [row_spacing, col_spacing], slice_thickness
.eye(3)))
        origin_patient_lps = np.array(self.image_properties_for_viz        # For world coord calculation: col_spacing (x), row_spacing (y), slice_spacing (z)
.get('origin', [0.0,0.0,0.0])) # LPS
        
        #        col_spacing = self.image_properties_for_viz.get('pixel_spacing', [1.0,1.0])[1]
        row_spacing = self.image_properties_for_viz. Vector of distances along image axes
        dist_vec_img_axes = np.array([img_x_distget('pixel_spacing', [1.0,1.0])[0]
        # Slice spacing calculation needs to, img_y_dist, img_z_dist])
        
        # Transform to patient LPS coordinates
        world_coords_lps = orientation_matrix @ dist_vec_img_axes + origin_patient_lps be robust. If slices are contiguous, it's SliceThickness.
        # If there are gaps, it'
        
        return world_coords_lps


    def display_beams(self, beam_viz_s the distance between ImagePositionPatient[2] of adjacent slices.
        # For simplicity, assuming contiguous for now.
data: Optional[Dict[str, Any]]):
        self._clear_beam_visualization()
        if        slice_spacing = self.image_properties_for_viz.get('slice_thickness', 1. beam_viz_data is None or self.image_properties_for_viz is None:
            logger.info("No beam visualization data or image properties. Beams not displayed.")
            if self.vtkWidget.GetRenderWindow0) 
        
        # Voxel coordinates in the image frame, scaled by spacing
        # These are displacements(): self.vtkWidget.GetRenderWindow().Render()
            return

        beam_directions_planner = beam_viz from the origin of the first voxel in each image axis direction
        img_x_disp = col_idx * col_data.get("beam_directions", []) # These are unit vectors in planner's CRS
        beam__spacing
        img_y_disp = row_idx * row_spacing
        img_z_dispweights = beam_viz_data.get("beam_weights", np.array([]))
        # source_positions_planner = slice_idx * slice_spacing # This assumes slice_idx corresponds to Z-depth step
        
         are (col,row,slice) indices from QRadPlan3D
        source_positions_planner_crs_scaled_img_coords_vec = np.array([img_x_disp, img_y_disp, img_list = beam_viz_data.get("source_positions_planner_coords", []) 
        isocenter_plannerz_disp], dtype=float)
        
        # Orientation matrix (image axes in patient coordinate system)
        #_crs = np.array(beam_viz_data.get("isocenter_planner_coords", [0, Columns are [Ximg_in_Pat, Yimg_in_Pat, Zimg_in_Pat]
0,0])) # (col,row,slice) indices

        isocenter_world_lps =        orientation_matrix = np.array(self.image_properties_for_viz.get('orientation_matrix_3 self._planner_coords_to_patient_world_coords(isocenter_planner_crs)
        if isocx3', np.eye(3)))
        origin_patient_lps = np.array(self.image_propertiesenter_world_lps is None:
            logger.error("Failed to convert isocenter to world coordinates_for_viz.get('origin', [0.0,0.0,0.0])) # LPS of. Cannot display beams.")
            return

        logger.info(f"Displaying beams. Isocenter (world first voxel center
        
        # P_world = Origin_LPS + OrientationMatrix @ ScaledImageCoords
        world LPS): {isocenter_world_lps}. Num beams: {len(beam_directions_planner)}")
_coords_lps = origin_patient_lps + orientation_matrix @ scaled_img_coords_vec
        max_weight = np.max(beam_weights) if beam_weights.size > 0 and np.max(beam_weights) > 1e-6 else 1.0

        for i, direction_        
        logger.debug(f"Planner coords (c,r,s): {planner_coords_crsvec_planner in enumerate(beam_directions_planner):
            weight = beam_weights[i] if i} -> World coords (LPS): {world_coords_lps}")
        return world_coords_lps

    def display_beams(self, beam_viz_data: Optional[Dict[str, Any]]): < len(beam_weights) else 0
            if weight < 0.01: continue # Only display active beams

            source_pos_planner_crs = np.array(source_positions_planner_crs
        self._clear_beam_visualization()
        if beam_viz_data is None or self.image_list[i])
            source_pos_world_lps = self._planner_coords_to__properties_for_viz is None:
            logger.info("No beam viz data or image_properties. Bepatient_world_coords(source_pos_planner_crs)
            if source_pos_world_lams not displayed.")
            if self.vtkWidget.GetRenderWindow(): self.vtkWidget.GetRenderWindow().Render()ps is None:
                logger.warning(f"Could not convert source for beam {i} to world. Skipping
            return

        beam_directions_planner = beam_viz_data.get("beam_directions", []) # Unit vectors in planner's CRS (usually same as patient if no 4D deformation)
        beam_weights = beam."); continue

            # Beam direction in planner CRS needs to be transformed to world LPS direction
            # If planner CRS_viz_data.get("beam_weights", np.array([]))
        # source_positions_planner is aligned with image CRS (no rotation between them),
            # then transforming the direction vector by the orientation matrix should_coords are in planner's voxel indices (col,row,slice)
        source_positions_planner_vox work.
            # Planner direction (dx_c, dy_r, dz_s) -> World direction (dx_ = beam_viz_data.get("source_positions_planner_coords", []) 
        isocenter_planner_vox = np.array(beam_viz_data.get("isocenter_planner_coords", [0.l, dy_p, dz_s_pat)
            orientation_matrix = np.array(self.image_properties_for_viz.get('orientation_matrix_3x3', np.eye(3)))
            direction0,0.0,0.0])) # Voxel indices

        isocenter_world_lps = self._planner_coords_to_patient_world_coords(isocenter_planner_vox)
        if_vec_world_lps = orientation_matrix @ np.array(direction_vec_planner)
            direction isocenter_world_lps is None:
            logger.error("Failed to convert isocenter to world coordinates_vec_world_lps /= (np.linalg.norm(direction_vec_world_lps) + for beam display."); return

        logger.info(f"Displaying beams. Isocenter (world LPS): {isoc 1e-9) # Re-normalize

            # For VTK line, we need two points in world coordinates.
enter_world_lps}. Num beams: {len(beam_directions_planner)}")
        max_weight = np.            # Source is one. Target can be isocenter, or a point further along the beam.
            # Letmax(beam_weights) if beam_weights.size > 0 and np.max(beam_weights) > 1's use source and isocenter for the line.
            line_source = vtkLineSource()
            e-6 else 1.0

        for i, direction_vec_planner in enumerate(beam_directionsline_source.SetPoint1(source_pos_world_lps[0], source_pos_world_planner):
            weight = beam_weights[i] if i < len(beam_weights) else _lps[1], source_pos_world_lps[2])
            line_source.SetPoint2(isocenter_world_lps[0], isocenter_world_lps[10
            if weight < 0.01: continue # Threshold for displaying beam

            source_pos_planner_], isocenter_world_lps[2])
            line_source.Update()

            mapper =vox_np = np.array(source_positions_planner_vox[i])
            source_pos_world vtkPolyDataMapper(); mapper.SetInputConnection(line_source.GetOutputPort())
            actor = vtk_lps = self._planner_coords_to_patient_world_coords(source_pos_planner_vox_npActor(); actor.SetMapper(mapper)
            
            intensity = weight / max_weight
            actor.GetProperty().)
            if source_pos_world_lps is None:
                logger.warning(f"CouldSetColor(intensity, 1.0 - intensity, 0) 
            actor.GetProperty().Set not convert source for beam {i} to world. Skipping."); continue

            line_source = vtkLineSource()
            LineWidth(1.0 + intensity * 2) # Max line width 3 for max weight
            actor.GetProperty().line_source.SetPoint1(source_pos_world_lps[0], source_pos_world_lps[1], source_pos_world_lps[2])
            line_source.SetSetOpacity(0.6 + intensity * 0.4) # Max opacity 1.0

            selfPoint2(isocenter_world_lps[0], isocenter_world_lps[1.ren.AddActor(actor)
            self.beam_visualization_actors.append(actor)
            ], isocenter_world_lps[2])
            line_source.Update()

            mapper =logger.debug(f"Added beam actor {i}: src_world={source_pos_world_lps}, iso_ vtkPolyDataMapper(); mapper.SetInputConnection(line_source.GetOutputPort())
            actor =world={isocenter_world_lps}, weight={weight:.2f}")

        if self.vtkWidget.Get vtkActor(); actor.SetMapper(mapper)
            
            intensity = weight / max_weight
            actorRenderWindow(): self.vtkWidget.GetRenderWindow().Render()
        logger.info("Beam visualization updated.").GetProperty().SetColor(intensity, 1.0 - intensity, 0.0) 
            actor.

    def _clear_beam_visualization(self):
        logger.debug(f"Clearing {len(selfGetProperty().SetLineWidth(1.0 + intensity * 3.0) 
            actor.GetProperty().SetOpacity.beam_visualization_actors)} beam visualization actors.")
        for actor in self.beam_visualization_actors:(0.6 + intensity * 0.4) # Make active beams more opaque

            self.ren.Add
            self.ren.RemoveActor(actor)
        self.beam_visualization_actors.clear()
Actor(actor); self.beam_visualization_actors.append(actor)
            logger.debug(f"    
    def _clear_oar_actors(self):
        logger.debug(f"Clearing {len(self.oar_actors)} OAR actors.")
        for actor in self.oar_actorsBeam {i}: src_world={source_pos_world_lps}, iso_world={isocenter_world.values():
            self.ren.RemoveActor(actor)
        self.oar_actors.clear_lps}, weight={weight:.2f}")

        if self.vtkWidget.GetRenderWindow(): self()

    def _clear_dose_isosurfaces(self):
        logger.debug(f"Cle.vtkWidget.GetRenderWindow().Render()
        logger.info("Beam visualization updated.")

    def _cleararing {len(self.dose_isosurface_actors)} dose isosurface actors.")
        for actor in self.dose_beam_visualization(self):
        logger.debug(f"Clearing {len(self.beam_visualization_isosurface_actors:
            self.ren.RemoveActor(actor)
        self.dose_isos_actors)} beam visualization actors.")
        for actor in self.beam_visualization_actors:
            self.urface_actors.clear()

    def clear_view(self):
        logger.info("Clearing ren.RemoveActor(actor)
        self.beam_visualization_actors.clear()
    
    def3D viewer (volume, tumor, OARs, beams, and dose isosurfaces).")
        if clear_view(self):
        logger.info("Clearing 3D viewer (volume, tumor, O self.volume_actor is not None: self.ren.RemoveVolume(self.volume_actor); self.volume_actor = None
        if self.tumor_actor is not None: self.ren.RemoveActor(ARs, beams, and dose isosurfaces).")
        if self.volume_actor is not None: self.ren.RemoveVolume(self.volume_actor); self.volume_actor = None
        if selfself.tumor_actor); self.tumor_actor = None
        self._clear_oar_actors().tumor_actor is not None: self.ren.RemoveActor(self.tumor_actor); self.tumor
        self._clear_dose_isosurfaces() 
        self._clear_beam_visualization() _actor = None
        self._clear_oar_actors()
        self._clear_dose_isos
        if self.vtkWidget.GetRenderWindow(): self.vtkWidget.GetRenderWindow().Render()

urfaces() 
        self._clear_beam_visualization() 
        if self.vtkWidget.Getif __name__ == '__main__':
    app = QApplication(sys.argv)
    logging.basicConfig(RenderWindow(): self.vtkWidget.GetRenderWindow().Render()

if __name__ == '__main__':
level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(    app = QApplication(sys.argv)
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    viewer3d = DicomViewer3DWidget()
    vol_shape_zyasctime)s - %(name)s - %(levelname)s - %(message)s')
    viewer3dx = (50, 64, 64) 
    dummy_volume_data_zy = DicomViewer3DWidget()
    vol_shape_zyx = (50, 64x = np.random.randint(-1000, 2000, size=vol_shape, 64) 
    dummy_volume_data_zyx = np.random.randint(-1_zyx, dtype=np.int16)
    dummy_tumor_mask_zyx = np.zeros(000, 2000, size=vol_shape_zyx, dtype=np.intvol_shape_zyx, dtype=bool)
    dummy_tumor_mask_zyx[2016)
    dummy_tumor_mask_zyx = np.zeros(vol_shape_zyx, dtype:30, 25:35, 25:35] = True
    dummy_=bool); dummy_tumor_mask_zyx[20:30, 25:35oar_masks_zyx = { "OAR1_Heart": np.zeros(vol_shape_, 25:35] = True
    dummy_oar_masks_zyx = {"Ozyx, dtype=bool), "OAR2_Lung": np.zeros(vol_shape_zyxAR1": np.zeros(vol_shape_zyx, dtype=bool)}; dummy_oar_masks_zyx, dtype=bool)}
    dummy_oar_masks_zyx["OAR1_Heart"][1["OAR1"][15:25, 10:20, 40:505:25, 10:20, 40:50] = True
    dummy_oar_masks_zyx["OAR2_Lung"][30:40, 40:55,] = True
    
    # Define image_properties consistent with _numpy_to_vtkimage and _planner_coords_to_patient_world_coords
    # PixelSpacing: [row, col]
    # Origin: LPS 10:25] = True
    
    # Example image_properties - crucial for correct VTK visualization
    dummy_image_properties = {
        'pixel_spacing': [0.8, 0.9], of first voxel center
    # Orientation: Columns are patient axes for image X,Y,Z directions
    dummy # [row_spacing_mm, col_spacing_mm]
        'slice_thickness': 2.5,_image_properties = {
        'pixel_spacing': [0.8, 0.9], # row_sp=0.8, col_sp=0.9
        'slice_thickness': 2.      # slice_thickness_mm
        'origin': [-128*0.9, -125,
        'origin': [-100.0, -120.0, -80.0],8*0.8, -50*2.5], # LPS origin of the center of the first voxel # Example LPS origin
        'orientation_matrix_3x3': np.eye(3).tolist() # Standard axial for
        'orientation_matrix_3x3': np.eye(3).tolist() # Simplest: identity orientation simplicity
    }
    
    viewer3d.update_volume(dummy_volume_data_zyx, dummy_
    }
    
    viewer3d.update_volume(dummy_volume_data_zyx,image_properties, 
                           tumor_mask_full_zyx=dummy_tumor_mask_zyx, 
 dummy_image_properties, 
                           tumor_mask_full_zyx=dummy_tumor_mask_zy                           oar_masks_full_zyx=dummy_oar_masks_zyx)
    
x,
                           oar_masks_full_zyx=dummy_oar_masks_zyx)
    
    # Test dose isosurfaces
    dose_shape_crs = (vol_shape_zy    # Test dose
    dose_shape_crs = (vol_shape_zyx[2], vol_shapex[2], vol_shape_zyx[1], vol_shape_zyx[0]) # (_zyx[1], vol_shape_zyx[0]) # (cols,rows,slices)
cols,rows,slices)
    dummy_dose_crs = np.zeros(dose_shape_crs, dtype=np    dummy_dose_crs = np.zeros(dose_shape_crs, dtype=np.float32)
    cx,cy,cz = [d//2 for d in dose_shape_crs]; radius_dose = min.float32)
    cx,cy,cz = [d//2 for d in dose_shape_crs]; radius_dose = min(cx,cy,cz)//2
    x,y,z = np.og(cx,cy,cz)//2
    x,y,z = np.ogrid[:dose_shape_crs[rid[:dose_shape_crs[0], :dose_shape_crs[1], :dose_shape_0], :dose_shape_crs[1], :dose_shape_crs[2]]
    mask_sphere_crscrs[2]]
    mask_sphere_crs = (x-cx)**2 + (y-cy)** = (x-cx)**2 + (y-cy)**2 + (z-cz)**2 <= radius_2 + (z-cz)**2 <= radius_dose**2
    dummy_dose_crs[mask_dose**2
    dummy_dose_crs[mask_sphere_crs] = 60.0; dummysphere_crs] = 60.0; dummy_dose_crs += np.random.rand(*dose_dose_crs += np.random.rand(*dose_shape_crs)*10
    viewer3d_shape_crs)*10
    viewer3d._update_dose_isosurfaces(dummy_dose._update_dose_isosurfaces(dummy_dose_crs, dummy_image_properties, isovalues_list_crs, dummy_image_properties, isovalues_list=[20.0, 40.0,=[20.0, 40.0, 55.0])

    # Test beams ( 55.0])

    # Test beam visualization
    # Planner coords (c,r,s) for isocenter and sources
    isocenter_planner_crs = np.array([dose_shape_crsassuming planner coords are voxel indices in CRS frame)
    # Planner grid size for this test: (cols, rows, slices[0]//2, dose_shape_crs[1]//2, dose_shape_crs[2]//2])
    )
    planner_grid_size_crs = dose_shape_crs 
    isocenter_plannerbeam_viz_test_data = {
        "beam_directions": [(1,0,0), (_vox_test = np.array([planner_grid_size_crs[0]//2, planner_grid_size_0,1,0), (-1,0,0), (0,-1,0)], # In planner'crs[1]//2, planner_grid_size_crs[2]//2])
    
    beam_vizs CRS
        "beam_weights": np.array([1.0, 0.8, 0.5_test_data = {
        "beam_directions": [(1,0,0), (0,1,, 0.2]),
        "source_positions_planner_coords": [ # (col, row, slice0), (-1,0,0), (0,-1,0), (0.707, ) indices
            (isocenter_planner_crs - np.array([50,0,0])).0.707, 0)], # Example directions
        "beam_weights": np.array([1.0,tolist(),
            (isocenter_planner_crs - np.array([0,50,0])). 0.8, 0.6, 0.4, 0.9]),
        "sourcetolist(),
            (isocenter_planner_crs - np.array([-50,0,0])).tolist(),
            (isocenter_planner_crs - np.array([0,-50,0])).tolist_positions_planner_coords": [ # Sources far out along negative direction from isocenter
            (isocenter_(),
        ],
        "isocenter_planner_coords": isocenter_planner_crs.tolistplanner_vox_test - np.array([1,0,0])*100).tolist(),
            (is()
    }
    viewer3d.display_beams(beam_viz_test_data)
    ocenter_planner_vox_test - np.array([0,1,0])*100).tolist
    viewer3d.resize(800, 600)
    viewer3d.show()(),
            (isocenter_planner_vox_test - np.array([-1,0,0])*
    sys.exit(app.exec_())
