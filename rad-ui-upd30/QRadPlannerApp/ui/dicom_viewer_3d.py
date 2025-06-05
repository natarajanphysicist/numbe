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
    vtkImageActor, # Added for 2D slice test
)
from vtkmodules.vtkImagingCore import vtkImageShiftScale, vtkImageReslice # Added vtkImageReslice for 2D slice test
# vtkImageMapper is usually implicitly handled by vtkImageActor or needs vtkDataSetMapper for some scenarios.
# For vtkImageActor, a vtkImageSliceMapper can be used, or it might default.
# Let's try with vtkImageActor's default capabilities first.
from vtkmodules.vtkFiltersCore import vtkMarchingCubes
from vtkmodules.vtkFiltersGeneral import vtkDiscreteMarchingCubes # Good for binary masks
from vtkmodules.vtkFiltersSources import vtkLineSource, vtkConeSource, vtkCylinderSource # For beam viz
from vtkmodules.vtkCommonMath import vtkMatrix4x4 # For transformations (not directly used in current _planner_coords_to_patient_world_coords but good to have if needed)


from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper 

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util import numpy_support
from typing import Optional, Dict, List, Tuple, Any
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

        self.volume_actor: Optional[vtkVolume] = None
        self.tumor_actor: Optional[vtkActor] = None
        self.oar_actors: Dict[str, vtkActor] = {}
        self.dose_isosurface_actors: List[vtkActor] = []
        self.beam_visualization_actors: List[vtkActor] = []
        self.image_properties_for_viz: Optional[Dict] = None # Store for transformations

        self.vtkWidget.Initialize()
        logger.info("DicomViewer3DWidget initialized.")

    def _numpy_to_vtkimage(self, np_array_s_r_c: np.ndarray, image_properties: Optional[Dict] = None) -> vtkImageData:
        """
        Converts a NumPy array (slices, rows, cols) to vtkImageData.
        Sets spacing and origin if properties are provided.
        VTK ImageData expects dimensions (width, height, depth) i.e. (cols, rows, slices).
        np_array_s_r_c.ravel(order='F') should produce data in VTK's xyz order.
        """
        vtk_image = vtkImageData()
        depth_s, height_r, width_c = np_array_s_r_c.shape # slices, rows, cols
        vtk_image.SetDimensions(width_c, height_r, depth_s) # VTK: width, height, depth

        if image_properties:
            # image_properties['pixel_spacing'] is [row_spacing, col_spacing]
            # image_properties['slice_thickness'] is z_spacing
            # VTK spacing is (spacing_x, spacing_y, spacing_z) -> (col_spacing, row_spacing, slice_thk)
            spacing_x_vtk = image_properties.get('pixel_spacing', [1.0, 1.0])[1] # Col spacing
            spacing_y_vtk = image_properties.get('pixel_spacing', [1.0, 1.0])[0] # Row spacing
            spacing_z_vtk = image_properties.get('slice_thickness', 1.0)
            vtk_image.SetSpacing(spacing_x_vtk, spacing_y_vtk, spacing_z_vtk)
            
            # Origin is (x,y,z) of the first voxel's corner (or center depending on DICOM interpretation).
            # For ITK/VTK, typically corner. DICOM ImagePositionPatient is center of first voxel.
            # This needs careful alignment if absolute patient coordinates are critical.
            # For now, assume image_properties['origin'] is directly usable.
            origin_pat = image_properties.get('origin', [0.0, 0.0, 0.0])
            vtk_image.SetOrigin(origin_pat[0], origin_pat[1], origin_pat[2])
        else: 
            vtk_image.SetSpacing(1.0,1.0,1.0); vtk_image.SetOrigin(0.0,0.0,0.0)

        vtk_array = numpy_support.numpy_to_vtk(num_array=np_array_s_r_c.ravel(order='F'), deep=True)
        vtk_image.GetPointData().SetScalars(vtk_array)
        return vtk_image
    
    def _create_surface_actor_from_mask(self, mask_data_zyx: np.ndarray,
                                        image_properties: Dict, 
                                        color: Tuple[float, float, float],
                                        opacity: float = 0.3) -> Optional[vtkActor]:
        if mask_data_zyx is None or not np.any(mask_data_zyx):
            return None
        try:
            vtk_mask_image = self._numpy_to_vtkimage(mask_data_zyx.astype(np.uint8), image_properties)
            mc = vtkDiscreteMarchingCubes()
            mc.SetInputData(vtk_mask_image)
            mc.SetValue(0, 1)
            mc.Update()
            if mc.GetOutput() is None or mc.GetOutput().GetNumberOfPoints() == 0:
                logger.debug("No surface generated by marching cubes for a mask.")
                return None
            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(mc.GetOutputPort())
            mapper.ScalarVisibilityOff()
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(color[0], color[1], color[2])
            actor.GetProperty().SetOpacity(opacity)
            return actor
        except Exception as e:
            logger.error(f"Error creating surface actor from mask: {e}", exc_info=True)
            return None

    def update_volume(self, volume_data_full_zyx: Optional[np.ndarray],    # (s,r,c)
                      image_properties: Optional[Dict],
                      tumor_mask_full_zyx: Optional[np.ndarray] = None,   # (s,r,c)
                      oar_masks_full_zyx: Optional[Dict[str, np.ndarray]] = None): # (s,r,c)
        
        self.image_properties_for_viz = image_properties # Store for beam viz and coord transforms

        logger.info("Updating 3D Viewer (Volume, Tumor, OARs)...")
        if self.volume_actor is not None: self.ren.RemoveVolume(self.volume_actor); self.volume_actor = None
        if self.tumor_actor is not None: self.ren.RemoveActor(self.tumor_actor); self.tumor_actor = None
        self._clear_oar_actors()
        self._clear_beam_visualization() # Clear beams when volume changes

        if volume_data_full_zyx is None or image_properties is None:
            logger.info("No volume data or properties to display in 3D view.")
            if self.vtkWidget.GetRenderWindow(): # Ensure render window exists
                self.ren.ResetCamera()
                self.vtkWidget.GetRenderWindow().Render()
            return

        try: 
            logger.info(f"3D View: Original full volume data range: {volume_data_full_zyx.min():.2f} to {volume_data_full_zyx.max():.2f}")
            min_hu_display = -1024.0  # Typical lower bound for CT visualization
            max_hu_display = 3000.0   # Typical upper bound for CT visualization (dense bone)
            volume_data_clipped_zyx = np.clip(volume_data_full_zyx, min_hu_display, max_hu_display)
            logger.info(f"3D View: Clipped volume data range for display: {volume_data_clipped_zyx.min():.2f} to {volume_data_clipped_zyx.max():.2f}")

            vtk_volume_image = self._numpy_to_vtkimage(volume_data_clipped_zyx.astype(np.float32), image_properties)
            try:
                scalar_range = vtk_volume_image.GetScalarRange()
                logger.info(f"3D View: VTK Volume Image Scalar Range: {scalar_range[0]:.2f} to {scalar_range[1]:.2f}")
            except Exception as e_sr:
                logger.warning(f"3D View: Could not get scalar range from vtk_volume_image: {e_sr}")

        # --- START OF VTK IMAGE PLANE TEST ---
        logger.info("3D View Test: Attempting to display a central 2D slice as vtkImageActor.")
        if volume_data_clipped_zyx is not None and volume_data_clipped_zyx.ndim == 3 and volume_data_clipped_zyx.shape[0] > 0:
            central_slice_idx = volume_data_clipped_zyx.shape[0] // 2
            slice_2d_data_zyx = volume_data_clipped_zyx[central_slice_idx, :, :]

            # Create a new 2D vtkImageData for this slice
            vtk_slice_image = vtkImageData()
            rows, cols = slice_2d_data_zyx.shape
            vtk_slice_image.SetDimensions(cols, rows, 1) # depth is 1 for a 2D image

            if image_properties: # Use original image_properties for spacing and origin
                col_spacing = image_properties.get('pixel_spacing', [1.0,1.0])[1]
                row_spacing = image_properties.get('pixel_spacing', [1.0,1.0])[0]
                vtk_slice_image.SetSpacing(col_spacing, row_spacing, 1.0) # X, Y spacing, Z spacing can be nominal

                slice_vtk_array = numpy_support.numpy_to_vtk(num_array=slice_2d_data_zyx.ravel(order='F'), deep=True)
                vtk_slice_image.GetPointData().SetScalars(slice_vtk_array)

                self.test_slice_actor = vtkImageActor()
                self.test_slice_actor.GetMapper().SetInputData(vtk_slice_image)
                self.test_slice_actor.SetDisplayExtent(0, cols-1, 0, rows-1, 0, 0)

                slice_min = np.min(slice_2d_data_zyx)
                slice_max = np.max(slice_2d_data_zyx)
                self.test_slice_actor.GetProperty().SetColorWindow(slice_max - slice_min)
                self.test_slice_actor.GetProperty().SetColorLevel((slice_max + slice_min) / 2.0)

                self.ren.AddActor(self.test_slice_actor)
                logger.info(f"3D View Test: Added vtkImageActor for slice {central_slice_idx}. Data range: {slice_min:.2f}-{slice_max:.2f}")
                try:
                    slice_actor_bounds = self.test_slice_actor.GetBounds()
                    logger.info(f"3D View Test: Slice actor bounds: {slice_actor_bounds}")
                except Exception as e_sab:
                    logger.warning(f"3D View Test: Could not get slice_actor bounds: {e_sab}")
            else:
                logger.warning("3D View Test: No image_properties, cannot set spacing/origin for test slice actor.")
        else:
            logger.warning("3D View Test: Clipped volume data not suitable for extracting a 2D slice.")
        # --- END OF VTK IMAGE PLANE TEST ---

            color_func = vtkColorTransferFunction(); opacity_func = vtkPiecewiseFunction()

            logger.info("3D View: Applying AGGRESSIVE simplified color transfer function.")
            color_func.AddRGBPoint(min_hu_display, 0.2, 0.2, 0.2) # Dark gray
            color_func.AddRGBPoint(max_hu_display, 0.9, 0.9, 0.9) # Light gray

            logger.info("3D View: Applying AGGRESSIVE simplified opacity transfer function for basic visibility test.")
            opacity_func.AddPoint(min_hu_display, 0.0)  # min_hu_display is -1024.0
            opacity_func.AddPoint(-300.0, 0.0)          # Air-like regions mostly transparent
            opacity_func.AddPoint(-299.0, 0.25)         # Sharp step to make soft tissues visible
            opacity_func.AddPoint(max_hu_display, 0.25) # max_hu_display is 3000.0. All tissue/bone has some opacity.

            self.volume_property = vtkVolumeProperty(); self.volume_property.SetColor(color_func); self.volume_property.SetScalarOpacity(opacity_func)
            self.volume_property.SetInterpolationTypeToLinear();
            logger.info("3D View: Turning OFF shading for basic visibility test.")
            self.volume_property.ShadeOff()
            self.volume_property.SetAmbient(0.3); self.volume_property.SetDiffuse(0.7); self.volume_property.SetSpecular(0.2); self.volume_property.SetSpecularPower(10.0)

            # Temporarily disable 3D volume rendering by not re-creating/adding self.volume_actor
            # volume_mapper = vtkSmartVolumeMapper(); volume_mapper.SetInputData(vtk_volume_image) # vtk_volume_image is 3D
            # self.volume_actor = vtkVolume(); self.volume_actor.SetMapper(volume_mapper); self.volume_actor.SetProperty(self.volume_property)
            # self.ren.AddVolume(self.volume_actor) # << THIS LINE IS EFFECTIVELY COMMENTED OUT
            # try:
            #     bounds = self.volume_actor.GetBounds()
            #     logger.info(f"3D View: Volume actor bounds: {bounds}")
            # except Exception as e_bounds:
            #     logger.warning(f"3D View: Could not get volume actor bounds: {e_bounds}")
            # logger.info("DICOM volume actor added.") # This log might be misleading now
            logger.info("3D View: Main volume rendering temporarily bypassed for 2D slice test.")

        except Exception as e_vol:
            logger.error(f"Error during 3D view update (possibly volume or slice actor): {e_vol}", exc_info=True)
            if self.volume_actor: self.ren.RemoveVolume(self.volume_actor); self.volume_actor = None


        if tumor_mask_full_zyx is not None and np.any(tumor_mask_full_zyx):
            self.tumor_actor = self._create_surface_actor_from_mask(
                tumor_mask_full_zyx, image_properties, color=(1.0, 0.0, 0.0), opacity=0.4
            )
            if self.tumor_actor: self.ren.AddActor(self.tumor_actor); logger.info("Tumor mask actor added.")

        if oar_masks_full_zyx:
            oar_colors = [(0,1,0), (0,0,1), (1,1,0), (0,1,1), (1,0,1), (0.5,0.5,0), (0,0.5,0.5), (0.5,0,0.5)]
            color_idx = 0
            for name, oar_mask_zyx_data in oar_masks_full_zyx.items():
                if oar_mask_zyx_data is None or not np.any(oar_mask_zyx_data): continue
                oar_actor = self._create_surface_actor_from_mask(
                    oar_mask_zyx_data, image_properties,
                    color=oar_colors[color_idx % len(oar_colors)], opacity=0.25
                )
                if oar_actor:
                    self.ren.AddActor(oar_actor); self.oar_actors[name] = oar_actor
                    logger.info(f"OAR actor for '{name}' added."); color_idx += 1
        
        if self.vtkWidget.GetRenderWindow():
            self.ren.ResetCamera()
            self.ren.ResetCameraClippingRange()
            try:
                camera = self.ren.GetActiveCamera()
                if camera:
                    logger.info(f"3D View: Camera Position: {camera.GetPosition()}, FocalPoint: {camera.GetFocalPoint()}, ViewUp: {camera.GetViewUp()}, ClipRange: {camera.GetClippingRange()}, ViewAngle: {camera.GetViewAngle()}")
                else:
                    logger.warning("3D View: No active camera found to log parameters.")
            except Exception as e_cam:
                logger.warning(f"3D View: Could not get camera parameters: {e_cam}")
            self.vtkWidget.GetRenderWindow().Render()
        logger.info("3D View updated (Volume, Tumor, OARs, Test Slice).") # Updated log

    def _clear_oar_actors(self):
        logger.debug(f"Clearing {len(self.oar_actors)} OAR actors.")
        for actor in self.oar_actors.values():
            self.ren.RemoveActor(actor)
        self.oar_actors.clear()

    def _clear_dose_isosurfaces(self):
        logger.debug(f"Clearing {len(self.dose_isosurface_actors)} dose isosurface actors.")
        for actor in self.dose_isosurface_actors:
            self.ren.RemoveActor(actor)
        self.dose_isosurface_actors.clear()

    def _update_dose_isosurfaces(self, dose_volume_full_crs: Optional[np.ndarray], # Dose is (cols,rows,slices)
                                 image_properties: Optional[Dict],
                                 isovalues_list: Optional[List[float]] = None):
        logger.info("Updating dose isosurfaces...")
        self._clear_dose_isosurfaces()

        if dose_volume_full_crs is None or image_properties is None or not isovalues_list:
            logger.info("No dose volume, properties, or isovalues provided. Isosurfaces cleared or not generated.")
            if self.vtkWidget.GetRenderWindow(): self.vtkWidget.GetRenderWindow().Render()
            return

        try:
            logger.debug(f"Dose volume for isosurfaces shape (c,r,s): {dose_volume_full_crs.shape}, dtype: {dose_volume_full_crs.dtype}")
            dose_volume_full_zyx = np.transpose(dose_volume_full_crs, (2, 1, 0)).astype(np.float32) # Transpose to (s,r,c)
            dose_vtk_image = self._numpy_to_vtkimage(dose_volume_full_zyx, image_properties)

            colors_vtk_dose = [
                (1,0,0, 0.3), (0,1,0, 0.3), (0,0,1, 0.3), (1,1,0, 0.3),
                (0,1,1, 0.3), (1,0,1, 0.3), (1,0.5,0, 0.3), (0.5,1,0, 0.3),
                (0,0.5,1, 0.3), (0.5,0,1, 0.3) 
            ]

            for i, value in enumerate(isovalues_list):
                if not isinstance(value, (int, float)): logger.warning(f"Skipping invalid isovalue: {value}"); continue
                logger.debug(f"Creating isosurface for value: {value} Gy")
                contour_filter = vtkMarchingCubes()
                contour_filter.SetInputData(dose_vtk_image)
                contour_filter.SetValue(0, value)
                contour_filter.Update()
                if contour_filter.GetOutput() is None or contour_filter.GetOutput().GetNumberOfPoints() == 0:
                    logger.info(f"No geometry for isovalue {value} Gy. Skipping."); continue
                mapper = vtkPolyDataMapper(); mapper.SetInputConnection(contour_filter.GetOutputPort()); mapper.ScalarVisibilityOff()
                actor = vtkActor(); actor.SetMapper(mapper)
                color_def = colors_vtk_dose[i % len(colors_vtk_dose)]
                actor.GetProperty().SetColor(color_def[0], color_def[1], color_def[2])
                actor.GetProperty().SetOpacity(color_def[3]) 
                self.ren.AddActor(actor); self.dose_isosurface_actors.append(actor)
                logger.info(f"Added isosurface actor for {value} Gy.")
        except Exception as e:
            logger.error(f"Error creating dose isosurfaces: {e}", exc_info=True)

        if self.vtkWidget.GetRenderWindow():
            self.ren.ResetCameraClippingRange()
            self.vtkWidget.GetRenderWindow().Render()
        logger.info("Dose isosurfaces update process finished.")        

    def _planner_coords_to_patient_world_coords(self, planner_coords_crs: np.ndarray) -> Optional[np.ndarray]:
        """
        Converts planner grid coordinates (cols,rows,slices indices) to patient world coordinates (LPS mm).
        Requires self.image_properties_for_viz to be set.
        """
        if self.image_properties_for_viz is None:
            logger.error("Cannot convert planner to world coords: image_properties_for_viz not set.")
            return None
        
        col_idx, row_idx, slice_idx = planner_coords_crs # These are 0-based indices

        # Spacing from image_properties: [row_spacing, col_spacing], slice_thickness
        # For world coord calculation: col_spacing (x), row_spacing (y), slice_spacing (z)
        col_spacing = self.image_properties_for_viz.get('pixel_spacing', [1.0,1.0])[1]
        row_spacing = self.image_properties_for_viz.get('pixel_spacing', [1.0,1.0])[0]
        # Slice spacing calculation needs to be robust. If slices are contiguous, it's SliceThickness.
        # If there are gaps, it's the distance between ImagePositionPatient[2] of adjacent slices.
        # For simplicity, assuming contiguous for now.
        slice_spacing = self.image_properties_for_viz.get('slice_thickness', 1.0)
        
        # Voxel coordinates in the image frame, scaled by spacing
        # These are displacements from the origin of the first voxel in each image axis direction
        img_x_disp = col_idx * col_spacing
        img_y_disp = row_idx * row_spacing
        img_z_disp = slice_idx * slice_spacing # This assumes slice_idx corresponds to Z-depth step
        
        scaled_img_coords_vec = np.array([img_x_disp, img_y_disp, img_z_disp], dtype=float)
        
        # Orientation matrix (image axes in patient coordinate system)
        # Columns are [Ximg_in_Pat, Yimg_in_Pat, Zimg_in_Pat]
        orientation_matrix = np.array(self.image_properties_for_viz.get('orientation_matrix_3x3', np.eye(3)))
        origin_patient_lps = np.array(self.image_properties_for_viz.get('origin', [0.0,0.0,0.0])) # LPS of first voxel center
        
        # P_world = Origin_LPS + OrientationMatrix @ ScaledImageCoords
        world_coords_lps = origin_patient_lps + orientation_matrix @ scaled_img_coords_vec

        logger.debug(f"Planner coords (c,r,s): {planner_coords_crs} -> World coords (LPS): {world_coords_lps}")
        return world_coords_lps

    def display_beams(self, beam_viz_data: Optional[Dict[str, Any]]):
        self._clear_beam_visualization()
        if beam_viz_data is None or self.image_properties_for_viz is None:
            logger.info("No beam viz data or image_properties. Beams not displayed.")
            if self.vtkWidget.GetRenderWindow(): self.vtkWidget.GetRenderWindow().Render()
            return

        beam_directions_planner = beam_viz_data.get("beam_directions", []) # Unit vectors in planner's CRS (usually same as patient if no 4D deformation)
        beam_weights = beam_viz_data.get("beam_weights", np.array([]))
        # source_positions_planner_coords are in planner's voxel indices (col,row,slice)
        source_positions_planner_vox = beam_viz_data.get("source_positions_planner_coords", [])
        isocenter_planner_vox = np.array(beam_viz_data.get("isocenter_planner_coords", [0.0,0.0,0.0])) # Voxel indices

        isocenter_world_lps = self._planner_coords_to_patient_world_coords(isocenter_planner_vox)
        if isocenter_world_lps is None:
            logger.error("Failed to convert isocenter to world coordinates for beam display."); return

        logger.info(f"Displaying beams. Isocenter (world LPS): {isocenter_world_lps}. Num beams: {len(beam_directions_planner)}")
        max_weight = np.max(beam_weights) if beam_weights.size > 0 and np.max(beam_weights) > 1e-6 else 1.0

        for i, direction_vec_planner in enumerate(beam_directions_planner):
            weight = beam_weights[i] if i < len(beam_weights) else 0
            if weight < 0.01: continue # Threshold for displaying beam

            source_pos_planner_vox_np = np.array(source_positions_planner_vox[i])
            source_pos_world_lps = self._planner_coords_to_patient_world_coords(source_pos_planner_vox_np)
            if source_pos_world_lps is None:
                logger.warning(f"Could not convert source for beam {i} to world. Skipping."); continue

            line_source = vtkLineSource()
            line_source.SetPoint1(source_pos_world_lps[0], source_pos_world_lps[1], source_pos_world_lps[2])
            line_source.SetPoint2(isocenter_world_lps[0], isocenter_world_lps[1], isocenter_world_lps[2])
            line_source.Update()

            mapper = vtkPolyDataMapper(); mapper.SetInputConnection(line_source.GetOutputPort())
            actor = vtkActor(); actor.SetMapper(mapper)
            
            intensity = weight / max_weight
            actor.GetProperty().SetColor(intensity, 1.0 - intensity, 0.0)
            actor.GetProperty().SetLineWidth(1.0 + intensity * 3.0)
            actor.GetProperty().SetOpacity(0.6 + intensity * 0.4) # Make active beams more opaque

            self.ren.AddActor(actor); self.beam_visualization_actors.append(actor)
            logger.debug(f"Beam {i}: src_world={source_pos_world_lps}, iso_world={isocenter_world_lps}, weight={weight:.2f}")

        if self.vtkWidget.GetRenderWindow(): self.vtkWidget.GetRenderWindow().Render()
        logger.info("Beam visualization updated.")

    def _clear_beam_visualization(self):
        logger.debug(f"Clearing {len(self.beam_visualization_actors)} beam visualization actors.")
        for actor in self.beam_visualization_actors:
            self.ren.RemoveActor(actor)
        self.beam_visualization_actors.clear()
    
    def clear_view(self):
        logger.info("Clearing 3D viewer (volume, tumor, OARs, beams, and dose isosurfaces).")
        if self.volume_actor is not None: self.ren.RemoveVolume(self.volume_actor); self.volume_actor = None
        if self.tumor_actor is not None: self.ren.RemoveActor(self.tumor_actor); self.tumor_actor = None # Line 393 in previous trace
        self._clear_oar_actors()
        self._clear_dose_isosurfaces()
        self._clear_beam_visualization() 
        if self.vtkWidget.GetRenderWindow(): self.vtkWidget.GetRenderWindow().Render()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    viewer3d = DicomViewer3DWidget()
    vol_shape_zyx = (50, 64, 64)
    dummy_volume_data_zyx = np.random.randint(-1000, 2000, size=vol_shape_zyx, dtype=np.int16)
    dummy_tumor_mask_zyx = np.zeros(vol_shape_zyx, dtype=bool); dummy_tumor_mask_zyx[20:30, 25:35, 25:35] = True
    dummy_oar_masks_zyx = {"OAR1": np.zeros(vol_shape_zyx, dtype=bool)}; dummy_oar_masks_zyx["OAR1"][15:25, 10:20, 40:50] = True
    
    # Example image_properties - crucial for correct VTK visualization
    dummy_image_properties = {
        'pixel_spacing': [0.8, 0.9], # row_sp=0.8, col_sp=0.9
        'slice_thickness': 2.5,      # slice_thickness_mm
        'origin': [-100.0, -120.0, -80.0], # Example LPS origin
        'orientation_matrix_3x3': np.eye(3).tolist() # Simplest: identity orientation
    }
    
    viewer3d.update_volume(dummy_volume_data_zyx,
                           dummy_image_properties,
                           tumor_mask_full_zyx=dummy_tumor_mask_zyx, 
                           oar_masks_full_zyx=dummy_oar_masks_zyx)
    
    # Test dose
    dose_shape_crs = (vol_shape_zyx[2], vol_shape_zyx[1], vol_shape_zyx[0]) # (cols,rows,slices)
    dummy_dose_crs = np.zeros(dose_shape_crs, dtype=np.float32)
    cx,cy,cz = [d//2 for d in dose_shape_crs]; radius_dose = min(cx,cy,cz)//2
    x,y,z = np.ogrid[:dose_shape_crs[0], :dose_shape_crs[1], :dose_shape_crs[2]]
    mask_sphere_crs = (x-cx)**2 + (y-cy)**2 + (z-cz)**2 <= radius_dose**2
    dummy_dose_crs[mask_sphere_crs] = 60.0; dummy_dose_crs += np.random.rand(*dose_shape_crs)*10
    viewer3d._update_dose_isosurfaces(dummy_dose_crs, dummy_image_properties, isovalues_list=[20.0, 40.0, 55.0])

    # Test beam visualization
    # Planner coords (c,r,s) for isocenter and sources
    planner_grid_size_crs = dose_shape_crs 
    isocenter_planner_vox_test = np.array([planner_grid_size_crs[0]//2, planner_grid_size_crs[1]//2, planner_grid_size_crs[2]//2])
    
    beam_viz_test_data = {
        "beam_directions": [(1,0,0), (0,1,0), (-1,0,0), (0,-1,0), (0.707, 0.707, 0)], # Example directions
        "beam_weights": np.array([1.0, 0.8, 0.6, 0.4, 0.9]),
        "source_positions_planner_coords": [ # Sources far out along negative direction from isocenter
            (isocenter_planner_vox_test - np.array([1,0,0])*100).tolist(),
            (isocenter_planner_vox_test - np.array([0,1,0])*100).tolist(),
            (isocenter_planner_vox_test - np.array([-1,0,0])*100).tolist(),
            (isocenter_planner_vox_test - np.array([0,-1,0])*100).tolist(),
            (isocenter_planner_vox_test - np.array([0.707,0.707,0])*100).tolist()
        ],
        "isocenter_planner_coords": isocenter_planner_vox_test.tolist()
    }
    viewer3d.display_beams(beam_viz_test_data)

    viewer3d.resize(800, 600)
    viewer3d.show()
    sys.exit(app.exec_())
