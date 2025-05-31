#--- START OF FILE plan_eval_utils.py ---

import numpy as np
import logging
from scipy.special import erf
from typing import Dict, Any, List, Optional, Tuple
from skimage.draw import polygon2mask # Add this import

logger = logging.getLogger(__name__)

def _calculate_tcp_external(dose_volume: np.ndarray, 
                            tumor_mask_roi: np.ndarray, 
                            radiobiological_params_tumor: Dict[str, float], 
                            voxel_volume_cm3: float, # Clarified unit
                            num_fractions: Optional[int] = None) -> float: # Added num_fractions
    """
    Calculates Tumor Control Probability (TCP) using the LQ-Poisson model.
    
    Args:
        dose_volume: Total physical dose per voxel (Gy).
        tumor_mask_roi: Boolean mask for the tumor.
        radiobiological_params_tumor: Dict containing 'alpha', 'beta', 'N0_density'.
                                      Alpha and beta are assumed to be per-fraction parameters.
        voxel_volume_cm3: Volume of a single voxel in cm^3.
        num_fractions: Number of fractions the total dose was delivered in. Required if alpha/beta are per-fraction.
    
    Returns:
        TCP as a percentage (0-100).
    """
    if tumor_mask_roi is None or not np.any(tumor_mask_roi):
        logger.warning("_calculate_tcp_external: Tumor mask is empty or None. TCP is 0.")
        return 0.0

    alpha_per_fraction = radiobiological_params_tumor.get("alpha", 0.3) # Gy^-1
    beta_per_fraction = radiobiological_params_tumor.get("beta", 0.03) # Gy^-2
    N0_density = radiobiological_params_tumor.get("N0_density", 1e7) # cells/cm^3

    if num_fractions is None or num_fractions <= 0:
        logger.error("_calculate_tcp_external: num_fractions must be a positive integer for TCP calculation with per-fraction alpha/beta.")
        # Fallback: Assume dose_volume is BED or alpha/beta are for total dose (less accurate)
        # For now, return 0 indicating error or inability to calculate.
        logger.warning("_calculate_tcp_external: num_fractions not provided or invalid. TCP calculation may be inaccurate or skipped.")
        return 0.0 # Or raise error

    tumor_mask_bool = tumor_mask_roi.astype(bool)
    relevant_total_doses_tumor = dose_volume[tumor_mask_bool]
    
    if relevant_total_doses_tumor.size == 0:
        logger.warning("_calculate_tcp_external: Tumor mask does not overlap with dose volume or is empty.")
        return 0.0

    dose_per_fraction_tumor_voxels = relevant_total_doses_tumor / num_fractions
    
    # SF_voxel = exp(-N * (alpha * d + beta * d^2))
    # SF_voxel = exp(-(alpha * D_total + beta * d * D_total))
    # where D_total is total dose, d is dose_per_fraction for that voxel.
    # log(SF_voxel) = -(alpha * D_total + beta * d * D_total)
    #               = -(alpha * D_total + beta * (D_total/N) * D_total)
    #               = -D_total * (alpha + beta * D_total / N) This is BED calculation essentially
    # So SF_voxel = exp(-(alpha_per_fraction * relevant_total_doses_tumor + 
    #                   beta_per_fraction * dose_per_fraction_tumor_voxels * relevant_total_doses_tumor))
    # This can be written as: SF = exp(-N_eff), where N_eff = Sum_voxels[ N0_voxel * SF_voxel ]
    # No, it's simpler: SF_per_voxel = exp(-(alpha_total_effect + beta_total_effect * d^2_total_effect))
    # Where SF for N fractions of dose d is exp(-N * (alpha*d + beta*d^2))
    
    # Cell kill term for each voxel: N * (alpha*d + beta*d^2)
    cell_kill_exponent = num_fractions * (alpha_per_fraction * dose_per_fraction_tumor_voxels + 
                                         beta_per_fraction * (dose_per_fraction_tumor_voxels**2))
    
    sf_map_tumor_voxels = np.exp(-cell_kill_exponent)
    
    surviving_clonogens_per_voxel = N0_density * voxel_volume_cm3 * sf_map_tumor_voxels
    total_surviving_clonogens = np.sum(surviving_clonogens_per_voxel)
    tcp = np.exp(-total_surviving_clonogens)
    
    logger.debug(f"_calculate_tcp_external: alpha_fx={alpha_per_fraction}, beta_fx={beta_per_fraction}, N0_density={N0_density}, voxel_vol={voxel_volume_cm3}, N_fx={num_fractions}")
    logger.debug(f"_calculate_tcp_external: Min/Max tumor total dose: {relevant_total_doses_tumor.min()}/{relevant_total_doses_tumor.max()}")
    logger.debug(f"_calculate_tcp_external: Min/Max tumor dose/fx: {dose_per_fraction_tumor_voxels.min()}/{dose_per_fraction_tumor_voxels.max()}")
    logger.debug(f"_calculate_tcp_external: Min/Max SF: {sf_map_tumor_voxels.min()}/{sf_map_tumor_voxels.max()}")
    logger.debug(f"_calculate_tcp_external: Total surviving clonogens: {total_surviving_clonogens}, TCP: {tcp*100.0}%")
    
    return tcp * 100.0


def _calculate_ntcp_external(dose_volume: np.ndarray, 
                             oar_mask_roi: np.ndarray, 
                             radiobiological_params_oar: Dict[str, float], 
                             voxel_volume_cm3: float, # Clarified unit
                             num_fractions_for_eqd2: int = 30) -> float:
    """
    Calculates Normal Tissue Complication Probability (NTCP) using the LKB model.
    Assumes dose_volume is total physical dose.
    """
    if oar_mask_roi is None or not np.any(oar_mask_roi):
        logger.warning("_calculate_ntcp_external: OAR mask is empty or None. NTCP is 0.")
        return 0.0

    oar_mask_bool = oar_mask_roi.astype(bool)
    relevant_oar_doses_total = dose_volume[oar_mask_bool]
    if relevant_oar_doses_total.size == 0:
        logger.warning("_calculate_ntcp_external: OAR mask does not overlap with dose volume or is empty.")
        return 0.0

    n_param = radiobiological_params_oar.get("n", 1.0) # LKB volume parameter
    m_param = radiobiological_params_oar.get("m", 0.5) # LKB slope parameter
    TD50_ref = radiobiological_params_oar.get("TD50", 50.0) # TD50 for reference fractionation (e.g. 2Gy/fx)
    alpha_beta_oar = radiobiological_params_oar.get("alpha_beta", 3.0) # Typical for normal tissue
    d_ref = 2.0 # Reference dose per fraction for TD50 (commonly 2 Gy)
    
    if num_fractions_for_eqd2 <= 0:
        logger.error("_calculate_ntcp_external: num_fractions_for_eqd2 must be positive.")
        return 0.0 

    dose_per_fraction_oar_voxels = relevant_oar_doses_total / num_fractions_for_eqd2
    
    # EQD2 = D_total * (d_frac_voxel + alpha/beta_OAR) / (d_ref + alpha/beta_OAR)
    eqd2_oar_voxels = relevant_oar_doses_total * \
                      (dose_per_fraction_oar_voxels + alpha_beta_oar) / (d_ref + alpha_beta_oar)
    
    # gEUD calculation
    if abs(n_param) < 1e-9: 
        gEUD = np.mean(eqd2_oar_voxels)
        logger.debug(f"_calculate_ntcp_external: n is close to zero, gEUD calculated as mean EQD2: {gEUD:.2f} Gy")
    else:
        try:
            gEUD = (np.mean(eqd2_oar_voxels**(1/n_param)))**n_param # Corrected gEUD formula
        except Exception as e:
            logger.warning(f"_calculate_ntcp_external: Error calculating gEUD with n={n_param}. Falling back to mean EQD2. Error: {e}")
            gEUD = np.mean(eqd2_oar_voxels)
    
    if abs(m_param * TD50_ref) < 1e-9: 
        logger.warning("_calculate_ntcp_external: m*TD50 is close to zero. NTCP will be 0 or 100 based on gEUD vs TD50.")
        return 100.0 if gEUD > TD50_ref else 0.0
        
    t_val = (gEUD - TD50_ref) / (m_param * TD50_ref)
    ntcp = 0.5 * (1 + erf(t_val / np.sqrt(2)))
    
    logger.debug(f"_calculate_ntcp_external for OAR: TD50_ref={TD50_ref}, m={m_param}, n={n_param}, alpha_beta={alpha_beta_oar}")
    logger.debug(f"_calculate_ntcp_external: Min/Max OAR total dose: {relevant_oar_doses_total.min()}/{relevant_oar_doses_total.max()}")
    logger.debug(f"_calculate_ntcp_external: Min/Max OAR EQD2: {eqd2_oar_voxels.min()}/{eqd2_oar_voxels.max()}")
    logger.debug(f"_calculate_ntcp_external: gEUD: {gEUD:.2f} Gy, t-value: {t_val:.2f}, NTCP: {ntcp*100.0:.2f}%")

    return ntcp * 100.0


def calculate_plan_metrics_external(dose_distribution_crs: np.ndarray, # (cols, rows, slices)
                                    tumor_mask_src: Optional[np.ndarray], # (slices, rows, cols)
                                    oar_masks_crs: Dict[str, np.ndarray], # (cols, rows, slices)
                                    radiobiological_params: Dict[str, Dict[str, float]], 
                                    voxel_volume_cm3: float, # cm^3
                                    target_prescription_dose: float, 
                                    num_fractions_for_radiobio: int = 30) -> Dict[str, Any]:
    """
    Calculates plan metrics using external utility functions.
    Args:
        dose_distribution_crs: 3D dose array (cols, rows, slices).
        tumor_mask_src: 3D tumor mask array (slices, rows, cols) or None.
        oar_masks_crs: Dict of OAR names to 3D OAR mask arrays (cols, rows, slices).
        radiobiological_params: Dict of radiobiological parameters.
        voxel_volume_cm3: Voxel volume in cm^3.
        target_prescription_dose: Total prescribed dose for the plan (Gy).
        num_fractions_for_radiobio: Number of fractions this dose is delivered in (for EQD2/TCP).
    """
    metrics: Dict[str, Any] = {'tumor': {}, 'oars': {}}
    logger.info(f"Calculating external plan metrics. Target prescription dose: {target_prescription_dose} Gy, Num fractions for radiobio: {num_fractions_for_radiobio}")

    # Transpose tumor_mask_src (slices, rows, cols) to planner orientation (cols, rows, slices) for consistent indexing with dose
    tumor_mask_crs = None
    if tumor_mask_src is not None:
        tumor_mask_crs = np.transpose(tumor_mask_src, (2, 1, 0)).astype(bool)

    # Tumor Metrics
    if tumor_mask_crs is not None and np.any(tumor_mask_crs):
        tumor_doses = dose_distribution_crs[tumor_mask_crs] # Direct indexing
        if tumor_doses.size > 0:
            metrics['tumor']['mean_dose'] = float(np.mean(tumor_doses))
            metrics['tumor']['min_dose'] = float(np.min(tumor_doses))
            metrics['tumor']['max_dose'] = float(np.max(tumor_doses))
            
            # V95% (dose covering 95% of PTV) - requires sorting, more complex
            # D95% (dose received by 95% of PTV) - easier:
            metrics['tumor']['D95_percent_dose'] = float(np.percentile(tumor_doses, 5)) # 5th percentile for D95

            # V(prescription_dose): % volume receiving >= prescription
            v_presc_val = float(np.sum(tumor_doses >= target_prescription_dose) * 100.0 / tumor_doses.size)
            metrics['tumor'][f'V{target_prescription_dose:.0f}Gy_pct_vol'] = v_presc_val
            
            # V95_prescription (coverage of PTV by 95% of prescription dose)
            v95_threshold = 0.95 * target_prescription_dose
            metrics['tumor']['coverage_at_95pct_prescription'] = float(np.sum(tumor_doses >= v95_threshold) * 100.0 / tumor_doses.size)
            
        else: # Tumor mask exists but no dose values (e.g., mask outside dose grid)
            metrics['tumor'].update({'mean_dose': 0.0, 'min_dose':0.0, 'max_dose':0.0, 'D95_percent_dose':0.0, 
                                     f'V{target_prescription_dose:.0f}Gy_pct_vol':0.0, 'coverage_at_95pct_prescription':0.0})
        
        metrics['tumor']['tcp'] = _calculate_tcp_external(
            dose_volume=dose_distribution_crs, 
            tumor_mask_roi=tumor_mask_crs, 
            radiobiological_params_tumor=radiobiological_params.get("tumor", {}), 
            voxel_volume_cm3=voxel_volume_cm3,
            num_fractions=num_fractions_for_radiobio
        )
    else:
        logger.warning("calculate_plan_metrics_external: Tumor mask not available or empty.")
        metrics['tumor'] = {'mean_dose': 0.0, 'min_dose':0.0, 'max_dose':0.0, 'D95_percent_dose':0.0, 
                            f'V{target_prescription_dose:.0f}Gy_pct_vol':0.0, 'coverage_at_95pct_prescription':0.0, 'tcp': 0.0}

    # OAR Metrics
    for oar_name, oar_mask_roi_crs in oar_masks_crs.items(): # oar_mask_roi_crs is already (cols,rows,slices)
        metrics['oars'][oar_name] = {}
        if oar_mask_roi_crs is not None and np.any(oar_mask_roi_crs):
            oar_doses = dose_distribution_crs[oar_mask_roi_crs.astype(bool)] # Direct indexing
            if oar_doses.size > 0:
                metrics['oars'][oar_name]['mean_dose'] = float(np.mean(oar_doses))
                metrics['oars'][oar_name]['max_dose'] = float(np.max(oar_doses))

                vx_threshold_gy = 5.0 
                oar_name_lower = oar_name.lower()
                if 'lung' in oar_name_lower: vx_threshold_gy = 20.0
                elif 'heart' in oar_name_lower: vx_threshold_gy = 30.0
                elif 'spinal_cord' in oar_name_lower or 'cord' in oar_name_lower: vx_threshold_gy = 45.0 # Example for cord max
                
                metrics['oars'][oar_name][f'V{vx_threshold_gy:.0f}Gy_pct_vol'] = float(np.sum(oar_doses >= vx_threshold_gy) * 100.0 / oar_doses.size)
            else:
                metrics['oars'][oar_name].update({'mean_dose': 0.0, 'max_dose': 0.0, 'V_genericGy_pct_vol': 0.0})

            metrics['oars'][oar_name]['ntcp'] = _calculate_ntcp_external(
                dose_volume=dose_distribution_crs, 
                oar_mask_roi=oar_mask_roi_crs, 
                radiobiological_params_oar=radiobiological_params.get(oar_name, radiobiological_params.get(oar_name_lower, {})), # Try exact then lower
                voxel_volume_cm3=voxel_volume_cm3,
                num_fractions_for_eqd2=num_fractions_for_radiobio
            )
        else:
            logger.warning(f"calculate_plan_metrics_external: OAR mask for '{oar_name}' not available or empty.")
            metrics['oars'][oar_name] = {'mean_dose':0.0, 'max_dose':0.0, 'V_genericGy_pct_vol':0.0, 'ntcp':0.0}
            
    logger.info(f"External plan metrics calculated.") # Metrics dict can be large, avoid logging full dict at INFO
    logger.debug(f"External plan metrics details: {metrics}")
    return metrics


def generate_dvh_data_external(dose_distribution_crs: np.ndarray, # (cols, rows, slices)
                               tumor_mask_src: Optional[np.ndarray], # (slices, rows, cols)
                               oar_masks_crs: Dict[str, np.ndarray], # (cols, rows, slices)
                               tumor_mask_name: str = 'Tumor', 
                               num_bins: int = 200) -> Dict[str, Dict[str, np.ndarray]]: # Increased num_bins
    """
    Generates DVH data for tumor and OARs from a given dose distribution.
    Args:
        dose_distribution_crs: 3D dose array (cols, rows, slices).
        tumor_mask_src: 3D tumor mask array (slices, rows, cols) or None.
        oar_masks_crs: Dict of OAR names to 3D OAR mask arrays (cols, rows, slices).
        tumor_mask_name: Name for the tumor ROI.
        num_bins: Number of bins for the DVH histogram.
    """
    logger.info("Generating DVH data externally...")
    if dose_distribution_crs is None:
        logger.error("generate_dvh_data_external: Dose distribution input is None.")
        return {}

    dvh_data: Dict[str, Dict[str, np.ndarray]] = {}
    max_dose_overall = np.max(dose_distribution_crs) if np.any(dose_distribution_crs) else 1.0 # Default max if dose is zero
    if max_dose_overall < 1e-6: max_dose_overall = 1.0 # Ensure max_dose_overall is at least a small positive number for binning

    rois_to_process: List[Tuple[str, Optional[np.ndarray]]] = []
    
    # Tumor: transpose from (s,r,c) to (c,r,s)
    if tumor_mask_src is not None and np.any(tumor_mask_src):
        tumor_mask_crs = np.transpose(tumor_mask_src, (2, 1, 0)).astype(bool)
        rois_to_process.append((tumor_mask_name, tumor_mask_crs))
    
    # OARs are already (c,r,s)
    if oar_masks_crs:
        for name, mask_data_crs in oar_masks_crs.items():
            if mask_data_crs is not None and np.any(mask_data_crs):
                rois_to_process.append((name, mask_data_crs.astype(bool)))

    if not rois_to_process:
        logger.warning("generate_dvh_data_external: No valid ROIs to generate DVH for.")
        return {}

    for roi_name, roi_mask_data_crs in rois_to_process:
        if roi_mask_data_crs is None: continue 

        # Ensure mask and dose distribution shapes are compatible for direct indexing
        if roi_mask_data_crs.shape != dose_distribution_crs.shape:
            logger.error(f"Shape mismatch for ROI '{roi_name}' ({roi_mask_data_crs.shape}) and dose ({dose_distribution_crs.shape}). Skipping DVH.")
            continue

        roi_doses = dose_distribution_crs[roi_mask_data_crs] # Direct indexing
        
        if roi_doses.size == 0: # Mask exists but no voxels (should be caught by np.any earlier) or no overlap
            logger.warning(f"ROI '{roi_name}' is empty or has no overlap with dose. DVH will be zero.")
            bins = np.linspace(0, max_dose_overall, num_bins + 1)
            volume_pct = np.zeros(num_bins) # All bins have zero volume
            dvh_data[roi_name] = {'bins': bins[:-1], 'volume_pct': volume_pct} # bins[:-1] are the left edges
            continue

        hist, bin_edges = np.histogram(roi_doses, bins=num_bins, range=(0, max_dose_overall))
        
        # Cumulative histogram (descending): Volume receiving >= dose
        cumulative_hist_desc = np.cumsum(hist[::-1])[::-1] 
        
        roi_total_voxels = roi_doses.size # Number of voxels in the ROI that have dose values
        
        volume_percentages = (cumulative_hist_desc / roi_total_voxels) * 100.0 if roi_total_voxels > 0 else np.zeros_like(cumulative_hist_desc, dtype=float)
        
        dvh_data[roi_name] = {'bins': bin_edges[:-1], 'volume_pct': volume_percentages}
        logger.debug(f"DVH for {roi_name}: {len(volume_percentages)} points, Max dose in ROI: {roi_doses.max():.2f} Gy")
            
    logger.info("External DVH data generation complete.")
    return dvh_data

def create_mask_from_slice_contours(
    slice_contours: Dict[int, List[List[tuple]]], 
    volume_shape_zyx: Tuple[int, int, int]
) -> Optional[np.ndarray]:
    """
    Creates a 3D boolean mask from 2D contours drawn on slices.

    Args:
        slice_contours: Dict where keys are slice indices (z) and values are lists of contours.
                        Each contour is a list of (x, y) points (col, row).
        volume_shape_zyx: The ZYX shape of the target volume (num_slices, num_rows, num_cols).

    Returns:
        A 3D NumPy boolean array representing the mask (slices, rows, cols), or None if errors occur.
    """
    if not volume_shape_zyx or len(volume_shape_zyx) != 3:
        logger.error("Invalid volume_shape_zyx provided for mask creation.")
        return None
    
    num_slices, num_rows, num_cols = volume_shape_zyx
    full_mask_zyx = np.zeros(volume_shape_zyx, dtype=bool) # (slices, rows, cols)
    logger.info(f"Creating mask from contours for volume shape (Z,Y,X): {volume_shape_zyx}")

    for slice_idx, contours_on_slice in slice_contours.items():
        if not (0 <= slice_idx < num_slices):
            logger.warning(f"Slice index {slice_idx} is out of bounds for volume shape {volume_shape_zyx}. Skipping.")
            continue

        slice_plane_mask = np.zeros((num_rows, num_cols), dtype=bool) # Mask for the current slice (Y, X)
        for contour_points_xy in contours_on_slice: # contour_points are (x,y) i.e. (col, row)
            if len(contour_points_xy) < 3:
                logger.debug(f"Skipping contour on slice {slice_idx} with < 3 points.")
                continue
            
            # polygon2mask expects (rows, cols) for image_shape, and polygon as [(r0,c0), (r1,c1), ...]
            # Our contour_points_xy are [(c0,r0), (c1,r1), ...]. We need to swap them.
            # However, skimage.draw.polygon takes r,c arrays separately.
            # Let's prepare points as list of (row, col) for polygon2mask's `polygon` argument.
            
            # Original points are (col, row)
            polygon_rc_coords = np.array([(p[1], p[0]) for p in contour_points_xy]) # Convert to (row, col) pairs

            try:
                # Create a mask for this single polygon
                # Ensure points are within bounds before calling polygon2mask, or clip.
                # polygon2mask's `shape` argument is (num_rows, num_cols)
                current_polygon_mask_yx = polygon2mask(shape=(num_rows, num_cols), polygon=polygon_rc_coords)
                slice_plane_mask |= current_polygon_mask_yx # Combine with OR if multiple contours on one slice
            except Exception as e:
                logger.error(f"Error creating mask for a contour on slice {slice_idx} with points {polygon_rc_coords[:3]}...: {e}", exc_info=True)
        
        full_mask_zyx[slice_idx, :, :] = slice_plane_mask
        if np.any(slice_plane_mask):
            logger.info(f"Processed contours for slice {slice_idx}. Masked voxels on this slice: {np.sum(slice_plane_mask)}")

    if np.any(full_mask_zyx):
        logger.info(f"Successfully created 3D mask from contours. Total masked voxels: {np.sum(full_mask_zyx)}")
    else:
        logger.warning("No voxels were masked from the provided contours. Resulting mask is empty.")
        
    return full_mask_zyx
