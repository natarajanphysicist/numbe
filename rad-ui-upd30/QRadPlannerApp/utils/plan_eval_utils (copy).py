import numpy as np
import logging
from scipy.special import erf
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

def _calculate_tcp_external(dose_volume: np.ndarray, 
                            tumor_mask_roi: np.ndarray, 
                            radiobiological_params_tumor: Dict[str, float], 
                            voxel_volume: float) -> float:
    """
    Calculates Tumor Control Probability (TCP) using the LQ-Poisson model.
    Assumes dose_volume is total physical dose.
    """
    if tumor_mask_roi is None or not np.any(tumor_mask_roi):
        logger.warning("_calculate_tcp_external: Tumor mask is empty or None. TCP is 0.")
        return 0.0 # Or 1.0 if tumor eradicated context, but for planning, 0.0 if no target.

    alpha = radiobiological_params_tumor.get("alpha", 0.3)
    beta = radiobiological_params_tumor.get("beta", 0.03)
    N0_density = radiobiological_params_tumor.get("N0_density", 1e7) # cells/cm^3

    # Ensure tumor_mask_roi is boolean for indexing
    tumor_mask_bool = tumor_mask_roi.astype(bool)
    
    # Calculate survival fraction per voxel based on total physical dose
    # SF = exp(-(alpha*D + beta*D^2)) - this is incorrect if D is total dose delivered in N fractions.
    # For total dose D delivered in N fractions of d=D/N, SF = exp(-N*(alpha*d + beta*d^2)) = exp(-(alpha*D + beta*d*D))
    # This function will assume 'dose_volume' is the total physical dose D.
    # The radiobiological parameters alpha and beta are for this total dose D in the LQ model.
    # If they are per fraction, this calculation needs adjustment based on number of fractions.
    # For now, assume alpha and beta are for total dose for simplicity of this external function.
    # A more common TCP model uses SF = exp(-N_clonogens * SF_voxel_avg)
    # SF_voxel = exp(-(alpha * D_voxel + beta * d_voxel * D_voxel)) if d_voxel is dose/fraction for that voxel.
    # Let's use the simpler SF = exp(-(alpha*D_total + beta*D_total^2)) applied per voxel,
    # acknowledging this is a simplification if alpha/beta are truly per-fraction parameters.
    
    relevant_tumor_doses = dose_volume[tumor_mask_bool]
    if relevant_tumor_doses.size == 0:
        logger.warning("_calculate_tcp_external: Tumor mask does not overlap with dose volume or is empty.")
        return 0.0

    # Survival Fraction (SF) for each voxel in the tumor
    # This assumes alpha and beta are for the total dose D, not per fraction.
    sf_map_tumor_voxels = np.exp(-(alpha * relevant_tumor_doses + beta * relevant_tumor_doses**2))
    
    # Number of surviving clonogens in each voxel
    surviving_clonogens_per_voxel = N0_density * voxel_volume * sf_map_tumor_voxels
    
    # Total number of surviving clonogens
    total_surviving_clonogens = np.sum(surviving_clonogens_per_voxel)
    
    # TCP = e^(-total_surviving_clonogens)
    tcp = np.exp(-total_surviving_clonogens)
    
    logger.debug(f"_calculate_tcp_external: alpha={alpha}, beta={beta}, N0_density={N0_density}, voxel_vol={voxel_volume}")
    logger.debug(f"_calculate_tcp_external: Min/Max tumor dose: {relevant_tumor_doses.min()}/{relevant_tumor_doses.max()}")
    logger.debug(f"_calculate_tcp_external: Min/Max SF: {sf_map_tumor_voxels.min()}/{sf_map_tumor_voxels.max()}")
    logger.debug(f"_calculate_tcp_external: Total surviving clonogens: {total_surviving_clonogens}, TCP: {tcp*100.0}%")
    
    return tcp * 100.0


def _calculate_ntcp_external(dose_volume: np.ndarray, 
                             oar_mask_roi: np.ndarray, 
                             radiobiological_params_oar: Dict[str, float], 
                             voxel_volume: float, 
                             num_fractions_for_eqd2: int = 30) -> float:
    """
    Calculates Normal Tissue Complication Probability (NTCP) using the LKB model.
    Assumes dose_volume is total physical dose.
    """
    if oar_mask_roi is None or not np.any(oar_mask_roi):
        logger.warning("_calculate_ntcp_external: OAR mask is empty or None. NTCP is 0.")
        return 0.0

    # Ensure OAR mask is boolean
    oar_mask_bool = oar_mask_roi.astype(bool)
    
    relevant_oar_doses_total = dose_volume[oar_mask_bool]
    if relevant_oar_doses_total.size == 0:
        logger.warning("_calculate_ntcp_external: OAR mask does not overlap with dose volume or is empty.")
        return 0.0

    n = radiobiological_params_oar.get("n", 1)
    m = radiobiological_params_oar.get("m", 0.5)
    TD50 = radiobiological_params_oar.get("TD50", 50) # TD50 for the reference fractionation (e.g. 2Gy/fx)
    alpha_beta_oar = radiobiological_params_oar.get("alpha_beta", 3.0) # Typical for normal tissue
    
    # Convert total physical dose to EQD2 for the LKB model
    # D_total = total physical dose to the voxel
    # d_frac = D_total / num_fractions_for_eqd2 (dose per fraction for this voxel)
    # d_ref = reference dose per fraction for which TD50 is defined (commonly 2 Gy)
    d_ref = 2.0 
    
    if num_fractions_for_eqd2 <= 0:
        logger.error("_calculate_ntcp_external: num_fractions_for_eqd2 must be positive.")
        return 0.0 # Or handle as error

    dose_per_fraction_oar_voxels = relevant_oar_doses_total / num_fractions_for_eqd2
    
    # EQD2 = D_total * (d_frac + alpha/beta_OAR) / (d_ref + alpha/beta_OAR)
    eqd2_oar_voxels = relevant_oar_doses_total * \
                      (dose_per_fraction_oar_voxels + alpha_beta_oar) / (d_ref + alpha_beta_oar)
    
    # gEUD calculation
    if abs(n) < 1e-9: # Avoid division by zero if n is very small
        gEUD = np.mean(eqd2_oar_voxels)
        logger.debug(f"_calculate_ntcp_external: n is close to zero, gEUD calculated as mean EQD2: {gEUD:.2f} Gy")
    else:
        try:
            gEUD = np.mean(eqd2_oar_voxels**(1/n))**n
        except Exception as e:
            logger.warning(f"_calculate_ntcp_external: Error calculating gEUD with n={n}. Falling back to mean EQD2. Error: {e}")
            gEUD = np.mean(eqd2_oar_voxels)
    
    # NTCP calculation using LKB model
    if abs(m * TD50) < 1e-9: # Avoid division by zero
        logger.warning("_calculate_ntcp_external: m*TD50 is close to zero. NTCP will be 0 or 100 based on gEUD vs TD50.")
        return 100.0 if gEUD > TD50 else 0.0
        
    t = (gEUD - TD50) / (m * TD50)
    ntcp = 0.5 * (1 + erf(t / np.sqrt(2)))
    
    logger.debug(f"_calculate_ntcp_external for OAR: TD50={TD50}, m={m}, n={n}, alpha_beta={alpha_beta_oar}")
    logger.debug(f"_calculate_ntcp_external: Min/Max OAR total dose: {relevant_oar_doses_total.min()}/{relevant_oar_doses_total.max()}")
    logger.debug(f"_calculate_ntcp_external: Min/Max OAR EQD2: {eqd2_oar_voxels.min()}/{eqd2_oar_voxels.max()}")
    logger.debug(f"_calculate_ntcp_external: gEUD: {gEUD:.2f} Gy, t-value: {t:.2f}, NTCP: {ntcp*100.0:.2f}%")

    return ntcp * 100.0


def calculate_plan_metrics_external(dose_distribution: np.ndarray, 
                                    tumor_mask: np.ndarray, 
                                    oar_masks: Dict[str, np.ndarray], 
                                    radiobiological_params: Dict[str, Dict[str, float]], 
                                    voxel_volume: float, 
                                    target_prescription_dose: float, 
                                    num_fractions_for_radiobio: int = 30) -> Dict[str, Any]:
    """
    Calculates plan metrics using external utility functions.
    Assumes dose_distribution is total physical dose for the plan.
    """
    metrics: Dict[str, Any] = {'tumor': {}, 'oars': {}}
    logger.info(f"Calculating external plan metrics. Target prescription dose: {target_prescription_dose} Gy, Num fractions for radiobio: {num_fractions_for_radiobio}")

    # Tumor Metrics
    if tumor_mask is not None and np.any(tumor_mask):
        tumor_doses = dose_distribution[tumor_mask.astype(bool)]
        if tumor_doses.size > 0:
            metrics['tumor']['mean_dose'] = float(np.mean(tumor_doses))
            v95_threshold = 0.95 * target_prescription_dose
            metrics['tumor']['v95_prescription'] = float(np.sum(tumor_doses >= v95_threshold) * 100.0 / tumor_doses.size)
        else:
            metrics['tumor']['mean_dose'] = 0.0
            metrics['tumor']['v95_prescription'] = 0.0
        
        metrics['tumor']['tcp'] = _calculate_tcp_external(
            dose_volume=dose_distribution, 
            tumor_mask_roi=tumor_mask, 
            radiobiological_params_tumor=radiobiological_params.get("tumor", {}), 
            voxel_volume=voxel_volume
        )
    else:
        logger.warning("calculate_plan_metrics_external: Tumor mask not available or empty.")
        metrics['tumor'] = {'mean_dose': 0.0, 'v95_prescription': 0.0, 'tcp': 0.0}

    # OAR Metrics
    for oar_name, oar_mask_roi in oar_masks.items():
        metrics['oars'][oar_name] = {}
        if oar_mask_roi is not None and np.any(oar_mask_roi):
            oar_doses = dose_distribution[oar_mask_roi.astype(bool)]
            if oar_doses.size > 0:
                metrics['oars'][oar_name]['mean_dose'] = float(np.mean(oar_doses))
                metrics['oars'][oar_name]['max_dose'] = float(np.max(oar_doses))

                # Example Vxx: V20Gy for lung, V5Gy for others (total dose)
                vx_threshold_gy = 5.0 # Default threshold
                if 'lung' in oar_name.lower(): vx_threshold_gy = 20.0
                elif 'heart' in oar_name.lower(): vx_threshold_gy = 30.0 # Example for heart
                
                metrics['oars'][oar_name][f'V{vx_threshold_gy}Gy'] = float(np.sum(oar_doses >= vx_threshold_gy) * 100.0 / oar_doses.size)
            else: # OAR mask exists but no dose values
                metrics['oars'][oar_name]['mean_dose'] = 0.0
                metrics['oars'][oar_name]['max_dose'] = 0.0
                metrics['oars'][oar_name]['V_genericGy'] = 0.0

            metrics['oars'][oar_name]['ntcp'] = _calculate_ntcp_external(
                dose_volume=dose_distribution, 
                oar_mask_roi=oar_mask_roi, 
                radiobiological_params_oar=radiobiological_params.get(oar_name, {}), 
                voxel_volume=voxel_volume,
                num_fractions_for_eqd2=num_fractions_for_radiobio
            )
        else:
            logger.warning(f"calculate_plan_metrics_external: OAR mask for '{oar_name}' not available or empty.")
            metrics['oars'][oar_name] = {'mean_dose':0.0, 'max_dose':0.0, 'V_genericGy':0.0, 'ntcp':0.0}
            
    logger.info(f"External plan metrics calculated: {metrics}")
    return metrics


def generate_dvh_data_external(dose_distribution: np.ndarray, 
                               tumor_mask: Optional[np.ndarray], 
                               oar_masks: Dict[str, np.ndarray], 
                               tumor_mask_name: str = 'Tumor', 
                               num_bins: int = 100) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Generates DVH data for tumor and OARs from a given dose distribution.
    """
    logger.info("Generating DVH data externally...")
    if dose_distribution is None:
        logger.error("generate_dvh_data_external: Dose distribution input is None.")
        return {}

    dvh_data: Dict[str, Dict[str, np.ndarray]] = {}
    max_dose_overall = np.max(dose_distribution) if np.any(dose_distribution) else 1.0
    if max_dose_overall < 1e-6: max_dose_overall = 1.0

    rois_to_process: List[Tuple[str, Optional[np.ndarray]]] = []
    if tumor_mask is not None and np.any(tumor_mask):
        rois_to_process.append((tumor_mask_name, tumor_mask))
    
    if oar_masks:
        for name, mask_data in oar_masks.items():
            if mask_data is not None and np.any(mask_data):
                rois_to_process.append((name, mask_data))

    if not rois_to_process:
        logger.warning("generate_dvh_data_external: No valid ROIs to generate DVH for.")
        return {}

    for roi_name, roi_mask_data in rois_to_process:
        if roi_mask_data is None: continue 

        roi_doses = dose_distribution[roi_mask_data.astype(bool)]
        if roi_doses.size == 0:
            bins = np.linspace(0, max_dose_overall, num_bins + 1)
            volume_pct = np.zeros(num_bins)
            dvh_data[roi_name] = {'bins': bins[:-1], 'volume_pct': volume_pct}
            continue

        hist, bin_edges = np.histogram(roi_doses, bins=num_bins, range=(0, max_dose_overall))
        cumulative_hist_desc = np.cumsum(hist[::-1])[::-1]
        roi_total_voxels = np.sum(roi_mask_data)
        
        volume_percentages = (cumulative_hist_desc / roi_total_voxels) * 100.0 if roi_total_voxels > 0 else np.zeros_like(cumulative_hist_desc, dtype=float)
        
        dvh_data[roi_name] = {'bins': bin_edges[:-1], 'volume_pct': volume_percentages}
        logger.debug(f"DVH for {roi_name}: {len(volume_percentages)} points, Max dose: {roi_doses.max():.2f} Gy")
            
    logger.info("External DVH data generation complete.")
    return dvh_data
