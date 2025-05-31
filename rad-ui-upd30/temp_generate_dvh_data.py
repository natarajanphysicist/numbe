def generate_dvh_data(self, dose_distribution_input: np.ndarray, num_bins: int = 100) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Generates DVH data for tumor and OARs from a given dose distribution.
        """
        logger.info("Generating DVH data...")
        if dose_distribution_input is None:
            logger.error("Dose distribution input is None. Cannot generate DVH.")
            return {}

        dvh_data: Dict[str, Dict[str, np.ndarray]] = {}
        # Determine overall max dose for consistent binning, handle empty or zero dose case
        max_dose_overall = np.max(dose_distribution_input) if np.any(dose_distribution_input) else 1.0 # Use 1.0 if dose is all zero to avoid issues with bins
        if max_dose_overall < 1e-6: # If max dose is effectively zero
            logger.warning("Max dose in distribution is close to 0. DVH will be trivial.")
            max_dose_overall = 1.0 # Ensure bins are created, though DVH will show 100% volume at 0 dose.

        # Prepare list of ROIs to process
        rois_to_process: List[Tuple[str, Optional[np.ndarray]]] = []
        
        # Tumor ROI
        # Ensure tumor_mask_name is an attribute, default to "Tumor" if not set during DICOM load
        current_tumor_mask_name = getattr(self, 'tumor_mask_name', 'Tumor')
        if self.tumor_mask is not None and np.any(self.tumor_mask) :
            rois_to_process.append((current_tumor_mask_name, self.tumor_mask))
        else:
            logger.warning(f"Tumor mask ('{current_tumor_mask_name}') is not available or empty. Skipping its DVH.")
        
        # OAR ROIs
        if hasattr(self, 'oar_masks') and self.oar_masks:
            for oar_name, oar_mask in self.oar_masks.items():
                if oar_mask is not None and np.any(oar_mask):
                    rois_to_process.append((oar_name, oar_mask))
                else:
                    logger.warning(f"OAR mask for '{oar_name}' is not available or empty. Skipping its DVH.")
        else:
            logger.warning("No OAR masks available to generate DVHs for.")

        if not rois_to_process:
            logger.warning("No valid ROIs (tumor or OARs) found to generate DVH for.")
            return {}

        for roi_name, roi_mask in rois_to_process:
            if roi_mask is None: continue # Should have been caught by np.any but defensive

            # Extract doses only within the ROI
            roi_doses = dose_distribution_input[roi_mask]
            
            if roi_doses.size == 0:
                logger.warning(f"ROI '{roi_name}' is empty after masking or mask does not overlap with dose grid. Creating trivial DVH.")
                # For an empty ROI or no overlap, all its volume (0 voxels) is effectively at 0 dose.
                # Or, more accurately, it has 0% volume at all dose levels.
                bins = np.linspace(0, max_dose_overall, num_bins + 1)
                volume_pct = np.zeros(num_bins) 
                dvh_data[roi_name] = {
                    'bins': bins[:-1], # Use bin starts (or centers: (bin_edges[:-1] + bin_edges[1:])/2 )
                    'volume_pct': volume_pct 
                }
                continue

            # Calculate histogram for this ROI
            # Ensure bins cover the range up to max_dose_overall for consistency across ROIs
            hist, bin_edges = np.histogram(roi_doses, bins=num_bins, range=(0, max_dose_overall))
            
            # Cumulative histogram (descending dose) - volume receiving >= dose
            cumulative_hist_descending = np.cumsum(hist[::-1])[::-1]
            
            # Normalize to percentage of ROI volume
            roi_total_voxels = np.sum(roi_mask) # Total number of voxels in this ROI
            if roi_total_voxels == 0: # Should be caught by roi_doses.size == 0 check
                 volume_percentages = np.zeros_like(cumulative_hist_descending, dtype=float)
            else:
                 volume_percentages = (cumulative_hist_descending / roi_total_voxels) * 100.0
            
            dvh_data[roi_name] = {
                'bins': bin_edges[:-1], # Use left edges of bins
                'volume_pct': volume_percentages
            }
            logger.debug(f"DVH for {roi_name}: {len(volume_percentages)} points. Max dose in ROI: {np.max(roi_doses):.2f} Gy. Min dose in ROI: {np.min(roi_doses):.2f} Gy.")
            
        logger.info("DVH data generation complete.")
        return dvh_data
