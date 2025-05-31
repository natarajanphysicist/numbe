def calculate_plan_metrics(self, beam_weights_input: np.ndarray, num_fractions_for_eval: int = 30) -> Dict[str, Any]:
        """
        Calculates various plan metrics based on the provided beam weights.
        TCP/NTCP calculations are based on a total dose scaled by num_fractions_for_eval.
        Other metrics (mean dose, Vxx) are based on the single-fraction dose.
        """
        logger.info(f"Calculating plan metrics using {num_fractions_for_eval} fractions for TCP/NTCP evaluation...")
        metrics: Dict[str, Any] = {'tumor': {}, 'oars': {}}
        
        # Store num_fractions_for_eval for _calculate_ntcp to use if it needs it for EQD2
        # This assumes _calculate_ntcp might be refactored or already designed to use such an attribute.
        current_num_fractions_for_radiobio = getattr(self, 'num_fractions_for_radiobio', None)
        self.num_fractions_for_radiobio = num_fractions_for_eval

        if self.tumor_mask is None or not np.any(self.tumor_mask):
            logger.warning("calculate_plan_metrics: Tumor mask not available or empty. Tumor metrics will be zero/None.")
        
        # This calculates a dose distribution, typically representing a single fraction.
        # The internal scaling of calculate_dose() might be to a reference dose (e.g., base_dose_per_fraction).
        fractional_dose_for_metrics = self.calculate_dose(beam_weights_input)
        if fractional_dose_for_metrics is None:
            logger.error("calculate_plan_metrics: Failed to calculate fractional dose for metrics evaluation.")
            if current_num_fractions_for_radiobio is not None: self.num_fractions_for_radiobio = current_num_fractions_for_radiobio # Restore
            else: del self.num_fractions_for_radiobio # Clean up if it was newly added
            return metrics 

        # Total physical dose for TCP/NTCP evaluation, assuming constant fractional dose over the course.
        total_physical_dose_for_eval = fractional_dose_for_metrics * num_fractions_for_eval
        logger.info(f"calculate_plan_metrics: Fractional dose range for metrics: [{fractional_dose_for_metrics.min():.2f}, {fractional_dose_for_metrics.max():.2f}] Gy")
        logger.info(f"calculate_plan_metrics: Total physical dose for TCP/NTCP eval: [{total_physical_dose_for_eval.min():.2f}, {total_physical_dose_for_eval.max():.2f}] Gy")

        # Tumor Metrics
        if self.tumor_mask is not None and np.any(self.tumor_mask):
            tumor_doses_fractional = fractional_dose_for_metrics[self.tumor_mask]
            if tumor_doses_fractional.size > 0:
                mean_tumor_dose_fractional = float(np.mean(tumor_doses_fractional))
                metrics['tumor']['mean_fractional_dose'] = mean_tumor_dose_fractional
                
                # V95 based on mean fractional dose to tumor
                v95_threshold = 0.95 * mean_tumor_dose_fractional 
                metrics['tumor']['v95_fractional_mean_ref'] = float(np.sum(tumor_doses_fractional >= v95_threshold) * 100.0 / tumor_doses_fractional.size) if mean_tumor_dose_fractional > 1e-6 else 0.0
            else: # Tumor mask exists but no dose values (e.g. mask outside grid or all zero dose)
                metrics['tumor']['mean_fractional_dose'] = 0.0
                metrics['tumor']['v95_fractional_mean_ref'] = 0.0
            
            metrics['tumor']['tcp'] = self._calculate_tcp(total_physical_dose_for_eval) # _calculate_tcp expects total physical dose
        else:
            metrics['tumor']['mean_fractional_dose'] = 0.0
            metrics['tumor']['v95_fractional_mean_ref'] = 0.0
            metrics['tumor']['tcp'] = 0.0 # Or appropriate value for no tumor

        # OAR Metrics
        for oar_name, oar_mask in self.oar_masks.items():
            metrics['oars'][oar_name] = {}
            if oar_mask is not None and np.any(oar_mask):
                oar_doses_fractional = fractional_dose_for_metrics[oar_mask]
                if oar_doses_fractional.size > 0:
                    metrics['oars'][oar_name]['mean_fractional_dose'] = float(np.mean(oar_doses_fractional))
                    metrics['oars'][oar_name]['max_fractional_dose'] = float(np.max(oar_doses_fractional))
                    
                    # VxxGy equivalent fractional dose: e.g. V(20Gy total / 30 fractions) = V0.66Gy_per_fraction
                    # The threshold is for the fractional dose distribution
                    total_dose_threshold_for_Vx = 0.0
                    if 'lung' in oar_name.lower(): total_dose_threshold_for_Vx = 20.0 
                    elif 'heart' in oar_name.lower(): total_dose_threshold_for_Vx = 30.0
                    else: total_dose_threshold_for_Vx = 5.0 # Generic threshold for other OARs

                    fractional_dose_threshold_for_Vx = total_dose_threshold_for_Vx / num_fractions_for_eval
                    
                    metrics['oars'][oar_name][f'V{fractional_dose_threshold_for_Vx:.2f}Gy_frac'] = {
                        'dose_gy_per_fraction_threshold': fractional_dose_threshold_for_Vx,
                        'volume_pct': float(np.sum(oar_doses_fractional >= fractional_dose_threshold_for_Vx) * 100.0 / oar_doses_fractional.size)
                    }
                else: # OAR mask exists but no dose values
                    metrics['oars'][oar_name]['mean_fractional_dose'] = 0.0
                    metrics['oars'][oar_name]['max_fractional_dose'] = 0.0
                    metrics['oars'][oar_name]['V_generic_frac'] = {'dose_gy_per_fraction_threshold': 0.0, 'volume_pct': 0.0}

                metrics['oars'][oar_name]['ntcp'] = self._calculate_ntcp(total_physical_dose_for_eval, oar_name, num_fractions_for_eval=num_fractions_for_eval)
            else: # OAR mask not present or empty
                metrics['oars'][oar_name] = {
                    'mean_fractional_dose': 0.0, 'max_fractional_dose': 0.0, 
                    'V_generic_frac': {'dose_gy_per_fraction_threshold': 0.0, 'volume_pct': 0.0}, 
                    'ntcp': 0.0
                }
        
        # Restore previous num_fractions_for_radiobio if it existed
        if current_num_fractions_for_radiobio is not None: self.num_fractions_for_radiobio = current_num_fractions_for_radiobio
        elif hasattr(self, 'num_fractions_for_radiobio'): del self.num_fractions_for_radiobio # Clean up if it was newly added by this call

        logger.info(f"Calculated plan metrics: {metrics}")
        return metrics
