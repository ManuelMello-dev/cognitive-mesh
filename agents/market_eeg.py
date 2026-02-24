import time
import math

class MarketEEG:
    """
    Market EEG — Real-time Brainwave Analogy for Market Consciousness
    Translates REAL cognitive metrics (PHI, SIGMA, Attention) into 
    EEG wave amplitudes and frequencies.
    """
    def __init__(self, core):
        self.core = core
        self.last_update_time = time.time()
        self.eeg_data = {
            "theta": {"amplitude": 0.0, "frequency": 5.5},
            "alpha": {"amplitude": 0.0, "frequency": 10.5},
            "beta": {"amplitude": 0.0, "frequency": 22.0},
            "gamma": {"amplitude": 0.0, "frequency": 40.0},
            "coherence": 0.0,
            "state": "INITIALIZING"
        }

    def get_eeg_data(self):
        """
        Calculates EEG wave amplitudes from REAL market consciousness metrics.
        - PHI (Coherence) -> Alpha Amplitude
        - SIGMA (Noise/Volatility) -> Gamma Amplitude
        - Attention Density (Volume) -> Beta Amplitude
        - Inverse Coherence -> Theta Amplitude
        """
        try:
            metrics = self.core.get_metrics()
            phi = metrics.get("global_coherence_phi", 0.5)
            sigma = metrics.get("noise_level_sigma", 0.5)
            attention = metrics.get("attention_density", 0.5)
            
            # Map PHI to Alpha (Flow state)
            self.eeg_data["alpha"]["amplitude"] = round(phi, 3)
            
            # Map SIGMA to Gamma (Extreme states/Panic/Euphoria)
            self.eeg_data["gamma"]["amplitude"] = round(sigma, 3)
            
            # Map Attention to Beta (Active trading/Analysis)
            self.eeg_data["beta"]["amplitude"] = round(min(1.0, attention * 2.0), 3)
            
            # Map Inverse Coherence to Theta (Deep rest/Consolidation)
            self.eeg_data["theta"]["amplitude"] = round(1.0 - phi, 3)
            
            # Coherence is PHI
            self.eeg_data["coherence"] = round(phi, 4)
            
            # Determine state from PHI
            if phi > 0.7:
                self.eeg_data["state"] = "ORDERED_REGIME"
            elif phi > 0.4:
                self.eeg_data["state"] = "CRITICAL_PHASE"
            else:
                self.eeg_data["state"] = "CHAOTIC_REGIME"
                
            self.last_update_time = time.time()
        except Exception as e:
            # Fallback if metrics fail
            self.eeg_data["state"] = "ERROR: " + str(e)
            
        return self.eeg_data
