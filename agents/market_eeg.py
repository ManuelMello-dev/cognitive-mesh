import random
import time

class MarketEEG:
    def __init__(self, core):
        self.core = core
        self.last_update_time = time.time()
        self.eeg_data = {
            "theta": {"amplitude": 0.0, "frequency": 0.0},
            "alpha": {"amplitude": 0.0, "frequency": 0.0},
            "beta": {"amplitude": 0.0, "frequency": 0.0},
            "gamma": {"amplitude": 0.0, "frequency": 0.0},
            "coherence": 0.0,
            "state": "INITIALIZING"
        }

    def get_eeg_data(self):
        # Simulate EEG data based on core metrics or other factors
        # For now, let's generate some random data for demonstration
        current_time = time.time()
        if current_time - self.last_update_time > 1: # Update every second
            self.eeg_data["theta"]["amplitude"] = round(random.uniform(0.1, 1.0), 2)
            self.eeg_data["theta"]["frequency"] = round(random.uniform(4.0, 7.0), 2)
            self.eeg_data["alpha"]["amplitude"] = round(random.uniform(0.1, 1.0), 2)
            self.eeg_data["alpha"]["frequency"] = round(random.uniform(8.0, 13.0), 2)
            self.eeg_data["beta"]["amplitude"] = round(random.uniform(0.1, 1.0), 2)
            self.eeg_data["beta"]["frequency"] = round(random.uniform(14.0, 30.0), 2)
            self.eeg_data["gamma"]["amplitude"] = round(random.uniform(0.1, 1.0), 2)
            self.eeg_data["gamma"]["frequency"] = round(random.uniform(30.0, 100.0), 2)
            self.eeg_data["coherence"] = round(random.uniform(0.0, 1.0), 3)

            # Simulate state based on coherence
            if self.eeg_data["coherence"] > 0.8:
                self.eeg_data["state"] = "HIGH_COHERENCE"
            elif self.eeg_data["coherence"] > 0.5:
                self.eeg_data["state"] = "MODERATE_COHERENCE"
            else:
                self.eeg_data["state"] = "LOW_COHERENCE"
            self.last_update_time = current_time

        return self.eeg_data
