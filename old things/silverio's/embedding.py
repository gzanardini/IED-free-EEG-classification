
class Embedding():
    def __init__(self, filename, eeg_label, is_patient, ied_presence=None):
        self.filename = filename
        self.eeg_label = eeg_label
        self.ied_presence = ied_presence
        self.is_patient = is_patient
        self.embedding = None
        self.sum_of_weights = 0
        
    def include_embedding(self, embedding, rmse):
        # weight = rmse
        # weight = 1 / (1 - rmse)
        weight = rmse**2
        if self.embedding is None: self.embedding = embedding * weight
        else: self.embedding += embedding * weight
        self.sum_of_weights += weight
    
    def get(self):
        return self.embedding / self.sum_of_weights