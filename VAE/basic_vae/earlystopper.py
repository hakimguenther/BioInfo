import numpy as np

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.min_model = None

    def early_stop(self, validation_loss, model):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.min_model = model
            self.counter = 0
            print(f"current patience counter: {self.counter} / {self.patience}")
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            print(f"current patience counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                return True
        return False