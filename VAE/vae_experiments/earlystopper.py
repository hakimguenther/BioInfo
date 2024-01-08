import numpy as np
import os
import torch

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.min_model = None

    def early_stop(self, validation_loss, model, docs_dir, experiment_name):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.min_model = model
            self.counter = 0
            self.save_model(model, docs_dir, experiment_name + "_best")
            print(f"saved new best model")
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            # print(f"current patience counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                return True
        return False
    
    def save_model(self, model, docs_dir, experiment_name):
        model_dir = docs_dir.replace('docs', 'models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(model.state_dict(), f'{model_dir}/{experiment_name}.pth')