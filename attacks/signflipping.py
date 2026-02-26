
import torch
import random
from worker import ByzantineWorker


class SignFlippingWorker(ByzantineWorker):
    def __str__(self) -> str:
        return "SignFlippingWorker"

    def get_gradient(self) -> torch.Tensor:
        gradients = []
        for w in self.simulator.workers:
            if not isinstance(w, ByzantineWorker):
                gradients.append(w.get_gradient())

        if gradients:
            self._gradient = -random.choice(gradients) 
        else:
            self._gradient = None  

        return self._gradient