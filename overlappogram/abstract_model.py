from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(order=True)
class AbstractModel(ABC):
    @classmethod
    @abstractmethod
    def invert(self, response_function, data) -> tuple[np.ndarray, np.ndarray]:
        """
        Invert data.

        Parameters
        ----------
        response_function : TYPE
            Response function for inversion.
        data : TYPE
            Data (i.e. image).

        Returns
        -------
        em : TYPE
            Emisions (i.e. coefficients).
        data_out : TYPE
            Predicted output.

        """
        pass

    @classmethod
    @abstractmethod
    def add_fits_keywords(self, header):
        pass
