#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 13:28:22 2021

@author: dbeabout
"""

from dataclasses import dataclass
from abc import ABC, abstractclassmethod
import numpy as np
from typing import Tuple

@dataclass(order=True)
class AbstractModel(ABC):
    @abstractclassmethod
    def invert(self, response_function, data) -> Tuple[np.ndarray, np.ndarray]:
        '''
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
            Emmisions (i.e. coefficients).
        data_out : TYPE
            Predicted output.

        '''
        pass
    @abstractclassmethod
    def add_fits_keywords(self, header):
        pass
    