#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 12:30:02 2020

@author: dbeabout
"""

from abc import ABC

class Element(ABC):
    @classmethod
    def __init_subclass__(cls):
        required_class_variables = [
            "temperature",
            "mass",
            "rest_wavelength"
        ]
        for var in required_class_variables:
            if not hasattr(cls, var):
                raise NotImplementedError(
                    f'Class {cls} lacks required `{var}` class attribute'
                )    

class OVElement(Element):  # Oxygen V Element
          temperature = 10**(5.35)  # K – Temp of O V
          mass = 2.66e-26           # kg – Mass of O V
          rest_wavelength = 629.7   # Angstroms - Rest Wavelength of O V

class HeIElement(Element):  # Helium I Element
          temperature = 10**(4.0)  # K – Temp of He I
          mass = 6.6464731e-27           # kg – Mass of He I
          rest_wavelength = 584.3   # Angstroms - Rest Wavelength of He I

class MgX609Element(Element):  # Magnesium X Element
          temperature = 10**(6.05)  # K – Temp of Mg X
          mass = 4.0359398e-26           # kg – Mass of Mg X
          rest_wavelength = 609.8   # Angstroms - Rest Wavelength of Mg X

class MgX624Element(Element):  # Magnesium X Element
          temperature = 10**(6.05)  # K – Temp of Mg X
          mass = 4.0359398e-26           # kg – Mass of Mg X
          rest_wavelength = 624.9   # Angstroms - Rest Wavelength of Mg X

