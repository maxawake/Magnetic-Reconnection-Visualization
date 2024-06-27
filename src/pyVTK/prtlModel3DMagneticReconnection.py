__all__ = ["prtlModel3DMagneticReconnection"]
from pyprtl.util.logging import Log
from pyprtl.models.ModelBase import *
import numpy as np


@smproxy.source(label="PRTL Model 3D Magnetic Reconnection")
@prtl_model(dimensions=[128, 128, 128], extent=[-10, 10, -10, 10, -10, 10])
class prtlModel3DMagneticReconnection(prtlModelBase):
    def __init__(self):
        self._epsilon = 0
        prtlModelBase.__init__(self)

    @smproperty.doublevector(name="TimeValue", label="Time Value", default_values=0.0)
    @smdomain.doublerange(min=0.0, max=20.0)
    def SetTimeValue(self, t):
        self._epsilon = t
        self.Modified()

    def Sample(self, x, y, z):
        L_z = 1
        L_y = 1
        dx = (-1-
            self._epsilon
            * (1 - y**2 / L_z**2)
            / ((1 + z**2 / L_y**2) * (1 + y**2 / L_z**2))
        ) #+ z**2
        dy = x
        dz =  x / x

        return np.stack([dx, dy, dz], axis=-1)
