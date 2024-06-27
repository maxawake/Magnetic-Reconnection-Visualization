__all__ = ["prtlModel3DBifurcation"]
from pyprtl.util.logging import Log
from pyprtl.models.ModelBase import *
import numpy as np


@smproxy.source(label="PRTL Model 3D Bifurcation")
@prtl_model(dimensions=[128, 128, 128], extent=[-5, 5, -5, 5, -5, 5])
class prtlModel3DBifurcation(prtlModelBase):
    def __init__(self):
        self._epsilon = 0
        prtlModelBase.__init__(self)

    @smproperty.doublevector(name="TimeValue", label="Time Value", default_values=0.0)
    @smdomain.doublerange(min=0.0, max=1.0)
    def SetTimeValue(self, t):
        self._epsilon = t
        self.Modified()

    def Sample(self, x, y, z):
    
        dx = (y-2)**2 - self._epsilon + z**2 # + np.abs(x)
        dy = -x#+ 0.1*y
        dz = 1 * x / x

        return np.stack([dx, dy, dz], axis=-1)
