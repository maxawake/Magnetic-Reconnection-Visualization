__all__ = ["prtlModelCriticalPoint"]
from pyprtl.util.logging import Log
from pyprtl.models.ModelBase import *
from typing import Union
import numpy as np


@smproxy.source(label="PRTL Model Critical Point")
@prtl_model(dimensions=[64, 64, 64], extent=[-1, 1, -1, 1, -1, 1])
class prtlModelCriticalPoint(prtlModelBase):
    def __init__(self):
        self._epsilon = 0
        self._typeID = 0
        prtlModelBase.__init__(self)

    @smproperty.intvector(label="Type", name="Type", default_values=0)
    @smdomain_enumeration(
        [
            "Source",
            "Sink",
            "Saddle 12",
            "Saddle 21",
            "Spiral Source",
            "Spiral Sink",
            "Spiral Saddle 12",
            "Spiral Saddle 21",
        ],
        [0, 1, 2, 3, 4, 5, 6, 7],
    )
    def SetType(self, n):
        self._typeID = n

        # Update the matrix based on the type
        if self._typeID == 0:
            self.matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        elif self._typeID == 1:
            self.matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        elif self._typeID == 2:
            self.matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        elif self._typeID == 3:
            self.matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        elif self._typeID == 4:
            self.matrix = np.array([[1, 0, -1], [0, 1, 0], [1, 0, 1]])
        elif self._typeID == 5:
            self.matrix = np.array([[-1, 0, -1], [0, -1, 0], [1, 0, -1]])
        elif self._typeID == 6:
            self.matrix = np.array([[1, 0, -1], [0, -1, 0], [1, 0, 1]])
        elif self._typeID == 7:
            self.matrix = np.array([[-1, 0, -1], [0, 1, 0], [1, 0, -1]])

        self.Modified()

    def Sample(self, x, y, z):
        """Sample the critical point model at the given point.

        Parameters:
        ----------
        x : numpy.ndarray
            x-coordinate(s) of the point(s) to sample.
        y : numpy.ndarray
            y-coordinate(s) of the point(s) to sample.
        z : numpy.ndarray
            z-coordinate(s) of the point(s) to sample.

        Returns:
        -------
        numpy.ndarray
            Array containing the sampled values at the given point(s).
            The shape of the returned array is the same as the input arrays.

        """
        shape = x.shape

        vector = self.matrix @ np.array([x, y, z]).reshape(3, -1)
        vector = vector.reshape(3, *shape)

        dx = vector[0]
        dy = vector[1]
        dz = vector[2]

        return np.stack([dx, dy, dz], axis=-1)
