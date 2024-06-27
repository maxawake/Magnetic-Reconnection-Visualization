import numpy as np
from .utils import save_as_vti

class Field:
    def __init__(self, dims, extent, expr : str):
        self.dimx, self.dimy, self.dimz = dims
        self.extent = extent

        self.linx = np.linspace(extent[0], extent[1], self.dimx)
        self.liny = np.linspace(extent[2], extent[3], self.dimx)
        self.linz = np.linspace(extent[4], extent[5], self.dimx)

        self.X, self.Y, self.Z = np.meshgrid(self.linx, self.liny, self.linz, indexing="ij")

        self.field_data = np.array([eval(expr.replace("X", "self.X").replace("Y", "self.Y").replace("Z", "self.Z"))])[0]

        self.get_derivatives()

    def get_derivatives(self):
        self.jac = self.jacobian()
        self.div = self.divergence()
        self.rot = self.curl()

    def jacobian(self):
        dudx, dudy, dudz = np.gradient(self.field_data[0], self.linx, self.liny, self.linz)
        dvdx, dvdy, dvdz = np.gradient(self.field_data[1], self.linx, self.liny, self.linz)
        dwdx, dwdy, dwdz = np.gradient(self.field_data[2], self.linx, self.liny, self.linz)

        return np.array([[dudx, dudy, dudz], [dvdx, dvdy, dvdz], [dwdx, dwdy, dwdz]])

    def divergence(self):
        return np.trace(self.jac**2)

    def curl(self):
        rot_x = self.jac[2,1] - self.jac[1,2]
        rot_y = self.jac[0,2] - self.jac[2,0]
        rot_z = self.jac[1,0] - self.jac[0,1]

        return np.array([rot_x, rot_y, rot_z])
    

    def save(self, savepath="./", name="vectorfield"):
        save_as_vti(self.field_data, savepath=savepath, name=f"{name}-field")
        #save_as_vti(self.jac, savepath=savepath, name=f"{name}-jacobian")
        save_as_vti(self.div, savepath=savepath, name=f"{name}-divergence")
        save_as_vti(self.rot, savepath=savepath, name=f"{name}-rotation")
        
