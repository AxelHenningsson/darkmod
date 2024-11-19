import numpy as np
import matplotlib.pyplot as plt 
from scipy.spatial.transform import Rotation

if __name__=='__main__':
    
    r = np.array([0,1,0])
    ang = 1e-1

    R = Rotation.from_rotvec(r*ang).as_matrix()
    
    radius = ( 2*np.pi / 3)

    Q = np.array([0,0,1])*radius
    print(Q - (R @ Q), radius - np.cos(ang)*radius)
