import numpy as np
import matplotlib.pyplot as plt


def _lab_to_Q_rot_mat(Q_lab):
    q_ll = Q_lab / np.linalg.norm(Q_lab)
    q_roll = np.array([0, 1, 0])
    q_rock = np.cross(q_roll, q_ll)
    q_rock = q_rock / np.linalg.norm(q_rock)
    return np.array((q_rock, q_roll, q_ll)).T

def lab_to_Q(lab_xyz, Q_lab):
    """Tranform from lab x,y,z coordinates to Q-system (q_rock, q_roll, q_ll)

    Args:
        lab_xyz (:obj:`numpy array`): Lab frame cartesian points. shape=(3,N)
        Q_lab (:obj:`numpy array`): The diffraction vector that will define the
            z-direction in the Q-system. shape=(3,)

    Returns:
        :obj:`numpy array`: Q-cooridnates of the xyz points as (q_rock, q_roll, q_ll)
    """
    tranformation_matrix = _lab_to_Q_rot_mat(Q_lab)
    return tranformation_matrix.T @ lab_xyz

def Q_to_lab(q_system_xyz, Q_lab):
    """Tranform from Q-system to lab x,y,z coordinates

    Args:
        q_system_xyz (:obj:`numpy array`): Q-system points. shape=(3,N)
        Q_lab (:obj:`numpy array`): The diffraction vector that will define the
            z-direction in the Q-system. shape=(3,)

    Returns:
        :obj:`numpy array`: Q-cooridnates of the xyz points as (q_rock, q_roll, q_ll)
    """
    tranformation_matrix = _lab_to_Q_rot_mat(Q_lab)
    return tranformation_matrix @ q_system_xyz

if __name__ == "__main__":
    pass