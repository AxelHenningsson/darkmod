import numpy as np
import matplotlib.pyplot as plt 

def crop(image, mask):
    """For vizualisation - crop an array to content.


    Args:
        image (:obj:`numpy array`): The array to be cropped. shape=(m,n,...)
        mask (:obj:`numpy array`): The mask defining the crop. shape=(m,n)

    Returns:
        :obj:`numpy array`: returns a cropped copy shape=(m,n,...) with np.nan where mask is false.

    """
    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1
    cropped_image = image[y_min:y_max, x_min:x_max].copy()
    cropped_mask = mask[y_min:y_max, x_min:x_max]
    cropped_image[~cropped_mask] = np.nan
    return cropped_image

if __name__=='__main__':
    pass


