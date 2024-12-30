import numpy as np


def crop(image, mask):
    """For vizualisation - crop an array to content.


    Args:
        image (:obj:`numpy array`): The array to be cropped. shape=(m,n,...)
        mask (:obj:`numpy array`): The mask defining the crop. shape=(m,n)

    Returns:
        :obj:`numpy array`: returns a cropped copy shape=(m,n,...) with np.nan where mask is false.

    """
    assert (
        mask.shape[0] == image.shape[0]
    ), "Mask and image must be of same shape along axis=0 and axis=1 to crop."
    assert (
        mask.shape[1] == image.shape[1]
    ), "Mask and image must be of same shape along axis=0 and axis=1 to crop."
    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1
    cropped_image = image[y_min:y_max, x_min:x_max].copy()
    cropped_mask = mask[y_min:y_max, x_min:x_max]
    cropped_image[~cropped_mask] = np.nan
    return cropped_image


if __name__ == "__main__":
    pass
