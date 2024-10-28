darkmod - A Model of Dark-Field X-ray Microscopy 
======================================================

This package is a forward model for Dark field X-ray Microscopy allowing for simulation of diffraction based image formation from samples that are spatially extended on 3D voxeldated grids.

Examples
==============================

.. code-block:: python

    import numpy as np
    from darkmod.crystal import Crystal

    unit_cell = [4.0493, 4.0493, 4.0493, 90.0, 90.0, 90.0]
    crystal = Crystal(unit_cell, orientation=np.eye(3, 3))


installation
==============================


.. code-block:: bash

    git clone https://github.com/AxelHenningsson/darkmod.git
    cd darkmod
    pip install -e .


Documentation
======================================
Documentation can is externally hosted on this webpage
