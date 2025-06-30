import numpy as np
import pytest
import torch

from cellpose.dynamics import masks_to_flows_gpu

CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="No CUDA device available")
def test__masks_to_flows_gpu__single_object():
    """
    Tests the `masks_to_flows_gpu` function with a single object mask on a CUDA device.

    This test function initializes a 32x32 mask with a single object and calls
    the `masks_to_flows_gpu` function to verify its behavior when using a GPU.

    Parameters:
        None

    Returns:
        None
    """
    masks = np.zeros((32, 32), dtype=int)
    masks[16:18, 16:18] = 1
    masks_to_flows_gpu(masks, device=torch.device("cuda"))
