import numpy as np
import pytest
import torch

from cellpose.dynamics import masks_to_flows_gpu

CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="No CUDA device available")
def test__masks_to_flows_gpu__single_object():
    """
    Tests the masks_to_flows_gpu function with a single object mask on a GPU.

        This method sets up a simple test for the masks_to_flows_gpu function
        by providing a single object mask with specific pixel values. It uses
        a CUDA device for execution, ensuring the function works correctly
        in a GPU environment.

        Parameters:
            None

        Returns:
            None
    """
    masks = np.zeros((32, 32), dtype=int)
    masks[16:18, 16:18] = 1
    masks_to_flows_gpu(masks, device=torch.device("cuda"))
