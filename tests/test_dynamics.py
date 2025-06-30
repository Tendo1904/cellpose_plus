import numpy as np
import pytest
import torch

from cellpose.dynamics import masks_to_flows_gpu

CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="No CUDA device available")
def test__masks_to_flows_gpu__single_object():
    """
    Tests the masks_to_flows_gpu function for a single object using a GPU.

    This method creates a test mask array where a small region is set to 1,
    representing a single object. The test is performed on a CUDA device to verify
    the correct functionality of the masks_to_flows_gpu function when processing
    masks on a GPU.

    Parameters:
        None

    Returns:
        None
    """
    masks = np.zeros((32, 32), dtype=int)
    masks[16:18, 16:18] = 1
    masks_to_flows_gpu(masks, device=torch.device("cuda"))
