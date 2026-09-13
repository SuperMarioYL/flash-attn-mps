import pytest
import torch


@pytest.fixture(autouse=True)
def mps_available():
    if not torch.backends.mps.is_available():
        pytest.skip("requires an Apple GPU with MPS")
    torch.manual_seed(123)


@pytest.fixture
def device():
    return torch.device("mps")
