import sys

import pytest

from nil_dj.gpu import Adapter, list_adapters, pick_adapter

LAPTOP = [
    Adapter(0, "AMD Radeon(TM) Graphics", 512 << 20, False),
    Adapter(1, "NVIDIA GeForce RTX 3060 Laptop GPU", 4 << 30, False),
    Adapter(2, "Microsoft Basic Render Driver", 0, True),
]


def test_pick_adapter_prefers_most_vram():
    assert pick_adapter(LAPTOP).index == 1


def test_pick_adapter_ties_take_lowest_index():
    tied = [Adapter(3, "b", 1 << 30, False), Adapter(1, "a", 1 << 30, False)]
    assert pick_adapter(tied).index == 1


def test_pick_adapter_none_when_only_software():
    assert pick_adapter([LAPTOP[2]]) is None
    assert pick_adapter([]) is None


@pytest.mark.skipif(sys.platform == "win32", reason="non-Windows behaviour")
def test_list_adapters_empty_off_windows():
    assert list_adapters() == []


@pytest.mark.windows
@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "win32", reason="needs DXGI")
def test_list_adapters_finds_hardware():
    assert any(not a.software for a in list_adapters())
