"""Which GPU to embed on. Hybrid laptops often list the integrated GPU as adapter 0, so nil-dj
enumerates the DXGI adapters and picks the one with the most dedicated video memory."""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass


@dataclass
class Adapter:
    index: int
    name: str
    dedicated_vram: int
    software: bool


def pick_adapter(adapters: list[Adapter]) -> Adapter | None:
    hardware = [a for a in adapters if not a.software]
    if not hardware:
        return None
    return min(hardware, key=lambda a: (-a.dedicated_vram, a.index))


def list_adapters() -> list[Adapter]:
    """DXGI adapters in enumeration order (the order DirectML's device_id uses). [] off Windows."""
    if sys.platform != "win32":
        return []
    return _dxgi_adapters()


# --- DXGI through ctypes (Windows only) ---------------------------------------------------------

_DXGI_ERROR_NOT_FOUND = 0x887A0002
_DXGI_ADAPTER_FLAG_SOFTWARE = 2
# Zero-based vtable slots.
_RELEASE = 2  # IUnknown::Release
_ENUM_ADAPTERS1 = 12  # IDXGIFactory1::EnumAdapters1
_GET_DESC1 = 10  # IDXGIAdapter1::GetDesc1


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32)]


class _DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", ctypes.c_uint),
        ("DeviceId", ctypes.c_uint),
        ("SubSysId", ctypes.c_uint),
        ("Revision", ctypes.c_uint),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _LUID),
        ("Flags", ctypes.c_uint),
    ]


# {770aae78-f26f-4dba-a829-253c83d1b387}
_IID_IDXGIFactory1 = _GUID(
    0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87)
)


def _method(obj: ctypes.c_void_p, slot: int, *argtypes):
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    proto = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)
    return proto(vtable[slot])


def _release(obj: ctypes.c_void_p) -> None:
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtable[_RELEASE])(obj)


def _dxgi_adapters() -> list[Adapter]:
    dxgi = ctypes.WinDLL("dxgi")
    factory = ctypes.c_void_p()
    hr = dxgi.CreateDXGIFactory1(ctypes.byref(_IID_IDXGIFactory1), ctypes.byref(factory))
    if hr != 0:
        return []
    adapters = []
    try:
        enum = _method(factory, _ENUM_ADAPTERS1, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))
        i = 0
        while True:
            adapter = ctypes.c_void_p()
            hr = enum(factory, i, ctypes.byref(adapter))
            if (hr & 0xFFFFFFFF) == _DXGI_ERROR_NOT_FOUND:
                break
            if hr != 0:
                break
            try:
                desc = _DXGI_ADAPTER_DESC1()
                get_desc = _method(adapter, _GET_DESC1, ctypes.POINTER(_DXGI_ADAPTER_DESC1))
                if get_desc(adapter, ctypes.byref(desc)) == 0:
                    adapters.append(
                        Adapter(
                            index=i,
                            name=desc.Description,
                            dedicated_vram=desc.DedicatedVideoMemory,
                            software=bool(desc.Flags & _DXGI_ADAPTER_FLAG_SOFTWARE),
                        )
                    )
            finally:
                _release(adapter)
            i += 1
    finally:
        _release(factory)
    return adapters
