"""Tools for cardiac meshes, geometry, fibers, and visualization."""

from importlib import import_module

__all__ = [
    "Frame2DRenderer", "Frame3DRenderer", "AnimationBuilder",
    "PyVistaMeshGrid", "PyVistaElemGrid", "PyVistaSurfaceGrid",
]

_EXPORTS = {
    "Frame2DRenderer": ".animation",
    "Frame3DRenderer": ".animation",
    "AnimationBuilder": ".animation",
    "PyVistaMeshGrid": ".visualization",
    "PyVistaElemGrid": ".visualization",
    "PyVistaSurfaceGrid": ".visualization",
}


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(import_module(_EXPORTS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
