"""
Persistent user settings (QSettings: registry on Windows, plist/ini elsewhere).
"""

from PyQt6.QtCore import QSettings

PRECISION_KEY = "solver/precision"
THEME_DARK_KEY = "ui/dark_theme"
GEOMETRY_KEY = "ui/window_geometry"
COLORMAP_KEYS = {
    "velocity_colormap_combo": "ui/velocity_colormap",
    "vorticity_colormap_combo": "ui/vorticity_colormap",
    "pressure_colormap_combo": "ui/pressure_colormap",
}


def app_settings() -> QSettings:
    return QSettings("AeroJAX", "AeroJAX")


def saved_precision() -> str:
    """'float32' or 'float64'; read before JAX is configured."""
    value = app_settings().value(PRECISION_KEY, "float32")
    return value if value in ("float32", "float64") else "float32"
