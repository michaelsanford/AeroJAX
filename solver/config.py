"""
JAX Configuration

Configure JAX settings before any JAX imports.

Environment overrides (read once at import):
    AEROJAX_PLATFORM   'cpu' (default), 'cuda', 'gpu', 'tpu' (comma lists allowed)
    AEROJAX_PRECISION  'float32' (default) or 'float64'
    AEROJAX_CACHE_DIR  persistent compilation cache directory ('' disables)
"""

import os
from pathlib import Path

import jax


def default_cache_dir() -> str:
    """Per-user directory for JAX's persistent compilation cache."""
    base = os.environ.get('LOCALAPPDATA') or os.environ.get('XDG_CACHE_HOME') or str(Path.home() / '.cache')
    return str(Path(base) / 'AeroJAX' / 'jax_cache')


def configure_jax(platform='cpu', enable_x64=False, debug_nans=False, cache_dir=None):
    """
    Configure JAX global settings.

    Args:
        platform: 'cpu', 'gpu', or 'tpu'
        enable_x64: Enable 64-bit floats
        debug_nans: Enable NaN debugging
        cache_dir: Persistent compilation cache directory (None/'' disables).
            Compiled steps are reused across runs, so restarts and solver/grid
            switches back to a previously used configuration skip compilation.
    """
    jax.config.update('jax_platforms', platform)
    jax.config.update('jax_enable_x64', enable_x64)
    jax.config.update('jax_debug_nans', debug_nans)
    if cache_dir:
        jax.config.update('jax_compilation_cache_dir', cache_dir)
        jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.5)
        jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)


# Default configuration
configure_jax(
    platform=os.environ.get('AEROJAX_PLATFORM', 'cpu'),
    enable_x64=os.environ.get('AEROJAX_PRECISION', 'float32') == 'float64',
    debug_nans=False,
    cache_dir=os.environ.get('AEROJAX_CACHE_DIR', default_cache_dir()),
)
