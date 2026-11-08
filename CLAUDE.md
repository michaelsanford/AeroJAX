# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

AeroJAX is an interactive, JAX-native 2D CFD application (PyQt6 + pyqtgraph GUI). Solver parameters, boundary conditions, and immersed geometry can change at runtime without restarting. Every solver step is JAX-differentiable, which is what the inverse-design module relies on. The project is tuned for CPU; GPU works but is optional.

## Commands

Always work in the repo-local `.venv`, and call its interpreter explicitly because activation doesn't persist between tool calls:

```bash
python -m venv .venv && .venv/Scripts/python.exe -m pip install -r requirements.txt   # Python 3.11+ (3.14 works)
.venv/Scripts/python.exe main.py                      # launch the GUI (writes aerojax.log to the CWD)
.venv/Scripts/python.exe -m inverse_design_WIP.main   # standalone inverse-design GUI (experimental; uses package-relative imports)
```

Optional GPU support: `pip install --upgrade "jax[cuda12]"` (or `cuda13`), then run with `AEROJAX_PLATFORM=cuda`. The dependency is `pygame-ce`, not `pygame`, because `pygame` has no wheels for current Python.

The repo has no test suite, linter config, or build step. Correctness checks are the benchmark validators in `validation/`: the LDC validator compares against Ghia et al. (1982), and `vk_validator` covers von Kármán shedding. Both read trace snapshots rather than the live solver; see `validation/README.md` and `validation/example_usage.py`. GUI code can be exercised headlessly with `QT_QPA_PLATFORM=offscreen`.

## Architecture

### Two solver families with a shared duck-typed interface
- `solver/` is the incompressible Navier–Stokes finite-difference solver. `BaselineSolver` lives in `solver/simulation/base_solver.py`. Its methods are attached from sibling modules at the bottom of that file, and it is re-exported from `solver/solver.py` and `solver/__init__.py`.
- `lbm/` is Lattice Boltzmann (D2Q9/D2Q7, BGK/MRT). `LBMSolver` lives in `lbm/solver.py`.
- The GUI swaps one solver for the other by constructing a new object (`main.py` and `viewer/parameter_handlers.py`). Both classes must keep exposing the same surface the viewer calls: `step_for_visualization(...)`, `get_step_jit()`, `inject_dye`, `u`/`v`/`mask`/`current_pressure`/`iteration`, `sim_params`, `metrics_frame_skip`, and so on. If you add a viewer feature that touches the solver, implement it in both classes.

### The jitted step (two separate implementations)
- **GUI path:** `BaselineSolver.step_for_visualization` (`solver/simulation/visualization_step.py`) calls `self._step_jit(u, v, mask, dt, dyn)`. The step body is `_step`, `_step_collocated` and `_step_mac` in `solver/simulation/step_handlers.py`: advection-diffusion, divergence, pressure solve, velocity correction, then BCs.
  - **Runtime values are traced inputs.** `sdf`, `nu`, `U_inf`, `brinkman_eta`, `smagorinsky_constant` and `iteration` are collected by `step_dynamic_params()`. Changing them takes effect on the next step with no recompile. Mask and SDF are traced as well, so moving or rotating an obstacle never recompiles.
  - **Everything else is structure.** Any other value `_step` reads from `self` at trace time is listed in `_step_static_key()`. `get_step_jit()` jits a fresh closure per static key, and `step_for_visualization` re-fetches it every step. Structural changes (grid type, pressure solver, LES, `fast_mode`, `slip_walls`, …) are therefore picked up automatically and compiled lazily on the worker thread.
  - **Adding a new `self.*` read to `_step`:** add it to `step_dynamic_params()` if it is a value, or to `_step_static_key()` if it is structure (it changes Python branching or array shapes). If you skip this, the change is silently ignored.
  - **Never call `jax.clear_caches()` for parameter changes.** It was removed everywhere because it only throws away compiled work.
- **Differentiable path:** `solver/pure_functions/step.py::step_pure` (jitted with many static args) is used by `differentiable_rollout` and `design_loss` in `solver/solver.py`, which `lax.scan` over it for inverse design. Code there must stay traceable: no Python side effects and no host syncs.
- `LBMSolver.get_step_jit()` jits `_step_pure` with explicit static args. `_step_pure` must read state only from its arguments, never from `self`; reading values from `self` freezes them at trace time.
- Pressure solvers in `pressure_solvers/` (multigrid, multigrid_mac, CG, FFT, Jacobi) and the `nn` neural operator from `neural_operators/` are selected by string inside the step. They are imported with try/except availability flags (`CG_PRESSURE_AVAILABLE` and similar).
- Collocated and MAC (staggered) grids have parallel implementations: `operators.py` / `operators_mac.py`, `multigrid_solver.py` / `multigrid_solver_mac.py`, and `advection_schemes/rk3_mac.py`. Keep the two paths consistent.
- Immersed obstacles use Brinkman penalisation (`solver/brinkman.py`) applied to a smooth mask built from an SDF. `obstacles/` contains the SDF generators (NACA, cylinders, buildings, freeform drawing). The mask generator is `solver/simulation/mask_generator.py`.
- Re-dependent stability parameters (Smagorinsky C_s, hyperviscosity, Brinkman η, dt_max) come from `solver/params.py::get_re_parameters` and are applied in `BaselineSolver.__init__`. `FlowParams.resolve()` locks two of U_inf, ν, Re and derives the third.

### GUI and threading (`viewer/`)
- `main.py::BaselineViewerRefactored` is the main window. It is assembled through **mixins**: `ParameterHandlers`, `DisplayManager`, `FlowManager`, and `NACAHandler` from `viewer/*.py`. A handler method can therefore live in any of those files; grep before adding a duplicate.
- `viewer/simulation_controller.py` runs two workers on Python threads:
  - `SimulationWorker` loops `solver.step_for_visualization`.
  - `MetricsWorker` computes diagnostics off the hot path. It only receives every `metrics_frame_skip`-th frame, and `u_prev` comes through its queue.
- **The worker thread must never touch Qt widgets.** It reads `worker.ui_state`, a dict snapshot that the GUI thread refreshes on every display refresh (`SimulationController.refresh_worker_ui_state`). It pushes UI changes back through signals, such as `aoa_changed`.
- **Frames go through a latest-frame mailbox.** The worker copies fields to the host and emits `data_ready` only after the GUI has consumed the previous frame (`_frame_pending`).
  - Emitting every step floods the Qt event queue and starves Windows `WM_TIMER`s.
  - Everything the renderer draws, pressure included, comes from that payload. Don't read `solver.*` arrays on the GUI thread.
- `main.refresh_display` re-arms `refresh_timer` after each refresh so that slow frames still leave the event loop some idle time.
- "Running" means `is_simulation_running()`, i.e. a live, unpaused worker thread. The Pause button's enabled state is not a reliable signal for this.
- `viewer/state/store.py` is a Redux-style store (actions, reducers, `store.dispatch`/`subscribe`) holding obstacle and flow parameters. UI controls dispatch actions and subscribers push changes into the solver. At startup the store is seeded from `solver.sim_params` so the subscription does not overwrite solver values.
- UI widgets are in `viewer/ui_components/`. Rendering (pyqtgraph, particles, SDF overlay) is in `viewer/visualization/`.

### Decoupled analysis tools
- `trace_viewer/`: `SnapshotRecorder` writes per-step snapshots to `snapshots/` (gitignored), and `TraceViewerWindow` replays them. It never couples to the running solver.
- `neural_operators/`: learned pressure operators (linear, nonlinear, and an FNO-like "advanced" one) plus `training_utils.py`. The training UI fills its dropdown from the `*.py` files in this folder.

### Experimental / non-core
`3D/`, `3D_WIP/`, `inverse_design_WIP/`, and `timestepping/` (adaptive-dt PID/CFL controllers).

## Conventions and gotchas

- `solver/config.py` sets JAX global config on import, so it must be imported before JAX does any work. The config covers platform, precision and the persistent compilation cache, all overridable through `AEROJAX_*` environment variables. The cache defaults to `%LOCALAPPDATA%/AeroJAX/jax_cache`.
- User settings are persisted in QSettings (`viewer/settings.py`):
  - **Precision:** `main.py` exports it as `AEROJAX_PRECISION` before importing the solver. Changing it restarts the app via `QProcess.startDetached`.
  - **Theme, colormaps and window geometry** are saved and restored by `_save_user_settings` / `_restore_user_settings`.
- **Logging and errors:**
  - The code logs heavily with `print("DEBUG ...")`.
  - `main.py` filters two noisy Qt messages out of `sys.stderr`.
  - Its `sys.excepthook` keeps the app running, logs the full traceback to `aerojax.log`, and shows the error in the status bar.
- Save/Load State (`solver/state_io.py`, Ctrl+S / Ctrl+O) uses pickle and requires the same grid size.
- Known physical limits (from the README): 2D only; CL/CD values show the right trends but are not quantitatively accurate under Brinkman penalisation; best for Re < 10,000; LBM is valid only at low Mach.
- When features change, update `README.md`. It documents every GUI control and shortcut.
