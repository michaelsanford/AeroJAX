"""
Step handler methods for BaselineSolver.
Contains time-stepping logic for collocated and MAC grids.
"""

import jax
import jax.numpy as jnp
from typing import Tuple

# Import from local modules
from ..operators import (
    grad_x, grad_y, grad_x_nonperiodic, grad_y_nonperiodic,
    divergence, divergence_nonperiodic
)
from ..les_models import dynamic_smagorinsky, constant_smagorinsky
from ..boundary_conditions import apply_taylor_green_boundary_conditions
from ..pure_functions import apply_corner_smooth_pressure_gradient

# Import pressure solvers
try:
    from pressure_solvers.multigrid_solver_mac import poisson_multigrid_mac
    MAC_PRESSURE_AVAILABLE = True
except ImportError:
    MAC_PRESSURE_AVAILABLE = False

try:
    from pressure_solvers.cg import poisson_cg_solve
    CG_PRESSURE_AVAILABLE = True
except ImportError:
    CG_PRESSURE_AVAILABLE = False

try:
    from pressure_solvers.FFT import poisson_fft_solve
    FFT_PRESSURE_AVAILABLE = True
except ImportError:
    FFT_PRESSURE_AVAILABLE = False

try:
    from pressure_solvers import poisson_multigrid, poisson_jacobi
except ImportError:
    poisson_multigrid = None
    poisson_jacobi = None

# Import MAC operators
try:
    from advection_schemes.rk3_mac import rk_step_unified_mac
    MAC_ADVECTION_AVAILABLE = True
except ImportError:
    MAC_ADVECTION_AVAILABLE = False

try:
    from solver.operators_mac import (
        divergence_staggered, divergence_nonperiodic_staggered,
        grad_x_staggered, grad_y_staggered,
        grad_x_nonperiodic_staggered, grad_y_nonperiodic_staggered
    )
    MAC_OPERATORS_AVAILABLE = True
except ImportError:
    MAC_OPERATORS_AVAILABLE = False


def _step_collocated(self, u: jnp.ndarray, v: jnp.ndarray, mask: jnp.ndarray, dt: float, dyn: dict) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single time step using projection method for collocated grid"""
    dx, dy = self.grid.dx, self.grid.dy
    
    # Compute SGS eddy viscosity (constant Smagorinsky for stability and performance)
    nu = dyn['nu']
    U_inf = dyn['U_inf']
    nu_total = nu
    nu_sgs = None
    
    if self.sim_params.use_les:
        delta = (dx * dy) ** 0.5  # Filter width for LES
        if self.sim_params.les_model == 'dynamic_smagorinsky':
            nu_sgs, _ = dynamic_smagorinsky(u, v, dx, dy, delta)
        else:
            nu_sgs = constant_smagorinsky(u, v, dx, dy, delta, dyn['smagorinsky_constant'])
        nu_total = nu + nu_sgs
    
    # Advection
    fast_mode = getattr(self.sim_params, 'fast_mode', False)
    brinkman_eta = dyn['brinkman_eta']
    advection_mask = mask  # Use mask for all flow types including LDC
    
    from advection_schemes.rk3_simple_new import rk_step_unified
    u_star, v_star = rk_step_unified(u, v, dt, nu_total, dx, dy, advection_mask, sdf=dyn['sdf'], 
                                     U_inf=U_inf, nu_sgs=nu_sgs, 
                                     nu_hyper_ratio=self.nu_hyper_ratio, 
                                     slip_walls=self.slip_walls, 
                                     fast_mode=fast_mode, 
                                     brinkman_eta=brinkman_eta,
                                     flow_type=self.sim_params.flow_type)

    # Divergence
    if self.sim_params.flow_type == 'von_karman' or self.sim_params.flow_type == 'lid_driven_cavity':
        div_star = divergence_nonperiodic(u_star, v_star, dx, dy)
    else:
        div_star = divergence(u_star, v_star, dx, dy)

    rhs = div_star / dt  # Proper scaling for projection method

    # Pressure solve
    if self.sim_params.pressure_solver == 'nn' and self.nn_pressure_model is not None:
        # Use learned pressure operator
        # Check architecture and pass appropriate arguments
        if hasattr(self, 'nn_pressure_architecture') and self.nn_pressure_architecture in ['NonLinear', 'Advanced']:
            p = self.nn_pressure_model(rhs, mask, u=u_star, v=v_star)
        else:
            p = self.nn_pressure_model(rhs, mask)
    elif self.sim_params.pressure_solver == 'jacobi':
        # Use Jacobi solver
        p = poisson_jacobi(rhs, mask, dx, dy, max_iter=self.sim_params.pressure_max_iter, 
                          tolerance=self.sim_params.pressure_tolerance, flow_type=self.sim_params.flow_type)
    elif self.sim_params.pressure_solver == 'cg' and CG_PRESSURE_AVAILABLE:
        # Use Conjugate Gradient solver
        p = poisson_cg_solve(rhs, mask, dx, dy, maxiter=self.sim_params.pressure_max_iter, 
                            tol=self.sim_params.pressure_tolerance, flow_type=self.sim_params.flow_type)
    elif self.sim_params.pressure_solver == 'fft' and FFT_PRESSURE_AVAILABLE:
        # Use FFT-based solver
        p = poisson_fft_solve(rhs, mask, dx, dy, flow_type=self.sim_params.flow_type)
        # FFT returns None if obstacles are present - fall back to multigrid
        if p is None:
            p = poisson_multigrid(rhs, mask, dx, dy, v_cycles=self.sim_params.multigrid_v_cycles, 
                                  flow_type=self.sim_params.flow_type)
    else:
        # Use collocated pressure solver
        p = poisson_multigrid(rhs, mask, dx, dy, v_cycles=self.sim_params.multigrid_v_cycles, 
                              flow_type=self.sim_params.flow_type)
    
    # Pressure correction
    if self.sim_params.flow_type == 'von_karman':
        dp_dx, dp_dy = grad_x_nonperiodic(p, dx), grad_y_nonperiodic(p, dy)
        # Apply corner smoothing to pressure gradient at inlet
        ny = p.shape[1]
        corner_smooth_width = 10
        dp_dx = dp_dx.at[0, :].set(0.0)  # No pressure gradient at inlet
        dp_dx = apply_corner_smooth_pressure_gradient(dp_dx, ny, corner_smooth_width)
    else:
        dp_dx, dp_dy = grad_x(p, dx), grad_y(p, dy)
    
    u_corr = u_star - dt * dp_dx
    v_corr = v_star - dt * dp_dy

    # Apply boundary conditions (Brinkman is now integrated into RHS computation)
    if self.sim_params.flow_type == 'von_karman':
        # Startup ramp - use cubic (smoothstep) for gentler initial ramp
        if self.startup_ramp_steps > 0:
            t = jnp.clip(dyn['iteration'] / self.startup_ramp_steps, 0.0, 1.0)
            ramp_factor = 3 * t**2 - 2 * t**3  # Cubic smoothstep (zero derivative at endpoints)
            inlet_velocity = U_inf * ramp_factor
        else:
            inlet_velocity = U_inf

        # Apply inlet BC (excluding corners - wall BC takes precedence)
        u_corr = u_corr.at[0, 1:-1].set(inlet_velocity)  # Inlet, excluding corners
        v_corr = v_corr.at[0, :].set(0.0)
        u_corr = u_corr.at[-1, :].set(u_corr.at[-2, :].get())
        v_corr = v_corr.at[-1, :].set(v_corr.at[-2, :].get())
        # Wall BCs (including corners)
        u_corr = u_corr.at[:, 0].set(0.0)  # Bottom wall (includes corner)
        u_corr = u_corr.at[:, -1].set(0.0)  # Top wall (includes corner)
        v_corr = v_corr.at[:, 0].set(0.0)
        v_corr = v_corr.at[:, -1].set(0.0)
    elif self.sim_params.flow_type == 'lid_driven_cavity':
        # Apply lid-driven cavity boundary conditions
        lid_velocity = U_inf
        # Top wall: moving lid
        u_corr = u_corr.at[:, -1].set(lid_velocity)
        v_corr = v_corr.at[:, -1].set(0.0)
        # Bottom wall: no-slip
        u_corr = u_corr.at[:, 0].set(0.0)
        v_corr = v_corr.at[:, 0].set(0.0)
        # Left wall: no-slip
        u_corr = u_corr.at[0, :].set(0.0)
        v_corr = v_corr.at[0, :].set(0.0)
        # Right wall: no-slip
        u_corr = u_corr.at[-1, :].set(0.0)
        v_corr = v_corr.at[-1, :].set(0.0)
    elif self.sim_params.flow_type == 'taylor_green':
        u_corr, v_corr = apply_taylor_green_boundary_conditions(u_corr, v_corr, U_inf, 2*jnp.pi, self.grid.nx, self.grid.ny)
    
    # Hard mask zeroing: force velocities to zero inside solid (mask < 0.5)
    # This is needed because mask has smooth sigmoid transition, so multiplication doesn't fully zero
    u_corr = jnp.where(mask > 0.5, u_corr, 0.0)
    v_corr = jnp.where(mask > 0.5, v_corr, 0.0)
    
    return u_corr, v_corr, p


def _step_mac(self, u: jnp.ndarray, v: jnp.ndarray, mask: jnp.ndarray, dt: float, dyn: dict) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single time step using projection method for MAC staggered grid"""
    dx, dy = self.grid.dx, self.grid.dy

    # Compute SGS eddy viscosity (constant Smagorinsky for stability and performance)
    nu = dyn['nu']
    U_inf = dyn['U_inf']
    nu_sgs_field = None

    if self.sim_params.use_les:
        delta = (dx * dy) ** 0.5  # Filter width for LES
        # Interpolate staggered velocities to cell centers for LES computation
        from solver.operators_mac import interpolate_to_cell_center
        u_center, v_center = interpolate_to_cell_center(u, v)
        if self.sim_params.les_model == 'dynamic_smagorinsky':
            nu_sgs_field, _ = dynamic_smagorinsky(u_center, v_center, dx, dy, delta)
        else:
            nu_sgs_field = constant_smagorinsky(u_center, v_center, dx, dy, delta, dyn['smagorinsky_constant'])
    
    # Advection
    fast_mode = getattr(self.sim_params, 'fast_mode', False)
    brinkman_eta = dyn['brinkman_eta']
    advection_mask = mask  # Use mask for all flow types including LDC

    u_star, v_star = rk_step_unified_mac(u, v, dt, nu, dx, dy, advection_mask,
                                         U_inf=U_inf, nu_sgs=nu_sgs_field,
                                         nu_hyper_ratio=self.nu_hyper_ratio,
                                         slip_walls=self.slip_walls,
                                         fast_mode=fast_mode,
                                         brinkman_eta=brinkman_eta,
                                         flow_type=self.sim_params.flow_type)

    # Divergence
    if self.sim_params.flow_type == 'von_karman' or self.sim_params.flow_type == 'lid_driven_cavity':
        div_star = divergence_nonperiodic_staggered(u_star, v_star, dx, dy)
    else:
        div_star = divergence_staggered(u_star, v_star, dx, dy)

    rhs = div_star / dt  # Proper scaling for projection method
    
    # Pressure solve
    if self.sim_params.pressure_solver == 'nn' and self.nn_pressure_model is not None:
        # Use learned pressure operator
        # Check architecture and pass appropriate arguments
        if hasattr(self, 'nn_pressure_architecture') and self.nn_pressure_architecture in ['NonLinear', 'Advanced']:
            p = self.nn_pressure_model(rhs, mask, u=u_star, v=v_star)
        else:
            p = self.nn_pressure_model(rhs, mask)
    else:
        p = poisson_multigrid_mac(rhs, mask, dx, dy, v_cycles=self.sim_params.multigrid_v_cycles, 
                                  flow_type=self.sim_params.flow_type)
    
    # Pressure correction
    if self.sim_params.flow_type == 'von_karman' or self.sim_params.flow_type == 'lid_driven_cavity':
        dp_dx = grad_x_nonperiodic_staggered(p, dx)
        dp_dy = grad_y_nonperiodic_staggered(p, dy)
        # Apply corner smoothing to pressure gradient at inlet
        if self.sim_params.flow_type == 'von_karman':
            ny = p.shape[1]
            corner_smooth_width = 10
            dp_dx = dp_dx.at[0, :].set(0.0)  # No pressure gradient at inlet
            dp_dx = apply_corner_smooth_pressure_gradient(dp_dx, ny, corner_smooth_width)
    else:
        dp_dx = grad_x_staggered(p, dx)
        dp_dy = grad_y_staggered(p, dy)
    
    u_corr = u_star - dt * dp_dx
    v_corr = v_star - dt * dp_dy

    # Apply boundary conditions (Brinkman is now integrated into RHS computation)
    if self.sim_params.flow_type == 'von_karman':
        # Startup ramp - use cubic (smoothstep) for gentler initial ramp
        if self.startup_ramp_steps > 0:
            t = jnp.clip(dyn['iteration'] / self.startup_ramp_steps, 0.0, 1.0)
            ramp_factor = 3 * t**2 - 2 * t**3  # Cubic smoothstep (zero derivative at endpoints)
            inlet_velocity = U_inf * ramp_factor
        else:
            inlet_velocity = U_inf

        # Apply inlet BC (excluding corners - wall BC takes precedence)
        u_corr = u_corr.at[0, 1:-1].set(inlet_velocity)  # Inlet, excluding corners
        v_corr = v_corr.at[0, :].set(0.0)
        u_corr = u_corr.at[-1, :].set(u_corr.at[-2, :].get())
        v_corr = v_corr.at[-1, :].set(v_corr.at[-2, :].get())
        # Wall BCs (including corners)
        u_corr = u_corr.at[:, 0].set(0.0)  # Bottom wall (includes corner)
        u_corr = u_corr.at[:, -1].set(0.0)  # Top wall (includes corner)
        v_corr = v_corr.at[:, 0].set(0.0)
        v_corr = v_corr.at[:, -1].set(0.0)
    elif self.sim_params.flow_type == 'lid_driven_cavity':
        # Apply lid-driven cavity boundary conditions
        lid_velocity = U_inf
        # Top wall: moving lid
        u_corr = u_corr.at[:, -1].set(lid_velocity)
        v_corr = v_corr.at[:, -1].set(0.0)
        # Bottom wall: no-slip
        u_corr = u_corr.at[:, 0].set(0.0)
        v_corr = v_corr.at[:, 0].set(0.0)
        # Left wall: no-slip
        u_corr = u_corr.at[0, :].set(0.0)
        v_corr = v_corr.at[0, :].set(0.0)
        # Right wall: no-slip
        u_corr = u_corr.at[-1, :].set(0.0)
        v_corr = v_corr.at[-1, :].set(0.0)
    elif self.sim_params.flow_type == 'taylor_green':
        u_corr, v_corr = apply_taylor_green_boundary_conditions(u_corr, v_corr, U_inf, 2*jnp.pi, self.grid.nx, self.grid.ny)
    
    return u_corr, v_corr, p


def _step(self, u: jnp.ndarray, v: jnp.ndarray, mask: jnp.ndarray, dt: float, dyn: dict) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single time step using projection method - delegates to grid-specific step function"""
    if self.sim_params.grid_type == 'mac' and MAC_ADVECTION_AVAILABLE and MAC_OPERATORS_AVAILABLE and MAC_PRESSURE_AVAILABLE:
        return self._step_mac(u, v, mask, dt, dyn)
    else:
        return self._step_collocated(u, v, mask, dt, dyn)


def step_dynamic_params(self) -> dict:
    """Runtime parameters passed to the jitted step as traced values.

    Changing any of these takes effect on the next step without recompiling.
    """
    return {
        'sdf': self.sdf,
        'nu': self.flow.nu,
        'U_inf': self.flow.U_inf,
        'brinkman_eta': getattr(self.sim_params, 'brinkman_eta', 0.01),
        'smagorinsky_constant': self.sim_params.smagorinsky_constant,
        'iteration': self.iteration,
    }


def _step_static_key(self) -> tuple:
    """Everything _step reads from self at trace time (structure, not values)."""
    sp = self.sim_params
    return (sp.grid_type, sp.flow_type, sp.advection_scheme, sp.pressure_solver,
            sp.pressure_max_iter, sp.pressure_tolerance, sp.multigrid_v_cycles,
            sp.use_les, sp.les_model, getattr(sp, 'fast_mode', False),
            self.slip_walls, self.nu_hyper_ratio, self.startup_ramp_steps,
            self.grid.nx, self.grid.ny, self.grid.dx, self.grid.dy,
            id(getattr(self, 'nn_pressure_model', None)),
            getattr(self, 'nn_pressure_architecture', None))


def get_step_jit(self):
    """Get the jitted step for the current static configuration.

    A fresh closure is jitted per static key, so JAX's trace cache can never
    serve a stale trace and jax.clear_caches() is not needed after changes.
    """
    key = self._step_static_key()
    if key not in self._jit_cache:
        def step(u, v, mask, dt, dyn):
            return self._step(u, v, mask, dt, dyn)
        self._jit_cache[key] = jax.jit(step)
    return self._jit_cache[key]
