import streamlit as st
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax.lax import scan, cond
import numpy as np
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import functools
from typing import Union, Tuple, Dict, List, Any, Callable, Optional
from scipy.optimize import minimize, OptimizeResult
import time
import datetime
import traceback
from collections import deque

MIN_THICKNESS_PHYS_NM = 0.01
BASE_NEEDLE_THICKNESS_NM = 0.1
DEFAULT_NEEDLE_SCAN_STEP_NM = 2.0
AUTO_NEEDLES_PER_CYCLE = 1 # Changed for single needle cycle
AUTO_MAX_CYCLES = 3
MSE_IMPROVEMENT_TOLERANCE = 1e-9
MAXITER_HARDCODED = 1000
MAXFUN_HARDCODED = 1000

def add_log_message(message: str):
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.log_messages.append(f"[{timestamp}] {message}")

def add_log(messages: Union[str, List[str]]):
    if isinstance(messages, str):
        messages = [messages]
    for msg in messages:
        if msg: 
            add_log_message(msg)

MaterialInputType = Union[complex, float, int]

def _get_nk_array_for_lambda_vec(material_definition: MaterialInputType,
                                     l_vec_target_jnp: jnp.ndarray) -> Tuple[Optional[jnp.ndarray], List[str]]:
    logs = []
    try:
        if isinstance(material_definition, (complex, float, int)):
            nk_complex = jnp.asarray(material_definition, dtype=jnp.complex128)
            if nk_complex.real <= 0:
                logs.append(f"WARNING: Constant index n'={nk_complex.real} <= 0. Using n'=1.0.")
                nk_complex = complex(1.0, nk_complex.imag)
            if nk_complex.imag < 0:
                logs.append(f"WARNING: Constant index k={nk_complex.imag} < 0. Using k=0.0.")
                nk_complex = complex(nk_complex.real, 0.0)
            result = jnp.full(l_vec_target_jnp.shape, nk_complex)
        else:
            raise TypeError(f"Unsupported material definition type: {type(material_definition)}")
        
        if jnp.any(jnp.isnan(result.real)) or jnp.any(result.real <= 0):
            logs.append(f"WARNING: n'<=0 or NaN detected. Replaced with n'=1.")
            result = jnp.where(jnp.isnan(result.real) | (result.real <= 0), 1.0 + 1j*result.imag, result)
        if jnp.any(jnp.isnan(result.imag)) or jnp.any(result.imag < 0):
            logs.append(f"WARNING: k<0 or NaN detected. Replaced with k=0.")
            result = jnp.where(jnp.isnan(result.imag) | (result.imag < 0), result.real + 0.0j, result)
        return result, logs
    except Exception as e:
        logs.append(f"Error preparing material data for '{material_definition}': {e}")
        st.error(f"Critical error preparing material '{material_definition}': {e}")
        return None, logs

def _get_nk_at_lambda(material_definition: MaterialInputType, l_nm_target: float) -> Tuple[Optional[complex], List[str]]:
    logs = []
    if l_nm_target <= 0:
        logs.append(f"Error: Target wavelength {l_nm_target}nm invalid for getting n+ik.")
        return None, logs
    l_vec_jnp = jnp.array([l_nm_target], dtype=jnp.float64)
    nk_array, prep_logs = _get_nk_array_for_lambda_vec(material_definition, l_vec_jnp)
    logs.extend(prep_logs)
    if nk_array is None:
        return None, logs
    else:
        nk_complex = complex(nk_array[0])
        return nk_complex, logs

@jax.jit
def _compute_layer_matrix_scan_step_jit(carry_matrix: jnp.ndarray, layer_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, None]:
    thickness, Ni, l_val = layer_data
    eta = Ni
    safe_l_val = jnp.maximum(l_val, 1e-9)
    phi = (2 * jnp.pi / safe_l_val) * (Ni * thickness)
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    def compute_M_layer(thickness_: jnp.ndarray) -> jnp.ndarray:
        safe_eta = jnp.where(jnp.abs(eta) < 1e-12, 1e-12 + 0j, eta)
        m01 = (1j / safe_eta) * sin_phi
        m10 = 1j * eta * sin_phi
        M_layer = jnp.array([[cos_phi, m01], [m10, cos_phi]], dtype=jnp.complex128)
        return M_layer @ carry_matrix
    def compute_identity(thickness_: jnp.ndarray) -> jnp.ndarray:
        return carry_matrix
    new_matrix = cond(thickness > 1e-12, compute_M_layer, compute_identity, thickness)
    return new_matrix, None

@jax.jit
def compute_stack_matrix_core_jax(ep_vector: jnp.ndarray, layer_indices: jnp.ndarray, l_val: jnp.ndarray) -> jnp.ndarray:
    num_layers = len(ep_vector)
    layers_scan_data = (ep_vector, layer_indices, jnp.full(num_layers, l_val))
    M_initial = jnp.eye(2, dtype=jnp.complex128)
    M_final, _ = scan(_compute_layer_matrix_scan_step_jit, M_initial, layers_scan_data)
    return M_final

@jax.jit
def calculate_single_wavelength_T_core(l_val: jnp.ndarray, ep_vector_contig: jnp.ndarray,
                                         layer_indices_at_lval: jnp.ndarray, nSub_at_lval: jnp.ndarray) -> jnp.ndarray:
    etainc = 1.0 + 0j
    etasub = nSub_at_lval
    def calculate_for_valid_l(l_: jnp.ndarray) -> jnp.ndarray:
        current_layer_indices = layer_indices_at_lval
        M = compute_stack_matrix_core_jax(ep_vector_contig, current_layer_indices, l_)
        m00, m01 = M[0, 0], M[0, 1]
        m10, m11 = M[1, 0], M[1, 1]
        rs_denominator = (etainc * m00 + etasub * m11 + etainc * etasub * m01 + m10)
        rs_denominator_abs = jnp.abs(rs_denominator)
        safe_denominator = jnp.where(rs_denominator_abs < 1e-12, 1e-12 + 0j, rs_denominator)
        ts = (2.0 * etainc) / safe_denominator
        real_etasub = jnp.real(etasub)
        real_etainc = jnp.real(etainc)
        safe_real_etainc = jnp.maximum(real_etainc, 1e-9)
        Ts_complex = (real_etasub / safe_real_etainc) * (ts * jnp.conj(ts))
        Ts = jnp.real(Ts_complex)
        return jnp.where(rs_denominator_abs < 1e-12, jnp.nan, Ts)
    def calculate_for_invalid_l(l_: jnp.ndarray) -> jnp.ndarray:
        return jnp.nan
    Ts_result = cond(l_val > 1e-9, calculate_for_valid_l, calculate_for_invalid_l, l_val)
    return Ts_result

def calculate_T_from_ep_jax(ep_vector: Union[np.ndarray, List[float]],
                              nH_material: MaterialInputType,
                              nL_material: MaterialInputType,
                              nSub_material: MaterialInputType,
                              l_vec: Union[np.ndarray, List[float]]) -> Tuple[Optional[Dict[str, np.ndarray]], List[str]]:
    logs = []
    l_vec_jnp = jnp.asarray(l_vec, dtype=jnp.float64)
    ep_vector_jnp = jnp.asarray(ep_vector, dtype=jnp.float64)
    if not l_vec_jnp.size:
        logs.append("Empty lambda vector, no T calculation performed.")
        return {'l': np.array([]), 'Ts': np.array([])}, logs
    if not ep_vector_jnp.size:
        logs.append("Empty structure (0 layers). Calculating for bare substrate.")
        nSub_arr, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_jnp)
        logs.extend(logs_sub)
        if nSub_arr is None:
            return None, logs
        Ts = jnp.ones_like(l_vec_jnp)
        return {'l': np.array(l_vec_jnp), 'Ts': np.array(Ts)}, logs
    
    nH_arr, logs_h = _get_nk_array_for_lambda_vec(nH_material, l_vec_jnp)
    logs.extend(logs_h)
    nL_arr, logs_l = _get_nk_array_for_lambda_vec(nL_material, l_vec_jnp)
    logs.extend(logs_l)
    nSub_arr, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_jnp)
    logs.extend(logs_sub)
    if nH_arr is None or nL_arr is None or nSub_arr is None:
        logs.append("Critical error: Failed to load one of the material indices.")
        return None, logs
    
    calculate_single_wavelength_T_hl_jit = jax.jit(calculate_single_wavelength_T_core)
    num_layers = len(ep_vector_jnp)
    indices_alternating = jnp.where(jnp.arange(num_layers)[:, None] % 2 == 0, nH_arr, nL_arr)
    indices_alternating_T = indices_alternating.T
    Ts_arr_raw = vmap(calculate_single_wavelength_T_hl_jit, in_axes=(0, None, 0, 0))(
        l_vec_jnp, ep_vector_jnp, indices_alternating_T, nSub_arr
    )
    Ts_arr = jnp.nan_to_num(Ts_arr_raw, nan=0.0)
    Ts_arr_clipped = jnp.clip(Ts_arr, 0.0, 1.0)
    return {'l': np.array(l_vec_jnp), 'Ts': np.array(Ts_arr_clipped)}, logs

def calculate_initial_ep(emp: Union[List[float], Tuple[float,...]], l0: float,
                           nH0_material: MaterialInputType, nL0_material: MaterialInputType) -> Tuple[Optional[np.ndarray], List[str]]:
    logs = []
    num_layers = len(emp)
    ep_initial = np.zeros(num_layers, dtype=np.float64)
    if l0 <= 0:
        logs.append(f"Warning: l0={l0} <= 0 in calculate_initial_ep. Initial thicknesses set to 0.")
        return ep_initial, logs
    nH_complex_at_l0, logs_h = _get_nk_at_lambda(nH0_material, l0)
    logs.extend(logs_h)
    nL_complex_at_l0, logs_l = _get_nk_at_lambda(nL0_material, l0)
    logs.extend(logs_l)
    if nH_complex_at_l0 is None or nL_complex_at_l0 is None:
        logs.append(f"Error: Could not get H or L indices at l0={l0}nm. Initial thicknesses set to 0.")
        st.error(f"Critical error getting indices at l0={l0}nm for initial thickness calculation.")
        return None, logs
    nH_real_at_l0 = nH_complex_at_l0.real
    nL_real_at_l0 = nL_complex_at_l0.real
    if nH_real_at_l0 <= 1e-9 or nL_real_at_l0 <= 1e-9:
        logs.append(f"WARNING: n'H({nH_real_at_l0:.3f}) or n'L({nL_real_at_l0:.3f}) at l0={l0}nm is <= 0. QWOT calculation may be incorrect.")
    for i in range(num_layers):
        multiplier = emp[i]
        is_H_layer = (i % 2 == 0)
        n_real_layer_at_l0 = nH_real_at_l0 if is_H_layer else nL_real_at_l0
        if n_real_layer_at_l0 <= 1e-9:
            ep_initial[i] = 0.0
        else:
            ep_initial[i] = multiplier * l0 / (4.0 * n_real_layer_at_l0)
    ep_initial_phys = np.where(ep_initial < MIN_THICKNESS_PHYS_NM, 0.0, ep_initial)
    num_clamped_zero = np.sum((ep_initial > 1e-12) & (ep_initial < MIN_THICKNESS_PHYS_NM))
    if num_clamped_zero > 0:
        logs.append(f"Warning: {num_clamped_zero} initial thicknesses < {MIN_THICKNESS_PHYS_NM}nm were set to 0.")
        ep_initial = np.where(ep_initial < MIN_THICKNESS_PHYS_NM, 0.0, ep_initial)
    valid_indices = True
    for i in range(num_layers):
        if emp[i] > 1e-9 and ep_initial[i] < 1e-12:
            layer_type = "H" if i % 2 == 0 else "L"
            n_val = nH_real_at_l0 if i % 2 == 0 else nL_real_at_l0
            logs.append(f"Error: Layer {i+1} ({layer_type}) has QWOT={emp[i]} but thickness=0 (likely n'({layer_type},l0)={n_val:.3f} <= 0).")
            valid_indices = False
    if not valid_indices:
        st.error("Error during initial thickness calculation due to invalid indices at l0.")
        return None, logs
    return ep_initial, logs

def calculate_qwot_from_ep(ep_vector: np.ndarray, l0: float,
                               nH0_material: MaterialInputType, nL0_material: MaterialInputType) -> Tuple[Optional[np.ndarray], List[str]]:
    logs = []
    num_layers = len(ep_vector)
    qwot_multipliers = np.full(num_layers, np.nan, dtype=np.float64)
    if l0 <= 0:
        logs.append(f"Warning: l0={l0} <= 0 in calculate_qwot_from_ep. QWOT set to NaN.")
        return qwot_multipliers, logs
    nH_complex_at_l0, logs_h = _get_nk_at_lambda(nH0_material, l0)
    logs.extend(logs_h)
    nL_complex_at_l0, logs_l = _get_nk_at_lambda(nL0_material, l0)
    logs.extend(logs_l)
    if nH_complex_at_l0 is None or nL_complex_at_l0 is None:
        logs.append(f"Error: Could not get n'H or n'L at l0={l0}nm to calculate QWOT. Returning NaN.")
        st.error(f"Error calculating QWOT: H/L indices not found at l0={l0}nm.")
        return None, logs
    nH_real_at_l0 = nH_complex_at_l0.real
    nL_real_at_l0 = nL_complex_at_l0.real
    if nH_real_at_l0 <= 1e-9 or nL_real_at_l0 <= 1e-9:
        logs.append(f"WARNING: n'H({nH_real_at_l0:.3f}) or n'L({nL_real_at_l0:.3f}) at l0={l0}nm is <= 0. QWOT calculation may be incorrect/NaN.")
    indices_ok = True
    for i in range(num_layers):
        n_real_layer_at_l0 = nH_real_at_l0 if i % 2 == 0 else nL_real_at_l0
        if n_real_layer_at_l0 <= 1e-9:
            if ep_vector[i] > 1e-9 :
                layer_type = "H" if i % 2 == 0 else "L"
                logs.append(f"Warning: Cannot calculate QWOT for layer {i+1} ({layer_type}) because n'({l0}nm) <= 0.")
                indices_ok = False
            else:
                qwot_multipliers[i] = 0.0
        else:
            qwot_multipliers[i] = ep_vector[i] * (4.0 * n_real_layer_at_l0) / l0
    if not indices_ok:
        st.warning("Some QWOT values could not be calculated (invalid indices at l0). They appear as NaN.")
        return qwot_multipliers, logs 
    else:
        return qwot_multipliers, logs

def calculate_final_rmse(res: Dict[str, np.ndarray], active_targets: List[Dict]) -> Tuple[Optional[float], int]:
    total_squared_error = 0.0
    total_points_in_targets = 0
    rmse = None
    if not active_targets or 'Ts' not in res or res['Ts'] is None or 'l' not in res or res['l'] is None:
        return rmse, total_points_in_targets
    res_l_np = np.asarray(res['l'])
    res_ts_np = np.asarray(res['Ts'])
    if res_l_np.size == 0 or res_ts_np.size == 0 or res_l_np.size != res_ts_np.size:
        return rmse, total_points_in_targets
    for target in active_targets:
        try:
            l_min = float(target['min'])
            l_max = float(target['max'])
            t_min = float(target['target_min'])
            t_max = float(target['target_max'])
            if not (0.0 <= t_min <= 1.0 and 0.0 <= t_max <= 1.0): continue
            if l_max < l_min: continue
        except (KeyError, ValueError, TypeError):
            continue
        indices = np.where((res_l_np >= l_min) & (res_l_np <= l_max))[0]
        if indices.size > 0:
            calculated_Ts_in_zone = res_ts_np[indices]
            target_lambdas_in_zone = res_l_np[indices]
            finite_mask = np.isfinite(calculated_Ts_in_zone)
            calculated_Ts_in_zone = calculated_Ts_in_zone[finite_mask]
            target_lambdas_in_zone = target_lambdas_in_zone[finite_mask]
            if calculated_Ts_in_zone.size == 0: continue
            if abs(l_max - l_min) < 1e-9: 
                interpolated_target_t = np.full_like(target_lambdas_in_zone, t_min)
            else: 
                slope = (t_max - t_min) / (l_max - l_min)
                interpolated_target_t = t_min + slope * (target_lambdas_in_zone - l_min)
            squared_errors = (calculated_Ts_in_zone - interpolated_target_t)**2
            total_squared_error += np.sum(squared_errors)
            total_points_in_targets += len(calculated_Ts_in_zone)
    if total_points_in_targets > 0:
        mse = total_squared_error / total_points_in_targets
        rmse = np.sqrt(mse)
    return rmse, total_points_in_targets

@jax.jit
def calculate_mse_for_optimization_penalized_jax(ep_vector: jnp.ndarray,
                                                     nH_arr: jnp.ndarray, nL_arr: jnp.ndarray, nSub_arr: jnp.ndarray,
                                                     l_vec_optim: jnp.ndarray,
                                                     active_targets_tuple: Tuple[Tuple[float, float, float, float], ...],
                                                     min_thickness_phys_nm: float) -> jnp.ndarray:
    below_min_mask = (ep_vector < min_thickness_phys_nm) & (ep_vector > 1e-12) 
    penalty_thin = jnp.sum(jnp.where(below_min_mask, (min_thickness_phys_nm - ep_vector)**2, 0.0))
    penalty_weight = 1e5 
    penalty_cost = penalty_thin * penalty_weight
    ep_vector_calc = jnp.maximum(ep_vector, min_thickness_phys_nm)
    num_layers = len(ep_vector_calc)
    indices_alternating = jnp.where(jnp.arange(num_layers)[:, None] % 2 == 0, nH_arr, nL_arr)
    indices_alternating_T = indices_alternating.T
    calculate_T_single_jit = jax.jit(calculate_single_wavelength_T_core)
    Ts_raw = vmap(calculate_T_single_jit, in_axes=(0, None, 0, 0))(
        l_vec_optim, ep_vector_calc, indices_alternating_T, nSub_arr
    )
    Ts = jnp.nan_to_num(Ts_raw, nan=0.0)
    total_squared_error = 0.0
    total_points_in_targets = 0
    for i in range(len(active_targets_tuple)):
        l_min, l_max, t_min, t_max = active_targets_tuple[i]
        target_mask = (l_vec_optim >= l_min) & (l_vec_optim <= l_max)
        slope = jnp.where(jnp.abs(l_max - l_min) < 1e-9, 0.0, (t_max - t_min) / (l_max - l_min))
        interpolated_target_t_full = t_min + slope * (l_vec_optim - l_min)
        squared_errors_full = (Ts - interpolated_target_t_full)**2
        masked_sq_error = jnp.where(target_mask, squared_errors_full, 0.0)
        total_squared_error += jnp.sum(masked_sq_error)
        total_points_in_targets += jnp.sum(target_mask)
    mse = jnp.where(total_points_in_targets > 0,
                        total_squared_error / total_points_in_targets,
                        jnp.inf)
    final_cost = mse + penalty_cost
    return jnp.nan_to_num(final_cost, nan=jnp.inf, posinf=jnp.inf)

def _run_core_optimization(ep_start_optim: np.ndarray,
                               validated_inputs: Dict, active_targets: List[Dict],
                               min_thickness_phys: float, log_prefix: str = ""
                               ) -> Tuple[Optional[np.ndarray], bool, float, List[str], str]:
    logs = []
    num_layers_start = len(ep_start_optim)
    optim_success = False
    final_cost = np.inf
    result_message_str = "Optimization not launched or failed early."
    final_ep = None
    if num_layers_start == 0:
        logs.append(f"{log_prefix}Cannot optimize an empty structure.")
        return None, False, np.inf, logs, "Empty structure"
    try:
        l_min_optim = validated_inputs['l_range_deb']
        l_max_optim = validated_inputs['l_range_fin']
        l_step_optim = validated_inputs['l_step']
        nH_material = validated_inputs['nH_material']
        nL_material = validated_inputs['nL_material']
        nSub_material = validated_inputs['nSub_material']
        maxiter = MAXITER_HARDCODED 
        maxfun = MAXFUN_HARDCODED
        num_pts_optim = min(max(2, int(np.round((l_max_optim - l_min_optim) / l_step_optim)) + 1), 100)
        l_vec_optim_np = np.geomspace(l_min_optim, l_max_optim, num_pts_optim)
        l_vec_optim_np = l_vec_optim_np[(l_vec_optim_np > 0) & np.isfinite(l_vec_optim_np)]
        if not l_vec_optim_np.size:
            raise ValueError("Failed to generate lambda vector for optimization.")
        l_vec_optim_jax = jnp.asarray(l_vec_optim_np)
        
        prep_start_time = time.time()
        nH_arr_optim, logs_h = _get_nk_array_for_lambda_vec(nH_material, l_vec_optim_jax)
        logs.extend(logs_h)
        nL_arr_optim, logs_l = _get_nk_array_for_lambda_vec(nL_material, l_vec_optim_jax)
        logs.extend(logs_l)
        nSub_arr_optim, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax)
        logs.extend(logs_sub)
        if nH_arr_optim is None or nL_arr_optim is None or nSub_arr_optim is None:
            raise RuntimeError("Failed to load indices for optimization.")
        
        active_targets_tuple = tuple((float(t['min']), float(t['max']), float(t['target_min']), float(t['target_max'])) for t in active_targets)
        static_args_for_jax = (
            nH_arr_optim, nL_arr_optim, nSub_arr_optim,
            l_vec_optim_jax, active_targets_tuple,
            min_thickness_phys
        )
        value_and_grad_fn = jax.jit(jax.value_and_grad(calculate_mse_for_optimization_penalized_jax))
        def scipy_obj_grad_wrapper(ep_vector_np_in, *args):
            try:
                ep_vector_jax = jnp.asarray(ep_vector_np_in, dtype=jnp.float64)
                value_jax, grad_jax = value_and_grad_fn(ep_vector_jax, *args)
                if not jnp.isfinite(value_jax):
                    value_np = np.inf
                    grad_np = np.zeros_like(ep_vector_np_in, dtype=np.float64)
                else:
                    value_np = float(np.array(value_jax))
                    grad_np_raw = np.array(grad_jax, dtype=np.float64)
                    grad_np = np.nan_to_num(grad_np_raw, nan=0.0, posinf=1e6, neginf=-1e6)
                return value_np, grad_np
            except Exception as e_wrap:
                print(f"Error in scipy_obj_grad_wrapper: {e_wrap}")
                return np.inf, np.zeros_like(ep_vector_np_in, dtype=np.float64)
        
        lbfgsb_bounds = [(min_thickness_phys, None)] * num_layers_start
        options = {'maxiter': maxiter, 'maxfun': maxfun,
                   'disp': False, 
                   'ftol': 1e-12, 'gtol': 1e-8}
        
        opt_start_time = time.time()
        result = minimize(scipy_obj_grad_wrapper,
                            ep_start_optim,
                            args=static_args_for_jax,
                            method='L-BFGS-B',
                            jac=True,
                            bounds=lbfgsb_bounds,
                            options=options)
        
        final_cost = result.fun if np.isfinite(result.fun) else np.inf
        result_message_str = result.message.decode('utf-8') if isinstance(result.message, bytes) else str(result.message)
        is_success_or_limit = (result.success or result.status == 1) and np.isfinite(final_cost)
        if is_success_or_limit:
            final_ep_raw = result.x
            final_ep = np.maximum(final_ep_raw, min_thickness_phys)
            optim_success = True
            log_status = "success" if result.success else "limit reached"
            logs.append(f"{log_prefix}Optimization finished ({log_status}). Final cost (MSE): {final_cost:.3e}, Msg: {result_message_str}")
        else:
            optim_success = False
            final_ep = np.maximum(ep_start_optim, min_thickness_phys) 
            logs.append(f"{log_prefix}Optimization FAILED. Status: {result.status}, Msg: {result_message_str}, Cost: {final_cost:.3e}")
            try:
                reverted_cost, _ = scipy_obj_grad_wrapper(final_ep, *static_args_for_jax)
                logs.append(f"{log_prefix}Reverted to initial (clamped) structure. Recalculated cost: {reverted_cost:.3e}")
                final_cost = reverted_cost if np.isfinite(reverted_cost) else np.inf
            except Exception as cost_e:
                logs.append(f"{log_prefix}Reverted to initial (clamped) structure. ERROR recalculating cost: {cost_e}")
                final_cost = np.inf
    except Exception as e_optim:
        logs.append(f"{log_prefix}Major ERROR during JAX/Scipy optimization: {e_optim}\n{traceback.format_exc(limit=2)}")
        st.error(f"Critical error during optimization: {e_optim}")
        final_ep = np.maximum(ep_start_optim, min_thickness_phys) if ep_start_optim is not None else None
        optim_success = False
        final_cost = np.inf 
        result_message_str = f"Exception: {e_optim}"
    return final_ep, optim_success, final_cost, logs, result_message_str

def _perform_layer_merge_or_removal_only(ep_vector_in: np.ndarray, min_thickness_phys: float,
                                           log_prefix: str = "", target_layer_index: Optional[int] = None,
                                           threshold_for_removal: Optional[float] = None) -> Tuple[Optional[np.ndarray], bool, List[str]]:
    current_ep = ep_vector_in.copy()
    logs = []
    num_layers = len(current_ep)
    structure_changed = False
    ep_after_merge = None 
    if num_layers <= 2 and target_layer_index is None:
        logs.append(f"{log_prefix}Structure <= 2 layers. Removal/merge not possible without target.")
        return current_ep, False, logs
    elif num_layers < 1:
        logs.append(f"{log_prefix}Empty structure.")
        return current_ep, False, logs
    try:
        thin_layer_index = -1 
        min_thickness_found = np.inf
        if target_layer_index is not None:
            if 0 <= target_layer_index < num_layers and current_ep[target_layer_index] >= min_thickness_phys:
                thin_layer_index = target_layer_index
                min_thickness_found = current_ep[target_layer_index]
                logs.append(f"{log_prefix}Manual targeting layer {thin_layer_index + 1} ({min_thickness_found:.3f} nm).")
            else:
                logs.append(f"{log_prefix}Manual target {target_layer_index+1} invalid/too thin. Auto search.")
                target_layer_index = None 
        if target_layer_index is None:
            candidate_indices = np.where(current_ep >= min_thickness_phys)[0]
            if candidate_indices.size == 0:
                logs.append(f"{log_prefix}No layer >= {min_thickness_phys:.3f} nm found.")
                return current_ep, False, logs
            candidate_thicknesses = current_ep[candidate_indices]
            indices_to_consider = candidate_indices
            thicknesses_to_consider = candidate_thicknesses
            if threshold_for_removal is not None:
                mask_below_threshold = thicknesses_to_consider < threshold_for_removal
                if np.any(mask_below_threshold):
                    indices_to_consider = indices_to_consider[mask_below_threshold]
                    thicknesses_to_consider = thicknesses_to_consider[mask_below_threshold]
                    logs.append(f"{log_prefix}Searching among layers < {threshold_for_removal:.3f} nm.")
                else:
                    logs.append(f"{log_prefix}No eligible layer (< {threshold_for_removal:.3f} nm) found.")
                    return current_ep, False, logs 
            if indices_to_consider.size > 0:
                min_idx_local = np.argmin(thicknesses_to_consider)
                thin_layer_index = indices_to_consider[min_idx_local]
                min_thickness_found = thicknesses_to_consider[min_idx_local]
            else:
                logs.append(f"{log_prefix}No final candidate layer found.")
                return current_ep, False, logs
        if thin_layer_index == -1:
            logs.append(f"{log_prefix}Failed to identify layer (unexpected case).")
            return current_ep, False, logs
        thin_layer_thickness = current_ep[thin_layer_index]
        logs.append(f"{log_prefix}Layer identified for action: Index {thin_layer_index} (Layer {thin_layer_index + 1}), thickness {thin_layer_thickness:.3f} nm.")
        if num_layers <= 2 and thin_layer_index == 0: 
            ep_after_merge = current_ep[2:] 
            merged_info = f"Removal of first 2 layers (structure size <= 2)."
            structure_changed = True
        elif num_layers <= 1 and thin_layer_index == 0: 
            ep_after_merge = np.array([]) 
            merged_info = f"Removal of the only layer."
            structure_changed = True
        elif num_layers <= 2 and thin_layer_index == 1: 
            ep_after_merge = current_ep[:-1] 
            merged_info = f"Removal of last layer (structure size 2)."
            structure_changed = True
        elif thin_layer_index == 0: 
            ep_after_merge = current_ep[2:]
            merged_info = f"Removal of first 2 layers."
            structure_changed = True
        elif thin_layer_index == num_layers - 1: 
            if num_layers >= 1: 
                ep_after_merge = current_ep[:-1] 
                merged_info = f"Removal of ONLY the last layer (Layer {num_layers})."
                structure_changed = True
            else: 
                logs.append(f"{log_prefix}Special case: cannot remove last layer (num_layers={num_layers}).")
                return current_ep, False, logs
        else: 
            merged_thickness = current_ep[thin_layer_index - 1] + current_ep[thin_layer_index + 1]
            ep_before = current_ep[:thin_layer_index - 1]
            ep_after = current_ep[thin_layer_index + 2:]
            ep_after_merge = np.concatenate((ep_before, [merged_thickness], ep_after))
            merged_info = f"Merging layers {thin_layer_index} and {thin_layer_index + 2} around removed layer {thin_layer_index + 1} -> new thickness {merged_thickness:.3f} nm."
            structure_changed = True
        if structure_changed and ep_after_merge is not None:
            logs.append(f"{log_prefix}{merged_info} New structure: {len(ep_after_merge)} layers.")
            ep_after_merge = np.maximum(ep_after_merge, min_thickness_phys) 
            return ep_after_merge, True, logs
        elif structure_changed and ep_after_merge is None: 
            logs.append(f"{log_prefix}Logic error: structure_changed=True but ep_after_merge=None.")
            return current_ep, False, logs
        else: 
            logs.append(f"{log_prefix}No structure modification performed.")
            return current_ep, False, logs
    except Exception as e_merge:
        logs.append(f"{log_prefix}ERROR during merge/removal logic: {e_merge}\n{traceback.format_exc(limit=1)}")
        st.error(f"Internal error during layer removal/merge: {e_merge}")
        return current_ep, False, logs 

def _perform_needle_insertion_scan(ep_vector_in: np.ndarray,
                                       nH_material: MaterialInputType, nL_material: MaterialInputType, nSub_material: MaterialInputType,
                                       l_vec_optim_np: np.ndarray, active_targets: List[Dict],
                                       cost_function_jax: Callable,
                                       min_thickness_phys: float, base_needle_thickness_nm: float,
                                       scan_step: float, l0_repr: float,
                                       log_prefix: str = ""
                                       ) -> Tuple[Optional[np.ndarray], float, List[str], int]:
    logs = []
    num_layers_in = len(ep_vector_in)
    if num_layers_in == 0:
        logs.append(f"{log_prefix}Needle scan impossible on empty structure.")
        return None, np.inf, logs, -1
    logs.append(f"{log_prefix}Starting needle scan ({num_layers_in} layers). Step: {scan_step} nm, needle thick: {base_needle_thickness_nm:.3f} nm.")
    try:
        l_vec_optim_jax = jnp.asarray(l_vec_optim_np)
        nH_arr_optim, logs_h = _get_nk_array_for_lambda_vec(nH_material, l_vec_optim_jax)
        logs.extend(logs_h)
        nL_arr_optim, logs_l = _get_nk_array_for_lambda_vec(nL_material, l_vec_optim_jax)
        logs.extend(logs_l)
        nSub_arr_optim, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax)
        logs.extend(logs_sub)
        if nH_arr_optim is None or nL_arr_optim is None or nSub_arr_optim is None:
            raise RuntimeError("Failed to load indices for needle scan.")
        active_targets_tuple = tuple((float(t['min']), float(t['max']), float(t['target_min']), float(t['target_max'])) for t in active_targets)
        static_args_cost_fn = (
            nH_arr_optim, nL_arr_optim, nSub_arr_optim,
            l_vec_optim_jax, active_targets_tuple,
            min_thickness_phys
        )
        initial_cost_jax = cost_function_jax(jnp.asarray(ep_vector_in), *static_args_cost_fn)
        initial_cost = float(np.array(initial_cost_jax))
        if not np.isfinite(initial_cost):
            logs.append(f"{log_prefix} ERROR: Initial cost not finite ({initial_cost}). Scan aborted.")
            st.error("Needle Scan Error: Cost of starting structure is not finite.")
            return None, np.inf, logs, -1
        logs.append(f"{log_prefix} Initial cost: {initial_cost:.6e}")
    except Exception as e_prep:
        logs.append(f"{log_prefix} ERROR preparing needle scan: {e_prep}")
        st.error(f"Error preparing needle scan: {e_prep}")
        return None, np.inf, logs, -1
    best_ep_found = None
    min_cost_found = initial_cost
    best_insertion_idx = -1
    tested_insertions = 0
    ep_cumsum = np.cumsum(ep_vector_in)
    total_thickness = ep_cumsum[-1] if num_layers_in > 0 else 0.0
    for z in np.arange(scan_step, total_thickness, scan_step):
        current_layer_idx = -1
        layer_start_z = 0.0
        for i in range(num_layers_in):
            layer_end_z = ep_cumsum[i]
            if z > layer_start_z and z <= layer_end_z:
                t_part1 = z - layer_start_z
                t_part2 = layer_end_z - z
                if t_part1 >= min_thickness_phys and t_part2 >= min_thickness_phys:
                    current_layer_idx = i
                else:
                    current_layer_idx = -2
                break
            layer_start_z = layer_end_z
        if current_layer_idx < 0:
            continue
        tested_insertions += 1
        t_layer_split_1 = z - (ep_cumsum[current_layer_idx-1] if current_layer_idx > 0 else 0.0)
        t_layer_split_2 = ep_cumsum[current_layer_idx] - z
        ep_temp_np = np.concatenate((
            ep_vector_in[:current_layer_idx],
            [t_layer_split_1, base_needle_thickness_nm, t_layer_split_2],
            ep_vector_in[current_layer_idx+1:]
        ))
        ep_temp_np_clamped = np.maximum(ep_temp_np, min_thickness_phys)
        try:
            current_cost_jax = cost_function_jax(jnp.asarray(ep_temp_np_clamped), *static_args_cost_fn)
            current_cost = float(np.array(current_cost_jax))
            if np.isfinite(current_cost) and current_cost < min_cost_found:
                min_cost_found = current_cost
                best_ep_found = ep_temp_np_clamped.copy()
                best_insertion_idx = current_layer_idx
        except Exception as e_cost:
            logs.append(f"{log_prefix} WARNING: Failed cost calculation for z={z:.2f}. {e_cost}")
            continue
    if best_ep_found is not None:
        improvement = initial_cost - min_cost_found
        logs.append(f"{log_prefix} Scan finished. {tested_insertions} points tested.")
        logs.append(f"{log_prefix} Best improvement found: {improvement:.6e} (MSE {min_cost_found:.6e})")
        logs.append(f"{log_prefix} Optimal insertion in original layer {best_insertion_idx + 1}.")
        return best_ep_found, min_cost_found, logs, best_insertion_idx
    else:
        logs.append(f"{log_prefix} Scan finished. {tested_insertions} points tested. No improvement found.")
        return None, initial_cost, logs, -1

def _run_needle_iterations(ep_start: np.ndarray, num_needles: int,
                               validated_inputs: Dict, active_targets: List[Dict],
                               min_thickness_phys: float, l_vec_optim_np_in: np.ndarray,
                               scan_step_nm: float, base_needle_thickness_nm: float,
                               log_prefix: str = ""
                               ) -> Tuple[np.ndarray, float, List[str]]:
    logs = []
    best_ep_overall = np.asarray(ep_start).copy()
    best_mse_overall = np.inf
    nH_material = validated_inputs['nH_material']
    nL_material = validated_inputs['nL_material']
    nSub_material = validated_inputs['nSub_material']
    l0_repr = validated_inputs.get('l0', 500.0)
    l_step_optim = validated_inputs['l_step']
    l_min_optim_glob, l_max_optim_glob = validated_inputs['l_range_deb'], validated_inputs['l_range_fin']
    num_pts_needle = min(max(2, int(np.round((l_max_optim_glob - l_min_optim_glob) / l_step_optim)) + 1), 100)
    l_vec_optim_np = np.geomspace(l_min_optim_glob, l_max_optim_glob, num_pts_needle)
    l_vec_optim_np = l_vec_optim_np[(l_vec_optim_np > 0) & np.isfinite(l_vec_optim_np)]
    if not l_vec_optim_np.size:
        logs.append(f"{log_prefix} ERROR: Failed to generate lambda vector for needle iterations.")
        st.error("Needle Iterations Error: Lambda vector generation failed.")
        return ep_start, np.inf, logs
    cost_fn_penalized_jit = jax.jit(calculate_mse_for_optimization_penalized_jax)
    try:
        l_vec_optim_jax = jnp.asarray(l_vec_optim_np)
        nH_arr, logs_h = _get_nk_array_for_lambda_vec(nH_material, l_vec_optim_jax)
        logs.extend(logs_h)
        nL_arr, logs_l = _get_nk_array_for_lambda_vec(nL_material, l_vec_optim_jax)
        logs.extend(logs_l)
        nSub_arr, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax)
        logs.extend(logs_sub)
        if nH_arr is None or nL_arr is None or nSub_arr is None:
            raise RuntimeError("Failed to load indices for needle iterations.")
        active_targets_tuple = tuple((float(t['min']), float(t['max']), float(t['target_min']), float(t['target_max'])) for t in active_targets)
        static_args_cost_fn = (nH_arr, nL_arr, nSub_arr, l_vec_optim_jax, active_targets_tuple, min_thickness_phys)
        initial_cost_jax = cost_fn_penalized_jit(jnp.asarray(best_ep_overall), *static_args_cost_fn)
        best_mse_overall = float(np.array(initial_cost_jax))
        if not np.isfinite(best_mse_overall):
            raise ValueError("Initial MSE for needle iterations is not finite.")
        logs.append(f"{log_prefix} Starting needle iterations ({num_needles} max). Initial MSE: {best_mse_overall:.6e}")
    except Exception as e_init:
        logs.append(f"{log_prefix} ERROR calculating initial MSE for needle iterations: {e_init}")
        st.error(f"Error initializing needle iterations: {e_init}")
        return ep_start, np.inf, logs
    for i in range(num_needles):
        logs.append(f"{log_prefix} --- Needle Iteration {i + 1}/{num_needles} ---")
        current_ep_iter = best_ep_overall.copy()
        num_layers_current = len(current_ep_iter)
        if num_layers_current == 0:
            logs.append(f"{log_prefix} Empty structure, stopping needle iterations."); break
        
        ep_after_scan, cost_after_scan, scan_logs, inserted_idx = _perform_needle_insertion_scan(
            current_ep_iter,
            nH_material, nL_material, nSub_material,
            l_vec_optim_np, active_targets,
            cost_fn_penalized_jit,
            min_thickness_phys, base_needle_thickness_nm, scan_step_nm, l0_repr,
            log_prefix=f"{log_prefix}  [Scan {i+1}] "
        )
        logs.extend(scan_logs)
        if ep_after_scan is None:
            logs.append(f"{log_prefix} Needle scan {i + 1} found no improvement. Stopping needle iterations."); break
        logs.append(f"{log_prefix} Scan {i + 1} found potential improvement. Re-optimizing...")
        
        ep_after_reopt, optim_success, final_cost_reopt, optim_logs, optim_status_msg = \
            _run_core_optimization(ep_after_scan, validated_inputs, active_targets,
                                     min_thickness_phys, log_prefix=f"{log_prefix}  [Re-Opt {i+1}] ")
        logs.extend(optim_logs)
        if not optim_success:
            logs.append(f"{log_prefix} Re-optimization after scan {i + 1} FAILED. Stopping needle iterations."); break
        logs.append(f"{log_prefix} Re-optimization {i + 1} successful. New MSE: {final_cost_reopt:.6e}.")
        if final_cost_reopt < best_mse_overall - MSE_IMPROVEMENT_TOLERANCE:
            logs.append(f"{log_prefix}  MSE improved compared to previous best ({best_mse_overall:.6e}). Updating.")
            best_ep_overall = ep_after_reopt.copy()
            best_mse_overall = final_cost_reopt
        else:
            logs.append(f"{log_prefix}  New MSE ({final_cost_reopt:.6e}) not significantly better than previous ({best_mse_overall:.6e}). Stopping needle iterations.")
            best_ep_overall = ep_after_reopt.copy()
            best_mse_overall = final_cost_reopt
            break
    logs.append(f"{log_prefix} End of needle iterations. Best final MSE: {best_mse_overall:.6e}")
    return best_ep_overall, best_mse_overall, logs

def validate_targets() -> Optional[List[Dict]]:
    active_targets = []
    logs = []
    is_valid = True
    if 'targets' not in st.session_state or not isinstance(st.session_state.targets, list):
        st.error("Internal error: Target list missing or invalid in session_state.")
        return None
    for i, target_state in enumerate(st.session_state.targets):
        if target_state.get('enabled', False):
            try:
                l_min = float(target_state['min'])
                l_max = float(target_state['max'])
                t_min = float(target_state['target_min'])
                t_max = float(target_state['target_max'])
                if l_max < l_min:
                    logs.append(f"Target {i+1} Error: λ max ({l_max:.1f}) < λ min ({l_min:.1f}).")
                    is_valid = False; continue
                if not (0.0 <= t_min <= 1.0 and 0.0 <= t_max <= 1.0):
                    logs.append(f"Target {i+1} Error: Transmittance out of [0, 1] (Tmin={t_min:.2f}, Tmax={t_max:.2f}).")
                    is_valid = False; continue
                active_targets.append({
                    'min': l_min, 'max': l_max,
                    'target_min': t_min, 'target_max': t_max
                })
            except (KeyError, ValueError, TypeError) as e:
                logs.append(f"Target {i+1} Error: Missing or invalid data ({e}).")
                is_valid = False; continue
    if not is_valid:
        st.warning("Errors exist in the active spectral target definitions. Please correct.")
        return None
    elif not active_targets:
        return []
    else:
        return active_targets

def get_lambda_range_from_targets(validated_targets: Optional[List[Dict]]) -> Tuple[Optional[float], Optional[float]]:
    overall_min, overall_max = None, None
    if validated_targets:
        all_mins = [t['min'] for t in validated_targets]
        all_maxs = [t['max'] for t in validated_targets]
        if all_mins: overall_min = min(all_mins)
        if all_maxs: overall_max = max(all_maxs)
    return overall_min, overall_max

def clear_optimized_state():
    st.session_state.optimized_ep = None
    st.session_state.is_optimized_state = False
    st.session_state.ep_history = deque(maxlen=5)
    st.session_state.optimized_qwot_str = ""
    st.session_state.last_rmse = None
    st.session_state.monte_carlo_results = None

def set_optimized_as_nominal_wrapper():
    if not st.session_state.get('is_optimized_state') or st.session_state.get('optimized_ep') is None:
        st.error("No valid optimized structure to set as nominal.")
        return
    try:
        l0 = st.session_state.l0
        nH_r = st.session_state.get('nH_r', 2.35)
        nH_i = st.session_state.get('nH_i', 0.0)
        nL_r = st.session_state.get('nL_r', 1.46)
        nL_i = st.session_state.get('nL_i', 0.0)
        nH_mat = complex(nH_r, nH_i)
        nL_mat = complex(nL_r, nL_i)
        
        optimized_qwots, logs_qwot = calculate_qwot_from_ep(st.session_state.optimized_ep, l0, nH_mat, nL_mat)
        if optimized_qwots is None:
            st.error("Error recalculating QWOT from the optimized structure.")
            return
        if np.any(np.isnan(optimized_qwots)):
            st.warning("Recalculated QWOT contains NaNs (likely invalid index at l0). Nominal QWOT not updated.")
        else:
            new_qwot_str = ",".join([f"{q:.6f}" for q in optimized_qwots])
            st.session_state.current_qwot = new_qwot_str
            st.success("Optimized structure set as new Nominal (QWOT updated).")
            clear_optimized_state()
    except Exception as e:
        st.error(f"Unexpected error setting optimized as nominal: {e}")

def undo_remove_wrapper():
    if not st.session_state.get('ep_history'):
        st.info("Undo history is empty.")
        return
    try:
        last_ep = st.session_state.ep_history.pop()
        st.session_state.optimized_ep = last_ep.copy()
        st.session_state.is_optimized_state = True
        l0 = st.session_state.l0
        nH_r = st.session_state.get('nH_r', 2.35)
        nH_i = st.session_state.get('nH_i', 0.0)
        nL_r = st.session_state.get('nL_r', 1.46)
        nL_i = st.session_state.get('nL_i', 0.0)
        nH_mat = complex(nH_r, nH_i)
        nL_mat = complex(nL_r, nL_i)
        
        qwots_recalc, logs_qwot = calculate_qwot_from_ep(last_ep, l0, nH_mat, nL_mat)
        if qwots_recalc is not None and not np.any(np.isnan(qwots_recalc)):
            st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_recalc])
        else:
            st.session_state.optimized_qwot_str = "QWOT N/A (after undo)"

        st.info("State restored. Recalculating...")
        st.session_state.needs_rerun_calc = True
        st.session_state.rerun_calc_params = {
            'is_optimized_run': True,
            'method_name': "Optimized (Undo)",
            'force_ep': st.session_state.optimized_ep
            }
    except IndexError:
        st.warning("Undo history is empty (internal error?).")
    except Exception as e:
        st.error(f"Unexpected error during undo: {e}")
        clear_optimized_state()

def run_calculation_wrapper(is_optimized_run: bool, method_name: str = "", force_ep: Optional[np.ndarray] = None):
    calc_type = 'Optimized' if is_optimized_run else 'Nominal'
    st.session_state.last_calc_results = {}
    st.session_state.last_rmse = None
    st.session_state.monte_carlo_results = None # Clear previous MC results
    with st.spinner(f"{calc_type} calculation in progress..."):
        try:
            active_targets = validate_targets()
            if active_targets is None:
                st.error("Target definition invalid. Check logs and correct.")
                return
            if not active_targets:
                st.warning("No active targets. Default lambda range used (400-700nm). RMSE calculation will be N/A.")
                l_min_plot, l_max_plot = 400.0, 700.0
            else:
                l_min_plot, l_max_plot = get_lambda_range_from_targets(active_targets)
                if l_min_plot is None or l_max_plot is None or l_max_plot < l_min_plot:
                    st.error("Could not determine a valid lambda range from targets.")
                    return
            
            nH_r = st.session_state.get('nH_r', 2.35)
            nH_i = st.session_state.get('nH_i', 0.0)
            nL_r = st.session_state.get('nL_r', 1.46)
            nL_i = st.session_state.get('nL_i', 0.0)
            nSub_r = st.session_state.get('nSub_r', 1.52)
            nH_mat = complex(nH_r, nH_i)
            nL_mat = complex(nL_r, nL_i)
            nSub_mat = complex(nSub_r, 0.0)
            
            validated_inputs = {
                'l0': st.session_state.l0,
                'l_step': st.session_state.l_step,
                'emp_str': st.session_state.current_qwot,
                'auto_thin_threshold': st.session_state.auto_thin_threshold,
                'l_range_deb': l_min_plot,
                'l_range_fin': l_max_plot,
                'nH_material': nH_mat,
                'nL_material': nL_mat,
                'nSub_material': nSub_mat,
            }

            ep_to_calculate = None
            if force_ep is not None:
                ep_to_calculate = force_ep.copy()
            elif is_optimized_run and st.session_state.get('optimized_ep') is not None:
                ep_to_calculate = st.session_state.optimized_ep.copy()
            else:
                emp_list = [float(e.strip()) for e in validated_inputs['emp_str'].split(',') if e.strip()]
                if not emp_list and calc_type == 'Nominal':
                    ep_to_calculate = np.array([], dtype=np.float64)
                elif not emp_list and calc_type == 'Optimized':
                    st.error("Cannot start an optimized calculation if nominal QWOT is empty and no previous optimized state exists.")
                    return
                else:
                    ep_calc, logs_ep_init = calculate_initial_ep(emp_list, validated_inputs['l0'], nH_mat, nL_mat)
                    if ep_calc is None:
                        st.error("Failed to calculate initial thicknesses from QWOT.")
                        return
                    ep_to_calculate = ep_calc.copy()
            st.session_state.current_ep = ep_to_calculate.copy() if ep_to_calculate is not None else None
            num_plot_points = max(501, int(np.round((l_max_plot - l_min_plot) / validated_inputs['l_step'])) * 3 + 1)
            l_vec_plot_fine_np = np.linspace(l_min_plot, l_max_plot, num_plot_points)
            l_vec_plot_fine_np = l_vec_plot_fine_np[(l_vec_plot_fine_np > 0) & np.isfinite(l_vec_plot_fine_np)]
            if not l_vec_plot_fine_np.size:
                st.error("Could not generate a valid lambda vector for plotting.")
                return
            start_calc_time = time.time()
            results_fine, calc_logs = calculate_T_from_ep_jax(
                ep_to_calculate, nH_mat, nL_mat, nSub_mat, l_vec_plot_fine_np
            )
            add_log(calc_logs)
            if results_fine is None:
                st.error("Main transmittance calculation failed.")
                return
            st.session_state.last_calc_results = {
                'res_fine': results_fine,
                'method_name': method_name,
                'ep_used': ep_to_calculate.copy() if ep_to_calculate is not None else None,
                'l0_used': validated_inputs['l0'],
                'nH_used': nH_mat, 'nL_used': nL_mat, 'nSub_used': nSub_mat,
            }
            if active_targets:
                num_pts_optim_display = min(max(2, int(np.round((l_max_plot - l_min_plot) / validated_inputs['l_step'])) + 1), 100)
                l_vec_optim_np_display = np.geomspace(l_min_plot, l_max_plot, num_pts_optim_display)
                l_vec_optim_np_display = l_vec_optim_np_display[(l_vec_optim_np_display > 0) & np.isfinite(l_vec_optim_np_display)]
                if l_vec_optim_np_display.size > 0:
                    results_optim_grid, logs_rmse_calc = calculate_T_from_ep_jax(
                        ep_to_calculate, nH_mat, nL_mat, nSub_mat, l_vec_optim_np_display
                    )
                    add_log(logs_rmse_calc)
                    if results_optim_grid is not None:
                        rmse_display, num_pts_rmse = calculate_final_rmse(results_optim_grid, active_targets)
                        st.session_state.last_rmse = rmse_display
                        st.session_state.last_calc_results['res_optim_grid'] = results_optim_grid
                    else:
                        st.session_state.last_rmse = None
                else:
                    st.session_state.last_rmse = None
            else:
                st.session_state.last_rmse = None
            st.session_state.is_optimized_state = is_optimized_run
            if not is_optimized_run:
                st.session_state.optimized_ep = None
                st.session_state.optimized_qwot_str = ""
                st.session_state.current_ep = ep_to_calculate.copy() if ep_to_calculate is not None else None
            add_log(f"{calc_type} calculation finished.")
        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Error during {calc_type} calculation: {e}")
            add_log(f"ERROR: {e}")
        except Exception as e_fatal:
            st.error(f"Unexpected error during {calc_type} calculation: {e_fatal}")
            add_log(f"FATAL ERROR: {e_fatal}")
        finally:
            pass

def run_local_optimization_wrapper():
    st.session_state.last_calc_results = {}
    st.session_state.last_rmse = None
    clear_optimized_state()

    with st.spinner("Local optimization in progress..."):
        try:
            active_targets = validate_targets()
            if active_targets is None or not active_targets:
                st.error("Local optimization requires active and valid targets.")
                return
            l_min_opt, l_max_opt = get_lambda_range_from_targets(active_targets)
            if l_min_opt is None:
                st.error("Could not determine lambda range for optimization.")
                return
            
            nH_r = st.session_state.get('nH_r', 2.35)
            nH_i = st.session_state.get('nH_i', 0.0)
            nL_r = st.session_state.get('nL_r', 1.46)
            nL_i = st.session_state.get('nL_i', 0.0)
            nSub_r = st.session_state.get('nSub_r', 1.52)
            nH_mat = complex(nH_r, nH_i)
            nL_mat = complex(nL_r, nL_i)
            nSub_mat = complex(nSub_r, 0.0)

            validated_inputs = {
                'l0': st.session_state.l0, 'l_step': st.session_state.l_step,
                'emp_str': st.session_state.current_qwot,
                'auto_thin_threshold': st.session_state.auto_thin_threshold,
                'l_range_deb': l_min_opt, 'l_range_fin': l_max_opt,
                'nH_material': nH_mat,
                'nL_material': nL_mat,
                'nSub_material': nSub_mat,
            }
            
            emp_list = [float(e.strip()) for e in validated_inputs['emp_str'].split(',') if e.strip()]
            if not emp_list:
                st.error("Nominal QWOT empty, cannot start local optimization.")
                return
            ep_start, logs_ep_init = calculate_initial_ep(emp_list, validated_inputs['l0'], nH_mat, nL_mat)
            add_log(logs_ep_init)
            if ep_start is None:
                st.error("Failed initial thickness calculation for local optimization.")
                return
            
            final_ep, success, final_cost, optim_logs, msg = \
                _run_core_optimization(ep_start, validated_inputs, active_targets,
                                         MIN_THICKNESS_PHYS_NM, log_prefix="  [Opt Local] ")
            add_log(optim_logs)
            
            if success and final_ep is not None:
                st.session_state.optimized_ep = final_ep.copy()
                st.session_state.current_ep = final_ep.copy()
                st.session_state.is_optimized_state = True
                st.session_state.last_rmse = np.sqrt(final_cost) if final_cost != np.inf else np.inf
                qwots_opt, logs_qwot = calculate_qwot_from_ep(final_ep, validated_inputs['l0'], nH_mat, nL_mat)
                add_log(logs_qwot)
                if qwots_opt is not None and not np.any(np.isnan(qwots_opt)):
                    st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_opt])
                else:
                    st.session_state.optimized_qwot_str = "QWOT N/A"
                st.success(f"Local optimization finished ({msg}). RMSE: {st.session_state.last_rmse:.4e}")
                st.session_state.needs_rerun_calc = True
                st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': f"Opt Local"}
            else:
                st.error(f"Local optimization failed: {msg}")
                st.session_state.is_optimized_state = False
                st.session_state.optimized_ep = None
                st.session_state.current_ep = ep_start.copy()
                st.session_state.last_rmse = None
        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Error during local optimization: {e}")
            add_log(f"ERROR: {e}")
            clear_optimized_state()
        except Exception as e_fatal:
            st.error(f"Unexpected error during local optimization: {e_fatal}")
            add_log(f"FATAL ERROR: {e_fatal}")
            clear_optimized_state()

def run_needle_wrapper():
    st.session_state.last_calc_results = {}
    st.session_state.last_rmse = None
    with st.spinner("Needle cycle in progress..."):
        try:
            active_targets = validate_targets()
            if active_targets is None or not active_targets:
                st.error("Needle optimization requires active and valid targets.")
                return
            l_min_opt, l_max_opt = get_lambda_range_from_targets(active_targets)
            if l_min_opt is None:
                st.error("Could not determine lambda range for Needle optimization.")
                return
            
            nH_r = st.session_state.get('nH_r', 2.35)
            nH_i = st.session_state.get('nH_i', 0.0)
            nL_r = st.session_state.get('nL_r', 1.46)
            nL_i = st.session_state.get('nL_i', 0.0)
            nSub_r = st.session_state.get('nSub_r', 1.52)
            nH_mat = complex(nH_r, nH_i)
            nL_mat = complex(nL_r, nL_i)
            nSub_mat = complex(nSub_r, 0.0)

            validated_inputs = {
                'l0': st.session_state.l0, 'l_step': st.session_state.l_step,
                'emp_str': st.session_state.current_qwot,
                'auto_thin_threshold': st.session_state.auto_thin_threshold,
                'l_range_deb': l_min_opt, 'l_range_fin': l_max_opt,
                'nH_material': nH_mat,
                'nL_material': nL_mat,
                'nSub_material': nSub_mat,
            }

            ep_start_needle = None
            if st.session_state.get('is_optimized_state') and st.session_state.get('optimized_ep') is not None:
                ep_start_needle = st.session_state.optimized_ep.copy()
                add_log("Needle Cycle: Starting from existing optimized structure.")
            else:
                emp_list = [float(e.strip()) for e in validated_inputs['emp_str'].split(',') if e.strip()]
                if not emp_list:
                    st.error("Nominal QWOT is empty. Cannot start Needle cycle.")
                    return
                ep_start_needle, logs_ep_init = calculate_initial_ep(emp_list, validated_inputs['l0'], nH_mat, nL_mat)
                add_log(logs_ep_init)
                if ep_start_needle is None:
                    st.error("Failed to calculate initial thicknesses for Needle cycle.")
                    return
                add_log("Needle Cycle: Starting from nominal structure.")

            l_vec_optim_np = np.geomspace(l_min_opt, l_max_opt, 100)
            final_ep, final_mse, needle_logs = _run_needle_iterations(
                ep_start=ep_start_needle,
                num_needles=1,
                validated_inputs=validated_inputs,
                active_targets=active_targets,
                min_thickness_phys=MIN_THICKNESS_PHYS_NM,
                l_vec_optim_np_in=l_vec_optim_np,
                scan_step_nm=DEFAULT_NEEDLE_SCAN_STEP_NM,
                base_needle_thickness_nm=BASE_NEEDLE_THICKNESS_NM,
                log_prefix="[Needle Cycle]"
            )
            add_log(needle_logs)

            if final_ep is not None and np.isfinite(final_mse):
                st.session_state.optimized_ep = final_ep.copy()
                st.session_state.current_ep = final_ep.copy()
                st.session_state.is_optimized_state = True
                st.session_state.last_rmse = np.sqrt(final_mse) if final_mse != np.inf else np.inf
                qwots_opt, logs_qwot = calculate_qwot_from_ep(final_ep, validated_inputs['l0'], nH_mat, nL_mat)
                add_log(logs_qwot)
                if qwots_opt is not None and not np.any(np.isnan(qwots_opt)):
                    st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_opt])
                else:
                    st.session_state.optimized_qwot_str = "QWOT N/A"
                st.success(f"Needle cycle finished. Final RMSE: {st.session_state.last_rmse:.4e}")
                st.session_state.needs_rerun_calc = True
                st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': "Needle Cycle"}
            else:
                st.error("Needle cycle failed or did not produce a valid result.")
                clear_optimized_state()
                st.session_state.needs_rerun_calc = True
                st.session_state.rerun_calc_params = {'is_optimized_run': False, 'method_name': "Nominal (After Needle Fail)"}
        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Error during Needle cycle: {e}")
            add_log(f"ERROR: {e}")
        except Exception as e_fatal:
            st.error(f"Unexpected error during Needle cycle: {e_fatal}")
            add_log(f"FATAL ERROR: {e_fatal}")
        finally:
            pass

def run_remove_thin_wrapper():
    st.session_state.last_calc_results = {}
    st.session_state.last_rmse = None
    ep_start_removal = None
    is_starting_from_optimized = False
    if st.session_state.get('is_optimized_state') and st.session_state.get('optimized_ep') is not None:
        ep_start_removal = st.session_state.optimized_ep.copy()
        is_starting_from_optimized = True
    else:
        try:
            nH_r = st.session_state.get('nH_r', 2.35)
            nH_i = st.session_state.get('nH_i', 0.0)
            nL_r = st.session_state.get('nL_r', 1.46)
            nL_i = st.session_state.get('nL_i', 0.0)
            nH_mat_temp = complex(nH_r, nH_i)
            nL_mat_temp = complex(nL_r, nL_i)
            
            emp_list_temp = [float(e.strip()) for e in st.session_state.current_qwot.split(',') if e.strip()]
            if not emp_list_temp:
                ep_start_removal = np.array([], dtype=np.float64) 
            else:
                ep_start_removal, logs_ep_init = calculate_initial_ep(
                    emp_list_temp, st.session_state.l0, nH_mat_temp, nL_mat_temp
                )
                add_log(logs_ep_init)
                if ep_start_removal is None:
                    st.error("Failed to calculate nominal structure from QWOT for removal.")
                    return
            st.session_state.current_ep = ep_start_removal.copy() 
        except Exception as e_nom:
            st.error(f"Error calculating nominal structure for removal: {e_nom}")
            return
    if ep_start_removal is None:
        st.error("Could not determine a valid starting structure for removal.")
        return
    if len(ep_start_removal) <= 2:
        st.error("Structure too small (<= 2 layers) for removal/merge.")
        return
    with st.spinner("Removing thin layer + Re-optimizing..."):
        try:
            st.session_state.ep_history.append(ep_start_removal.copy())
            active_targets = validate_targets()
            if active_targets is None or not active_targets:
                st.session_state.ep_history.pop()
                st.error("Removal aborted: invalid or missing targets for re-optimization.")
                return
            l_min_opt, l_max_opt = get_lambda_range_from_targets(active_targets)
            if l_min_opt is None:
                st.session_state.ep_history.pop()
                st.error("Removal aborted: invalid lambda range for re-optimization.")
                return

            nH_r = st.session_state.get('nH_r', 2.35)
            nH_i = st.session_state.get('nH_i', 0.0)
            nL_r = st.session_state.get('nL_r', 1.46)
            nL_i = st.session_state.get('nL_i', 0.0)
            nSub_r = st.session_state.get('nSub_r', 1.52)
            nH_mat = complex(nH_r, nH_i)
            nL_mat = complex(nL_r, nL_i)
            nSub_mat = complex(nSub_r, 0.0)
            
            validated_inputs = {
                'l0': st.session_state.l0, 'l_step': st.session_state.l_step,
                'emp_str': st.session_state.current_qwot,
                'auto_thin_threshold': st.session_state.auto_thin_threshold,
                'l_range_deb': l_min_opt, 'l_range_fin': l_max_opt,
                'nH_material': nH_mat,
                'nL_material': nL_mat,
                'nSub_material': nSub_mat,
            }

            threshold = validated_inputs['auto_thin_threshold']
            ep_after_removal, structure_changed, removal_logs = _perform_layer_merge_or_removal_only(
                ep_start_removal, MIN_THICKNESS_PHYS_NM,
                log_prefix="  [Remove] ",
                threshold_for_removal=None 
            )
            add_log(removal_logs)
            if structure_changed and ep_after_removal is not None:
                add_log("Re-optimizing after removal...")
                final_ep, success, final_cost, optim_logs, msg = \
                    _run_core_optimization(ep_after_removal, validated_inputs, active_targets,
                                             MIN_THICKNESS_PHYS_NM, log_prefix="  [ReOpt Thin] ")
                add_log(optim_logs)
                if success and final_ep is not None:
                    st.session_state.optimized_ep = final_ep.copy()
                    st.session_state.current_ep = final_ep.copy() 
                    st.session_state.is_optimized_state = True 
                    st.session_state.last_rmse = np.sqrt(final_cost) if final_cost != np.inf else np.inf
                    qwots_opt, logs_qwot = calculate_qwot_from_ep(final_ep, validated_inputs['l0'], nH_mat, nL_mat)
                    add_log(logs_qwot)
                    if qwots_opt is not None and not np.any(np.isnan(qwots_opt)):
                        st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_opt])
                    else: st.session_state.optimized_qwot_str = "QWOT N/A"
                    st.success(f"Removal + Re-optimization finished ({msg}). Final RMSE: {st.session_state.last_rmse:.4e}")
                    st.session_state.needs_rerun_calc = True
                    st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': f"Optimized (Post-Remove)"}
                else:
                    st.warning(f"Layer removed, but re-optimization failed ({msg}). State is AFTER removal but BEFORE failed re-opt attempt.")
                    st.session_state.optimized_ep = ep_after_removal.copy()
                    st.session_state.current_ep = ep_after_removal.copy()
                    st.session_state.is_optimized_state = True 
                    try:
                        l_min_opt_disp, l_max_opt_disp = validated_inputs['l_range_deb'], validated_inputs['l_range_fin']
                        l_step_optim_disp = validated_inputs['l_step']
                        num_pts_optim_disp = min(max(2, int(np.round((l_max_opt_disp - l_min_opt_disp) / l_step_optim_disp)) + 1), 100)
                        l_vec_optim_np_disp = np.geomspace(l_min_opt_disp, l_max_opt_disp, num_pts_optim_disp)
                        l_vec_optim_np_disp = l_vec_optim_np_disp[(l_vec_optim_np_disp > 0) & np.isfinite(l_vec_optim_np_disp)]
                        if l_vec_optim_np_disp.size > 0:
                            results_fail_grid, logs_fail = calculate_T_from_ep_jax(ep_after_removal, nH_mat, nL_mat, nSub_mat, l_vec_optim_np_disp)
                            add_log(logs_fail)
                            if results_fail_grid:
                                rmse_fail, _ = calculate_final_rmse(results_fail_grid, active_targets)
                                st.session_state.last_rmse = rmse_fail
                            else: st.session_state.last_rmse = None
                        qwots_fail, logs_qwot_fail = calculate_qwot_from_ep(ep_after_removal, validated_inputs['l0'], nH_mat, nL_mat)
                        add_log(logs_qwot_fail)
                        if qwots_fail is not None and not np.any(np.isnan(qwots_fail)):
                            st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_fail])
                        else: st.session_state.optimized_qwot_str = "QWOT N/A (ReOpt Fail)"
                    except Exception as e_recalc:
                        st.session_state.last_rmse = None
                        st.session_state.optimized_qwot_str = "Recalc Error"
                        add_log(f"ERROR recalculating state after failed re-opt: {e_recalc}")
                    st.session_state.needs_rerun_calc = True
                    st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': "Optimized (Post-Remove, Re-Opt Fail)"}
            else:
                st.info("No layer was removed (criteria not met).")
                try:
                    st.session_state.ep_history.pop() 
                except IndexError: pass
        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Error during Thin Layer Removal: {e}")
            add_log(f"ERROR: {e}")
            try: st.session_state.ep_history.pop()
            except IndexError: pass
        except Exception as e_fatal:
            st.error(f"Unexpected error during Thin Layer Removal: {e_fatal}")
            add_log(f"FATAL ERROR: {e_fatal}")
            try: st.session_state.ep_history.pop()
            except IndexError: pass
        finally:
            pass

def run_monte_carlo_wrapper(container):
    with container:
        if 'last_calc_results' not in st.session_state or not st.session_state.last_calc_results:
            st.warning("Veuillez d'abord calculer une structure de base dans l'onglet 'Résultats'.")
            return

        with st.spinner("Lancement de la simulation Monte-Carlo..."):
            try:
                # Get base data from the last calculation
                base_results = st.session_state.last_calc_results
                ep_base = base_results.get('ep_used')
                if ep_base is None or ep_base.size == 0:
                    st.error("Aucune structure de base à simuler. Veuillez d'abord évaluer une conception.")
                    return

                nH_mat = base_results['nH_used']
                nL_mat = base_results['nL_used']
                nSub_mat = base_results['nSub_used']
                l_vec = base_results['res_fine']['l']
                std_dev = st.session_state.monte_carlo_std_dev
                num_draws = 100
                active_targets = validate_targets()
                
                # Pre-calculate n,k arrays
                nH_arr, _ = _get_nk_array_for_lambda_vec(nH_mat, l_vec)
                nL_arr, _ = _get_nk_array_for_lambda_vec(nL_mat, l_vec)
                nSub_arr, _ = _get_nk_array_for_lambda_vec(nSub_mat, l_vec)

                @jax.jit
                def get_spectrum_for_one_ep(ep_vector, nH, nL, nSub, lambdas):
                    ep_vector_jnp = jnp.asarray(ep_vector)
                    num_layers = len(ep_vector_jnp)
                    indices_alternating = jnp.where(jnp.arange(num_layers)[:, None] % 2 == 0, nH, nL)
                    indices_alternating_T = indices_alternating.T
                    
                    Ts_arr_raw = vmap(calculate_single_wavelength_T_core, in_axes=(0, None, 0, 0))(
                        lambdas, ep_vector_jnp, indices_alternating_T, nSub
                    )
                    Ts_arr = jnp.nan_to_num(Ts_arr_raw, nan=0.0)
                    return jnp.clip(Ts_arr, 0.0, 1.0)

                vmap_calculate_T = jax.vmap(get_spectrum_for_one_ep, in_axes=(0, None, None, None, None))

                # Generate all random structures at once
                noise = np.random.normal(0, std_dev, (num_draws, len(ep_base)))
                perturbed_eps = ep_base + noise
                perturbed_eps = np.maximum(perturbed_eps, MIN_THICKNESS_PHYS_NM) # Ensure no negative thickness

                # Run all calculations in a single batch call
                all_ts_results = np.array(vmap_calculate_T(jnp.array(perturbed_eps), nH_arr, nL_arr, nSub_arr, l_vec))

                # Calculate RMSE for each run
                all_rmses = []
                if active_targets:
                    for i in range(num_draws):
                        res_temp = {'l': l_vec, 'Ts': all_ts_results[i]}
                        rmse, _ = calculate_final_rmse(res_temp, active_targets)
                        if rmse is not None:
                            all_rmses.append(rmse)
                
                plausible_rmse = np.percentile(all_rmses, 80) if all_rmses else None

                # Calculate confidence interval
                lower_bound = np.percentile(all_ts_results, 10, axis=0)
                upper_bound = np.percentile(all_ts_results, 90, axis=0)
                
                st.session_state.monte_carlo_results = {
                    'l_vec': l_vec,
                    'all_ts_results': all_ts_results,
                    'base_ts': base_results['res_fine']['Ts'],
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'std_dev': std_dev,
                    'plausible_rmse': plausible_rmse
                }
                add_log("Monte Carlo simulation finished.")

            except Exception as e:
                st.error(f"Une erreur est survenue pendant la simulation Monte-Carlo: {e}")
                add_log(f"FATAL ERROR during Monte Carlo: {e}")
                traceback.print_exc()

def run_tolerance_analysis_wrapper(container):
    with container:
        if 'last_calc_results' not in st.session_state or not st.session_state.last_calc_results:
            st.warning("Veuillez d'abord calculer une structure de base dans l'onglet 'Résultats'.")
            return

        with st.spinner("Lancement de l'analyse de tolérance..."):
            try:
                base_results = st.session_state.last_calc_results
                ep_base = base_results.get('ep_used')
                if ep_base is None or ep_base.size == 0:
                    st.error("Aucune structure de base à analyser.")
                    return
                
                nH_mat = base_results['nH_used']
                nL_mat = base_results['nL_used']
                nSub_mat = base_results['nSub_used']
                active_targets = validate_targets()
                if not active_targets:
                    st.error("L'analyse de tolérance nécessite des cibles actives pour calculer le RMSE.")
                    return

                l_min_plot, l_max_plot = get_lambda_range_from_targets(active_targets)
                l_vec = np.geomspace(l_min_plot, l_max_plot, 100)
                
                std_devs_abs = np.linspace(0, 5, 15)
                std_devs_rel = np.linspace(0, 10, 15)
                num_draws = 100
                
                plausible_rmses_abs = []
                plausible_rmses_rel = []

                # JAX-compatible calculation function
                @jax.jit
                def get_spectrum_for_one_ep(ep_vector, nH, nL, nSub, lambdas):
                    ep_vector_jnp = jnp.asarray(ep_vector)
                    num_layers = len(ep_vector_jnp)
                    indices_alternating = jnp.where(jnp.arange(num_layers)[:, None] % 2 == 0, nH, nL)
                    indices_alternating_T = indices_alternating.T
                    Ts_arr_raw = vmap(calculate_single_wavelength_T_core, in_axes=(0, None, 0, 0))(
                        lambdas, ep_vector_jnp, indices_alternating_T, nSub
                    )
                    Ts_arr = jnp.nan_to_num(Ts_arr_raw, nan=0.0)
                    return jnp.clip(Ts_arr, 0.0, 1.0)
                
                vmap_calculate_T = jax.vmap(get_spectrum_for_one_ep, in_axes=(0, None, None, None, None))
                
                nH_arr, _ = _get_nk_array_for_lambda_vec(nH_mat, l_vec)
                nL_arr, _ = _get_nk_array_for_lambda_vec(nL_mat, l_vec)
                nSub_arr, _ = _get_nk_array_for_lambda_vec(nSub_mat, l_vec)

                # Absolute error analysis
                for std_dev in std_devs_abs:
                    noise = np.random.normal(0, std_dev, (num_draws, len(ep_base)))
                    perturbed_eps = ep_base + noise
                    perturbed_eps = np.maximum(perturbed_eps, MIN_THICKNESS_PHYS_NM)
                    all_ts_results = np.array(vmap_calculate_T(jnp.array(perturbed_eps), nH_arr, nL_arr, nSub_arr, l_vec))
                    
                    all_rmses = [calculate_final_rmse({'l': l_vec, 'Ts': ts}, active_targets)[0] for ts in all_ts_results]
                    all_rmses = [r for r in all_rmses if r is not None]
                    plausible_rmses_abs.append(np.percentile(all_rmses, 80) if all_rmses else 0)

                # Relative error analysis
                for std_dev_percent in std_devs_rel:
                    std_dev_values = ep_base * (std_dev_percent / 100.0)
                    noise = np.random.normal(0, 1, (num_draws, len(ep_base))) * std_dev_values
                    perturbed_eps = ep_base + noise
                    perturbed_eps = np.maximum(perturbed_eps, MIN_THICKNESS_PHYS_NM)
                    all_ts_results = np.array(vmap_calculate_T(jnp.array(perturbed_eps), nH_arr, nL_arr, nSub_arr, l_vec))

                    all_rmses = [calculate_final_rmse({'l': l_vec, 'Ts': ts}, active_targets)[0] for ts in all_ts_results]
                    all_rmses = [r for r in all_rmses if r is not None]
                    plausible_rmses_rel.append(np.percentile(all_rmses, 80) if all_rmses else 0)

                # Store results for plotting
                st.session_state.tolerance_analysis_results = {
                    'std_devs_abs': std_devs_abs,
                    'plausible_rmses_abs': plausible_rmses_abs,
                    'std_devs_rel': std_devs_rel,
                    'plausible_rmses_rel': plausible_rmses_rel,
                }
                add_log("Tolerance analysis finished.")

            except Exception as e:
                st.error(f"Une erreur est survenue pendant l'analyse de tolérance: {e}")
                add_log(f"FATAL ERROR during tolerance analysis: {e}")
                traceback.print_exc()


st.set_page_config(layout="wide", page_title="formation_CMO_2025")
if 'init_done' not in st.session_state:
    st.session_state.log_messages = []
    st.session_state.current_ep = None
    st.session_state.current_qwot = "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
    st.session_state.optimized_ep = None
    st.session_state.is_optimized_state = False
    st.session_state.optimized_qwot_str = ""
    st.session_state.material_sequence = None
    st.session_state.ep_history = deque(maxlen=5)
    st.session_state.last_rmse = None
    st.session_state.needs_rerun_calc = False
    st.session_state.rerun_calc_params = {}
    st.session_state.action = None
    st.session_state.monte_carlo_results = None
    st.session_state.tolerance_analysis_results = None

    st.session_state.l0 = 500.0
    st.session_state.l_step = 10.0
    st.session_state.auto_thin_threshold = 1.0
    st.session_state.auto_scale_y = False
    st.session_state.monte_carlo_std_dev = 2.0
    
    st.session_state.targets = [
        {'enabled': True, 'min': 400.0, 'max': 500.0, 'target_min': 1.0, 'target_max': 1.0},
        {'enabled': True, 'min': 500.0, 'max': 600.0, 'target_min': 1.0, 'target_max': 0.2},
        {'enabled': True, 'min': 600.0, 'max': 700.0, 'target_min': 0.2, 'target_max': 0.2},
        {'enabled': False, 'min': 700.0, 'max': 800.0, 'target_min': 0.0, 'target_max': 0.0},
        {'enabled': False, 'min': 800.0, 'max': 900.0, 'target_min': 0.0, 'target_max': 0.0},
    ]
    st.session_state.nH_r = 2.35
    st.session_state.nH_i = 0.0
    st.session_state.nL_r = 1.46
    st.session_state.nL_i = 0.0
    st.session_state.nSub_r = 1.52
    st.session_state.init_done = True
    st.session_state.needs_rerun_calc = True
    st.session_state.rerun_calc_params = {'is_optimized_run': False, 'method_name': "Initial Load"}
    add_log("Application initialized.")

def trigger_nominal_recalc():
    if not st.session_state.get('calculating', False):
        st.session_state.needs_rerun_calc = True
        st.session_state.rerun_calc_params = {
            'is_optimized_run': False,
            'method_name': "Nominal (Param Update)",
            'force_ep': None
        }

st.title("formation_CMO_2025")

st.divider()
main_layout = st.columns([1, 3]) 

with main_layout[0]: 
    st.subheader("Materials (Constant Index)")
    st.markdown("**H Material**")
    hc1, hc2 = st.columns(2)
    st.session_state.nH_r = hc1.number_input("n' H", value=st.session_state.nH_r, format="%.4f", key="nH_r_const", on_change=trigger_nominal_recalc)
    st.session_state.nH_i = hc2.number_input("k H", value=st.session_state.nH_i, min_value=0.0, format="%.4f", key="nH_i_const", on_change=trigger_nominal_recalc)
    
    st.markdown("**L Material**")
    lc1, lc2 = st.columns(2)
    st.session_state.nL_r = lc1.number_input("n' L", value=st.session_state.nL_r, format="%.4f", key="nL_r_const", on_change=trigger_nominal_recalc)
    st.session_state.nL_i = lc2.number_input("k L", value=st.session_state.nL_i, min_value=0.0, format="%.4f", key="nL_i_const", on_change=trigger_nominal_recalc)

    st.markdown("**Substrate Material**")
    st.session_state.nSub_r = st.number_input("n' Substrate", value=st.session_state.nSub_r, format="%.4f", key="nSub_const", on_change=trigger_nominal_recalc)

    st.subheader("Nominal Structure")
    st.session_state.current_qwot = st.text_area(
        "QWOT Multipliers (comma-separated)", value=st.session_state.current_qwot, key="qwot_input",
        on_change=clear_optimized_state, height=100
    )
    num_layers_from_qwot = len([q for q in st.session_state.current_qwot.split(',') if q.strip()])
    qwot_cols = st.columns([3,2])
    with qwot_cols[0]:
        st.session_state.l0 = st.number_input("Reference Wavelength λ₀ (nm) for QWOT", value=st.session_state.l0, min_value=1.0, format="%.2f", key="l0_input", on_change=trigger_nominal_recalc)
    with qwot_cols[1]:
        init_layers_num = st.number_input("Number of Layers to Generate", min_value=0, value=num_layers_from_qwot, step=1, key="init_layers_gen_num")
        if st.button("Gen 1s QWOT", key="gen_qwot_btn_main", use_container_width=True, help="Generate a QWOT string with the specified number of layers, all set to 1.0"):
            if init_layers_num > 0:
                new_qwot = ",".join(['1'] * init_layers_num)
                if new_qwot != st.session_state.current_qwot:
                    st.session_state.current_qwot = new_qwot
                    clear_optimized_state()
                    st.session_state.needs_rerun_calc = True
                    st.session_state.rerun_calc_params = {'is_optimized_run': False, 'method_name': "Nominal (Generated 1s)"}
                    st.rerun()
            elif st.session_state.current_qwot != "":
                st.session_state.current_qwot = ""
                clear_optimized_state()
                st.session_state.needs_rerun_calc = True
                st.session_state.rerun_calc_params = {'is_optimized_run': False, 'method_name': "Nominal (QWOT Cleared)"}
                st.rerun()
    st.caption(f"Current Nominal Layers: {num_layers_from_qwot}")

    # === BLOC DE CODE AJOUTÉ ===
    if 'last_calc_results' in st.session_state and st.session_state.last_calc_results:
        ep_display = st.session_state.last_calc_results.get('ep_used')

        if ep_display is not None and ep_display.size > 0:
            st.markdown("**Épaisseurs réelles (nm)**", unsafe_allow_html=True)
            formatted_thicknesses = [f"<span style='color:blue;'>{i + 1}.</span> {t:.1f}" for i, t in enumerate(ep_display)]
            
            # Divise la liste en 3 parties pour l'affichage sur 3 lignes
            chunks = np.array_split(np.array(formatted_thicknesses), 3)
            
            # Affiche chaque partie sur une ligne séparée
            for chunk in chunks:
                if chunk.size > 0:
                    st.code(", ".join(chunk), language='text')
    # === FIN DU BLOC AJOUTÉ ===

    st.subheader("Targets (T) & Calculation Parameters")
    st.session_state.l_step = st.number_input("λ Step for MSE Grid (nm)", value=st.session_state.l_step, min_value=0.1, format="%.2f", key="l_step_input_main", on_change=trigger_nominal_recalc, help="Wavelength step for optimization grid points (max 100 points). Plotting uses a finer grid.")
    st.session_state.auto_thin_threshold = st.number_input("Auto Thin Layer Removal Threshold (nm)", value=st.session_state.auto_thin_threshold, min_value=MIN_THICKNESS_PHYS_NM, format="%.3f", key="auto_thin_input_main", help="In Auto mode, layers thinner than this may be removed.")
    st.session_state.auto_scale_y = st.checkbox("Echelle Y automatique", value=st.session_state.auto_scale_y, key="auto_scale_y_cb", help="Ajuster automatiquement l'axe Y du graphe de transmittance.")

    hdr_cols = st.columns([0.5, 1, 1, 1, 1])
    hdrs = ["On", "λmin", "λmax", "Tmin", "Tmax"]
    for c, h in zip(hdr_cols, hdrs): c.caption(h)
    for i in range(len(st.session_state.targets)):
        target = st.session_state.targets[i]
        cols = st.columns([0.5, 1, 1, 1, 1])
        current_enabled = target.get('enabled', False)
        new_enabled = cols[0].checkbox("", value=current_enabled, key=f"target_enable_{i}_main", label_visibility="collapsed", on_change=trigger_nominal_recalc)
        st.session_state.targets[i]['enabled'] = new_enabled
        st.session_state.targets[i]['min'] = cols[1].number_input(f"λmin Target {i+1}", value=target.get('min', 0.0), format="%.1f", step=10.0, key=f"target_min_{i}_main", label_visibility="collapsed", on_change=trigger_nominal_recalc)
        st.session_state.targets[i]['max'] = cols[2].number_input(f"λmax Target {i+1}", value=target.get('max', 0.0), format="%.1f", step=10.0, key=f"target_max_{i}_main", label_visibility="collapsed", on_change=trigger_nominal_recalc)
        st.session_state.targets[i]['target_min'] = cols[3].number_input(f"Tmin Target {i+1}", value=target.get('target_min', 0.0), min_value=0.0, max_value=1.0, format="%.3f", step=0.01, key=f"target_tmin_{i}_main", label_visibility="collapsed", on_change=trigger_nominal_recalc)
        st.session_state.targets[i]['target_max'] = cols[4].number_input(f"Tmax Target {i+1}", value=target.get('target_max', 0.0), min_value=0.0, max_value=1.0, format="%.3f", step=0.01, key=f"target_tmax_{i}_main", label_visibility="collapsed", on_change=trigger_nominal_recalc)

with main_layout[1]: 
    results_tab, random_draws_tab, tolerance_tab, logs_tab = st.tabs(["**Résultats**", "**Tirages Aléatoires**", "**Analyse de Tolérance**", "**Logs**"])
    with results_tab:
        st.subheader("Actions")
        menu_cols = st.columns(6)
        with menu_cols[0]:
            if st.button("📊 Eval Nom.", key="eval_nom_top", help="Evaluate Nominal Structure", use_container_width=True):
                st.session_state.action = 'eval_nom'
        with menu_cols[1]:
            if st.button("✨ Opt Local", key="optim_local_top", help="Local Optimization", use_container_width=True):
                st.session_state.action = 'opt_local'
        with menu_cols[2]:
            if st.button("💉 Needle", key="optim_needle_top", help="Run a single Needle cycle (insert one layer and re-optimize)", use_container_width=True):
                st.session_state.action = 'needle'
        with menu_cols[3]:
            current_structure_for_check = st.session_state.get('current_ep')
            can_remove_structurally = current_structure_for_check is not None and len(current_structure_for_check) > 2
            if st.button("🗑️ Thin+ReOpt", key="remove_thin_top", help="Remove Thin Layer + Re-optimize", disabled=not can_remove_structurally, use_container_width=True):
                st.session_state.action = 'remove_thin'
        with menu_cols[4]:
            can_optimize_top = st.session_state.get('is_optimized_state', False) and st.session_state.get('optimized_ep') is not None
            if st.button("💾 Opt->Nom", key="set_optim_as_nom_top", help="Set Optimized as Nominal", disabled=not can_optimize_top, use_container_width=True):
                set_optimized_as_nominal_wrapper()
                st.rerun()
        with menu_cols[5]:
            can_undo_top = bool(st.session_state.get('ep_history'))
            if st.button(f"↩️ Undo ({len(st.session_state.get('ep_history', deque()))})", key="undo_remove_top", help="Undo Last Removal", disabled=not can_undo_top, use_container_width=True):
                undo_remove_wrapper()
                st.rerun()
        
        st.divider()
        
        if st.session_state.get('last_calc_results'):
            st.subheader("Final Results")
            results_data = st.session_state.last_calc_results
            state_desc = "Optimized" if st.session_state.is_optimized_state else "Nominal"
            ep_display = results_data.get('ep_used')
            num_layers_display = len(ep_display) if ep_display is not None else 0
            
            res_info_cols = st.columns(3)
            res_info_cols[0].caption(f"State: {state_desc} ({num_layers_display} layers)")
            res_info_cols[1].metric(label="RMSE", value=f"{st.session_state.last_rmse:.4e}" if st.session_state.last_rmse is not None else "N/A")
            min_thick_str = "N/A"
            if ep_display is not None and ep_display.size > 0:
                valid_thick = ep_display[ep_display >= MIN_THICKNESS_PHYS_NM - 1e-9]
                if valid_thick.size > 0:
                    min_thick_str = f"{np.min(valid_thick):.3f} nm"
            res_info_cols[2].caption(f"Min Thick: {min_thick_str}")

            if st.session_state.is_optimized_state and st.session_state.get('optimized_qwot_str'):
                st.text_input("Optimized QWOT (at original λ₀)", value=st.session_state.optimized_qwot_str, disabled=True, key="opt_qwot_display_main_res")

            res_fine_plot = results_data.get('res_fine')
            active_targets_plot = validate_targets()
            rmse_plot = st.session_state.last_rmse
            method_name_plot = results_data.get('method_name', '')
            res_optim_grid_plot = results_data.get('res_optim_grid')

            if res_fine_plot and active_targets_plot is not None:
                fig_spec, ax_spec = plt.subplots(figsize=(12, 4)) 
                opt_method_str = f" ({method_name_plot})" if method_name_plot else ""
                window_title = f'Spectral Response {"Optimized" if st.session_state.is_optimized_state else "Nominal"}{opt_method_str}'
                fig_spec.suptitle(window_title, fontsize=12, weight='bold')
                line_ts = None
                try:
                    if res_fine_plot and 'l' in res_fine_plot and 'Ts' in res_fine_plot and res_fine_plot['l'] is not None and len(res_fine_plot['l']) > 0:
                        res_l_plot = np.asarray(res_fine_plot['l'])
                        res_ts_plot = np.asarray(res_fine_plot['Ts'])
                        line_ts, = ax_spec.plot(res_l_plot, res_ts_plot, label='Transmittance', linestyle='-', color='blue', linewidth=1.5)
                        plotted_target_label = False
                        if active_targets_plot:
                            for i, target in enumerate(active_targets_plot):
                                l_min, l_max = target['min'], target['max']
                                t_min, t_max_corr = target['target_min'], target['target_max']
                                x_coords, y_coords = [l_min, l_max], [t_min, t_max_corr]
                                label = 'Target(s)' if not plotted_target_label else "_nolegend_"
                                line_target, = ax_spec.plot(x_coords, y_coords, 'r--', linewidth=1.0, alpha=0.7, label=label, zorder=5)
                                marker_target = ax_spec.plot(x_coords, y_coords, marker='x', color='red', markersize=6, linestyle='none', label='_nolegend_', zorder=6)
                                plotted_target_label = True
                                if res_optim_grid_plot and 'l' in res_optim_grid_plot and res_optim_grid_plot['l'].size > 0:
                                    res_l_optim = np.asarray(res_optim_grid_plot['l'])
                                    indices_optim = np.where((res_l_optim >= l_min) & (res_l_optim <= l_max))[0]
                                    if indices_optim.size > 0:
                                        optim_lambdas = res_l_optim[indices_optim]
                                        if abs(l_max - l_min) < 1e-9: optim_target_t = np.full_like(optim_lambdas, t_min)
                                        else: slope = (t_max_corr - t_min) / (l_max - l_min); optim_target_t = t_min + slope * (optim_lambdas - l_min)
                                        ax_spec.plot(optim_lambdas, optim_target_t, marker='.', color='darkred', linestyle='none', markersize=4, alpha=0.5, label='_nolegend_', zorder=6)
                    ax_spec.set_xlabel("Wavelength (nm)")
                    ax_spec.set_ylabel('Transmittance')
                    ax_spec.grid(True, which='major', linestyle='-', linewidth='0.5', color='gray')
                    ax_spec.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgray')
                    ax_spec.minorticks_on()
                    if len(res_l_plot) > 0 : ax_spec.set_xlim(res_l_plot[0], res_l_plot[-1])
                    if not st.session_state.get('auto_scale_y', False):
                        ax_spec.set_ylim(-0.05, 1.05)
                    if plotted_target_label or (line_ts is not None): ax_spec.legend(fontsize=8)
                    if rmse_plot is not None and np.isfinite(rmse_plot): rmse_text = f"RMSE = {rmse_plot:.3e}"
                    else: rmse_text = "RMSE: N/A"
                    ax_spec.text(0.98, 0.98, rmse_text, transform=ax_spec.transAxes, ha='right', va='top', fontsize=9,
                                         bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))
                except Exception as e_spec:
                    ax_spec.text(0.5, 0.5, f"Error plotting spectrum:\n{e_spec}", ha='center', va='center', transform=ax_spec.transAxes, color='red')
                plt.tight_layout(rect=[0, 0, 1, 0.93])
                st.pyplot(fig_spec)
                plt.close(fig_spec)

            plot_col1, plot_col2 = st.columns(2)
            ep_plot = results_data.get('ep_used')
            l0_plot = results_data.get('l0_used')
            nH_plot = results_data.get('nH_used')
            nL_plot = results_data.get('nL_used')
            nSub_plot = results_data.get('nSub_used')
            if ep_plot is not None and l0_plot is not None and nH_plot is not None and nL_plot is not None and nSub_plot is not None:
                with plot_col1:
                    fig_idx, ax_idx = plt.subplots(figsize=(6, 4)) 
                    try:
                        nH_c_repr, _ = _get_nk_at_lambda(nH_plot, l0_plot)
                        nL_c_repr, _ = _get_nk_at_lambda(nL_plot, l0_plot)
                        nSub_c_repr, _ = _get_nk_at_lambda(nSub_plot, l0_plot)
                        if nH_c_repr is None or nL_c_repr is None or nSub_c_repr is None:
                            raise ValueError("Indices at l0 not found for profile plot.")
                        nH_r_repr, nL_r_repr, nSub_r_repr = nH_c_repr.real, nL_c_repr.real, nSub_c_repr.real
                        num_layers = len(ep_plot)
                        n_real_layers_repr = [nH_r_repr if i % 2 == 0 else nL_r_repr for i in range(num_layers)]
                        ep_cumulative = np.cumsum(ep_plot) if num_layers > 0 else np.array([0])
                        total_thickness = ep_cumulative[-1] if num_layers > 0 else 0
                        margin = max(50, 0.1 * total_thickness) if total_thickness > 0 else 50
                        x_coords_plot, y_coords_plot = [-margin], [nSub_r_repr]
                        if num_layers > 0:
                            x_coords_plot.append(0); y_coords_plot.append(nSub_r_repr)
                            for i in range(num_layers):
                                layer_start = ep_cumulative[i-1] if i > 0 else 0
                                layer_end = ep_cumulative[i]
                                x_coords_plot.extend([layer_start, layer_end]); y_coords_plot.extend([n_real_layers_repr[i], n_real_layers_repr[i]])
                            x_coords_plot.extend([ep_cumulative[-1], ep_cumulative[-1] + margin]); y_coords_plot.extend([1.0, 1.0])
                        else:
                            x_coords_plot.extend([0, 0, margin]); y_coords_plot.extend([nSub_r_repr, 1.0, 1.0])
                        ax_idx.plot(x_coords_plot, y_coords_plot, drawstyle='steps-post', color='purple')
                        ax_idx.set_xlabel('Profondeur (nm)'); ax_idx.set_ylabel("n'")
                        ax_idx.set_title(f"Profil d'indice (at λ={l0_plot:.0f}nm)", fontsize=10)
                        ax_idx.grid(True, linestyle=':')
                    except Exception as e_idx:
                        ax_idx.text(0.5, 0.5, f"Erreur de tracé:\n{e_idx}", ha='center', va='center')
                    plt.tight_layout()
                    st.pyplot(fig_idx)
                    plt.close(fig_idx)
                with plot_col2:
                    fig_stack, ax_stack = plt.subplots(figsize=(6, 4)) 
                    try:
                        num_layers = len(ep_plot)
                        if num_layers > 0:
                            layer_types = ["H" if i % 2 == 0 else "L" for i in range(num_layers)]
                            colors = ['lightblue' if i % 2 == 0 else 'lightcoral' for i in range(num_layers)]
                            bar_pos = np.arange(num_layers)
                            ax_stack.barh(bar_pos, ep_plot, align='center', color=colors, edgecolor='grey')
                            ax_stack.set_yticks(bar_pos); ax_stack.set_yticklabels([f"L{i+1} ({t})" for i, t in enumerate(layer_types)], fontsize=7)
                            ax_stack.invert_yaxis()
                        else:
                            ax_stack.text(0.5, 0.5, "Structure Vide", ha='center', va='center')
                        ax_stack.set_xlabel('Épaisseur (nm)')
                        ax_stack.set_title(f"Structure ({num_layers} layers)", fontsize=10)
                    except Exception as e_stack:
                        ax_stack.text(0.5, 0.5, f"Erreur de tracé:\n{e_stack}", ha='center', va='center')
                    plt.tight_layout()
                    st.pyplot(fig_stack)
                    plt.close(fig_stack)
    
    with random_draws_tab:
        st.subheader("Simulation de Monte-Carlo")
        st.session_state.monte_carlo_std_dev = st.number_input("Écart-type pour l'épaisseur (nm)", min_value=0.0, value=st.session_state.monte_carlo_std_dev, step=0.1, format="%.2f", key="mc_std_dev")
        if st.button("Lancer la simulation", key="run_mc"):
            st.session_state.action = 'monte_carlo'
            st.rerun()
        
        if 'monte_carlo_results' in st.session_state and st.session_state.monte_carlo_results:
            mc_data = st.session_state.monte_carlo_results
            plausible_rmse = mc_data.get('plausible_rmse')
            if plausible_rmse is not None:
                st.metric(label="RMSE Plausible (80% des cas)", value=f"{plausible_rmse:.4e}")
            
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(mc_data['l_vec'], mc_data['base_ts'], color='red', linewidth=2, label='Réponse Idéale', zorder=3)
            ax.scatter(np.tile(mc_data['l_vec'], 100), mc_data['all_ts_results'].flatten(), color='lightgray', alpha=0.2, s=2, zorder=1)
            ax.fill_between(mc_data['l_vec'], mc_data['lower_bound'], mc_data['upper_bound'], color='blue', alpha=0.3, label='Intervalle de confiance à 80%', zorder=2)
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Transmittance")
            ax.set_title(f"Simulation de Monte-Carlo (100 tirages, Écart-type: {mc_data['std_dev']} nm)")
            ax.legend()
            ax.grid(True, linestyle=':')
            if not st.session_state.get('auto_scale_y', False):
                ax.set_ylim(-0.05, 1.05)
            st.pyplot(fig)
            plt.close(fig)
    
    with tolerance_tab:
        st.subheader("Analyse de Tolérance")
        if st.button("Lancer l'analyse de tolérance", key="run_tolerance"):
            st.session_state.action = 'tolerance_analysis'
            st.rerun()
        
        if 'tolerance_analysis_results' in st.session_state and st.session_state.tolerance_analysis_results:
            tol_data = st.session_state.tolerance_analysis_results
            fig, ax1 = plt.subplots(figsize=(12, 5))
            
            color1 = 'tab:blue'
            ax1.set_xlabel('Écart-type absolu (nm)', color=color1)
            ax1.set_ylabel('RMSE Plausible (80%)', color=color1)
            ax1.plot(tol_data['std_devs_abs'], tol_data['plausible_rmses_abs'], color=color1, marker='o', linewidth=2.5, label='Erreur Absolue')
            ax1.tick_params(axis='x', labelcolor=color1)
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(True, linestyle=':', color=color1, alpha=0.5)

            ax2 = ax1.twiny()
            color2 = 'tab:orange'
            ax2.set_xlabel('Écart-type relatif (%)', color=color2)
            ax2.plot(tol_data['std_devs_rel'], tol_data['plausible_rmses_rel'], color=color2, marker='x', linewidth=2.5, label='Erreur Relative')
            ax2.tick_params(axis='x', labelcolor=color2)
            
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


    with logs_tab:
        st.subheader("Logs")
        log_text = "\n".join(st.session_state.get('log_messages', ['No logs yet.']))
        st.code(log_text, language='text')

# --- Action Handling Logic ---
action_to_run = st.session_state.get('action')
if action_to_run:
    st.session_state.action = None # Reset flag
    if action_to_run == 'eval_nom':
        st.session_state.needs_rerun_calc = True
        st.session_state.rerun_calc_params = {'is_optimized_run': False, 'method_name': "Nominal (Evaluated)"}
    elif action_to_run == 'opt_local':
        run_local_optimization_wrapper()
        st.session_state.needs_rerun_calc = True
    elif action_to_run == 'needle':
        run_needle_wrapper()
    elif action_to_run == 'remove_thin':
        run_remove_thin_wrapper()
    elif action_to_run == 'monte_carlo':
        run_monte_carlo_wrapper(random_draws_tab)
    elif action_to_run == 'tolerance_analysis':
        run_tolerance_analysis_wrapper(tolerance_tab)
    st.rerun()

if st.session_state.get('needs_rerun_calc', False):
    params = st.session_state.rerun_calc_params
    force_ep_val = params.get('force_ep')
    st.session_state.needs_rerun_calc = False
    st.session_state.rerun_calc_params = {}
    st.session_state.calculating = True
    run_calculation_wrapper(
        is_optimized_run=params.get('is_optimized_run', False),
        method_name=params.get('method_name', 'Auto Recalc'),
        force_ep=force_ep_val
    )
    st.session_state.calculating = False
    st.rerun()
