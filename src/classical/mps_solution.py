# src/classical/mps_solution.py
# MPS/TEBD backend for 1D XX/XXZ chains (OBC) using TeNPy.
#   - Matches exact (Pauli) convention via rescale: Jxx=4J, Jz=4Δ, hz=2h
#   - Closed system TEBD + (optional) open-system quantum trajectories (MCWF)
#   - Dense readout path at small N: pZ/pX/pY_*, exp_* (Pauli strings), Loschmidt

import argparse, json, os, csv, math, itertools
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Literal, Dict, Any

import numpy as np

try:
    from tenpy.models.xxz_chain import XXZChain
    from tenpy.networks.mps import MPS
    from tenpy.algorithms import tebd
    from tenpy.algorithms.exact_diag import ExactDiag
except Exception as e:
    raise SystemExit("Please install TeNPy first: pip install tenpy-physics") from e

# -------------------------
# Config
# -------------------------
@dataclass
class NoiseConfig:
    dephasing_rate: float = 0.0         # γφ
    relaxation_rate: float = 0.0        # γ↓  (|1>→|0>)
    thermal_pop: Optional[float] = 0.0  # γ↑  (|0>→|1>), # excited-state pop for thermal (None => 0)

@dataclass
class DisorderConfig:
    kind: Literal["none", "uniform", "normal"] = "none"
    strength: float = 0.0               # half-width (uniform) or std (normal)
    seed: Optional[int] = None

@dataclass
class PotentialConfig:
    type: Literal["none", "linear"] = "none"
    slope: float = 0.0    # F in F * i * Z_i  (i starts at 0)

@dataclass
class ModelConfig:
    N: int = 6
    J: float = 1.0                      # XX coupling
    Delta: float = 0.0                  # ZZ coupling (XXZ)
    boundary: Literal["OBC", "PBC"] = "OBC"
    disorder: DisorderConfig = field(default_factory=DisorderConfig)
    potential: PotentialConfig = field(default_factory=PotentialConfig)

@dataclass
class InitStateConfig:
    kind: Literal["neel", "domain_wall", "product_random_z", "basis_string"] = "neel"
    basis_string: Optional[str] = None

@dataclass
class TimeConfig:
    t_max: float = 8.0
    steps: int = 200

@dataclass
class TrajectoriesConfig:
    n_traj: int = 1
    seed: Optional[int] = None

@dataclass
class MPSConfig:
    method: Literal["TEBD2", "TDVP"] = "TEBD2"
    bond_dim_max: int = 256
    svd_trunc_tol: float = 1e-9
    conserve: Literal["Sz", "None"] = "Sz"
    two_site: bool = True
    trajectories: TrajectoriesConfig = field(default_factory=TrajectoriesConfig)

@dataclass
class RunConfig:
    backend: Literal["mps"] = "mps"
    model: ModelConfig = field(default_factory=ModelConfig)
    init: InitStateConfig = field(default_factory=InitStateConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)

    observables: List[str] = field(default_factory=lambda: ["energy", "magnetization_z", "sz_sites"])
    random_seed: Optional[int] = None
    pbc_wrap: bool = False

    measure_pauli_strings: List[str] = field(default_factory=list)
    entanglement_cut: Optional[int] = None
    store_probabilities: bool = False
    store_prob_bases: List[str] = field(default_factory=lambda: ["Z"])
    store_loschmidt: bool = True
    dense_readout: bool = True
    dense_readout_N_max: int = 12

    outdir: str = "experiments/results_mps"
    run_id: Optional[str] = None

    mps: MPSConfig = field(default_factory=MPSConfig)

# -------------------------
# Utilities
# -------------------------
_SIGMA_X = np.array([[0, 1],[1, 0]], dtype=complex)
_SIGMA_Y = np.array([[0, -1j],[1j, 0]], dtype=complex)
_SIGMA_Z = np.array([[1, 0],[0, -1]], dtype=complex)
_I2      = np.eye(2, dtype=complex)
_H       = np.array([[1, 1],[1, -1]], dtype=complex) / np.sqrt(2)
_SDG     = np.array([[1, 0],[0, -1j]], dtype=complex) 

def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def _rng(seed: Optional[int]):
    return np.random.default_rng(seed) if seed is not None else np.random.default_rng()

def _build_hz_array(cfg: ModelConfig) -> np.ndarray:
    N = cfg.N
    rng = _rng(cfg.disorder.seed)
    if cfg.disorder.kind == "none":
        h = np.zeros(N)
    elif cfg.disorder.kind == "uniform":
        w = float(cfg.disorder.strength)
        h = rng.uniform(-w, w, size=N)
    elif cfg.disorder.kind == "normal":
        h = rng.normal(0.0, float(cfg.disorder.strength), size=N)
    else:
        raise ValueError("Unknown disorder.kind")

    if cfg.potential.type == "linear":
        slope = float(cfg.potential.slope)
        h = h + slope * np.arange(N)
    return h

def _basis_string_from_init(init: InitStateConfig, N: int, seed: Optional[int]) -> str:
    """
    |0> := |↑z>, |1> := |↓z>.
    """
    if init.kind == "neel":
        return "".join("10"[i % 2] for i in range(N)) # "1010..."
    elif init.kind == "domain_wall":
        return "1"*(N//2) + "0"*(N - N//2)
    elif init.kind == "product_random_z":
        rng = _rng(seed)
        return "".join("0" if r < 0.5 else "1" for r in rng.random(N))
    elif init.kind == "basis_string":
        if not init.basis_string or len(init.basis_string) != N or set(init.basis_string) - set("01"):
            raise ValueError("init.basis_string must be a length-N string of 0/1")
        return init.basis_string
    else:
        raise ValueError("Unknown initial state kind")

def _product_state_list(basis_string: str) -> List[str]:
    # '0' -> 'up' (Z=+1), '1' -> 'down' (Z=-1)
    return ['up' if ch == '0' else 'down' for ch in basis_string]

def _build_model_and_state(cfg: RunConfig):
    N = cfg.model.N
    if cfg.model.boundary != "OBC":
        raise ValueError("XXZChain backend currently supports OBC only (bc_MPS='finite').")

    # Rescale to match Pauli convention of exact backend
    hz_exact = _build_hz_array(cfg.model)
    hz = 2.0 * hz_exact

    # Add non-Hermitian effective field from relaxation/excitation jumps: +i*(γ↓−γ↑)/2 * Sz
    gam_d = float(cfg.noise.relaxation_rate or 0.0)
    gam_u = float(cfg.noise.thermal_pop or 0.0)
    if np.any(hz) or (gam_d != 0.0 or gam_u != 0.0):
        hz = hz.astype(complex)
    if (gam_d != 0.0 or gam_u != 0.0):
        hz = hz + 1j * 0.5 * (gam_d - gam_u)

    model_params = dict(
        L=N,
        Jxx=4.0 * float(cfg.model.J),
        Jz =4.0 * float(cfg.model.Delta),
        hz=hz,
        bc_MPS='finite',
        conserve=cfg.mps.conserve,
        )
    M = XXZChain(model_params)

    bstr = _basis_string_from_init(cfg.init, N, cfg.random_seed)
    prod = _product_state_list(bstr)
    sites = M.lat.mps_sites()
    psi = MPS.from_product_state(sites, prod, bc='finite')
    psi.canonical_form()
    return M, psi, bstr

def _tebd_engine(M, psi, cfg: RunConfig):
    dt = float(cfg.time.t_max) / int(cfg.time.steps)
    tebd_params = {
        "order": 2 if cfg.mps.method.upper().startswith("TEBD2") else 2,
        "dt": dt,
        "N_steps": 1,
        "trunc_params": {
            "chi_max": int(cfg.mps.bond_dim_max),
            "svd_min": float(cfg.mps.svd_trunc_tol),
            },
            }
    try:
        eng = tebd.TEBDEngine(psi, M, tebd_params)
    except Exception:
        legacy = dict(tebd_params)
        legacy["delta_t"] = legacy.pop("dt")
        legacy.pop("N_steps", None)
        eng = tebd.TEBDEngine(psi, M, legacy)
    return eng, dt

# To dense (small N)
def _state_to_dense(psi: MPS) -> np.ndarray:
    """Dense |psi> as a complex vector (length 2**N), site-0 as MSB."""
    try:
        theta = psi.get_theta(0, psi.L)  # npc.Array
    except Exception as e:
        try:
            dense = ExactDiag.mps_to_full(psi)
            return np.asarray(dense, dtype=complex).reshape(-1)
        except Exception:
            raise RuntimeError("Cannot densify MPS: get_theta failed.") from e
    arr = theta.to_ndarray() if hasattr(theta, "to_ndarray") else np.asarray(theta)
    arr = np.squeeze(arr)
    return arr.reshape(-1).astype(complex)

def _apply_local_ops_to_state(vec: np.ndarray, ops: List[np.ndarray]) -> np.ndarray:
    N = len(ops)
    psi = vec.reshape([2]*N)
    for i, op in enumerate(ops):
        if op is None:
            continue
        psi = np.tensordot(op, psi, axes=(1, i))
        psi = np.moveaxis(psi, 0, i)
    return psi.reshape(-1)

def _unitary_for_basis(base: str) -> np.ndarray:
    b = base.upper()
    if b == "Z":
        return _I2
    elif b == "X":
        return _H
    elif b == "Y":
        return _SDG @ _H
    else:
        raise ValueError("Unknown basis: " + base)

def _bitstrings_lex(N: int):
    for bits in itertools.product("01", repeat=N):
        yield "".join(bits)

def _prob_columns_for_basis(N: int, base: str):
    prefix = f"p{base.upper()}_"
    return [prefix + b for b in _bitstrings_lex(N)]

def _measure_probabilities_dense(vec: np.ndarray, bases: List[str]) -> Dict[str, float]:
    N = int(round(np.log2(vec.size)))
    out: Dict[str, float] = {}
    psi0 = vec
    for base in bases:
        U = _unitary_for_basis(base)
        ops = [U] * N
        rotated = _apply_local_ops_to_state(psi0, ops)
        probs = np.abs(rotated)**2
        s = float(np.sum(probs))
        probs = probs / s if s > 0.0 else probs
        cols = _prob_columns_for_basis(N, base)
        for name, p in zip(cols, probs):
            out[name] = float(np.real(p))
    return out

def _parse_pauli_string_token(token: str):
    if len(token) < 2:
        raise ValueError(f"Bad pauli token: {token}")
    op = token[0].upper()
    idx = int(token[1:])
    if op not in ("X","Y","Z"):
        raise ValueError("Pauli must be X/Y/Z: " + token)
    return op, idx

def _ops_list_for_pauli_pattern(N: int, pattern: str) -> List[np.ndarray]:
    ops = [None] * N
    i = 0
    while i < len(pattern):
        op = pattern[i].upper()
        j = i + 1
        while j < len(pattern) and pattern[j].isdigit():
            j += 1
        token = pattern[i:j]
        op, idx = _parse_pauli_string_token(token)
        ops[idx] = _SIGMA_X if op=="X" else (_SIGMA_Y if op=="Y" else _SIGMA_Z)
        i = j
    return ops

def _measure_pauli_strings_dense(vec: np.ndarray, patterns: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for pat in patterns:
        ops = _ops_list_for_pauli_pattern(int(round(np.log2(vec.size))), pat)
        Opsi = _apply_local_ops_to_state(vec, ops)
        exp = np.vdot(vec, Opsi)  # <psi|O|psi>
        out["exp_" + pat] = float(np.real(exp))
    return out

def _compute_entropy_at_cut(psi: MPS, cut: int) -> float:
    if cut is None:
        return None
    S_e = psi.entanglement_entropy()  # per bond, in nats
    b = int(cut) - 1
    if b < 0 or b >= len(S_e):
        raise ValueError(f"entanglement_cut={cut} is out of range for N={psi.L}. Use 1..{psi.L-1}.")
    return float(S_e[b] / math.log(2.0))

def _measure_all_basic(M, psi, want: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        E = float(np.real(np.sum(M.bond_energies(psi))))
    except Exception:
        E = float(np.real(M.H_MPO.expectation_value_finite(psi)))
    if "energy" in want:
        out["energy"] = E
    need_sz = ("magnetization_z" in want) or ("sz_sites" in want)
    if need_sz:
        sz_vals = 2.0 * np.asarray(psi.expectation_value("Sz"), dtype=float)
        if "magnetization_z" in want:
            out["magnetization_z"] = float(sz_vals.mean())
        if "sz_sites" in want:
            for i, v in enumerate(sz_vals):
                out[f"sz_{i}"] = float(v)
    return out

# Open-system (trajectories) helpers
def _compute_jump_rates_per_site(psi, gamma_phi: float, gamma_down: float, gamma_up: float):
    """Compute per-site jump rates using current populations."""
    N = psi.L
    sz_vals = 2.0 * np.asarray(psi.expectation_value("Sz"), dtype=float)
    pop1 = (1.0 - sz_vals) * 0.5  # |1>
    pop0 = (1.0 + sz_vals) * 0.5  # |0>
    lam_phi  = np.full(N, float(gamma_phi), dtype=float)
    lam_down = np.asarray(gamma_down * pop1, dtype=float)  # 1->0 (σ-), TeNPy operator 'Sp'
    lam_up   = np.asarray(gamma_up * pop0, dtype=float)    # 0->1 (σ+), TeNPy operator 'Sm'
    return lam_phi, lam_down, lam_up

def _apply_local_jump(psi, i: int, kind: str):
    """Apply local jump by name using site ops (npc.Array). kind in {'Z','Sp','Sm'}.
    'Z'  -> σz (use 'Sigmaz' if present; else 2*Sz)
    'Sp' -> TeNPy 'Sp' (matrix [[0,1],[0,0]]) == σ-
    'Sm' -> TeNPy 'Sm' (matrix [[0,0],[1,0]]) == σ+
    """
    site = psi.sites[i]
    if kind == 'Z':
        try:
            op = site.get_op('Sigmaz')
        except Exception:
            op = 2.0 * site.get_op('Sz')
    elif kind == 'Sp':
        op = site.get_op('Sp')
    elif kind == 'Sm':
        op = site.get_op('Sm')
    else:
        raise ValueError(f"Unknown jump kind: {kind}")
    psi.apply_local_op(i, op)
    try:
        psi.normalize()
    except Exception:
        pass

def _sample_and_apply_jumps(psi, dt: float, gamma_phi: float, gamma_down: float, gamma_up: float, rng: np.random.Generator):
    if gamma_phi==0.0 and gamma_down==0.0 and gamma_up==0.0:
        return 0
    lam_phi, lam_down, lam_up = _compute_jump_rates_per_site(psi, gamma_phi, gamma_down, gamma_up)
    N = psi.L
    jumps = 0
    for i in range(N):
        lam_tot = lam_phi[i] + lam_down[i] + lam_up[i]
        p_tot = lam_tot * dt
        if p_tot <= 0.0:
            continue
        if rng.random() < p_tot:
            r = rng.random() * lam_tot
            if r < lam_phi[i]:
                _apply_local_jump(psi, i, 'Z')
            elif r < lam_phi[i] + lam_down[i]:
                _apply_local_jump(psi, i, 'Sp')   # 1->0 (σ-)
            else:
                _apply_local_jump(psi, i, 'Sm')   # 0->1 (σ+)
            jumps += 1
    return jumps

def run(cfg: RunConfig) -> Dict[str, Any]:
    if cfg.pbc_wrap:
        cfg.model.boundary = "PBC"

    n_traj = int(getattr(cfg.mps, "trajectories", TrajectoriesConfig()).n_traj if hasattr(cfg.mps, "trajectories") else 1)
    seed   = getattr(cfg.mps.trajectories, "seed", None) if hasattr(cfg.mps, "trajectories") else None
    rng_master = np.random.default_rng(seed)

    gamma_phi  = float(cfg.noise.dephasing_rate or 0.0)
    gamma_down = float(cfg.noise.relaxation_rate or 0.0)
    gamma_up   = float(cfg.noise.thermal_pop or 0.0)

    M_proto, psi_proto, bitstr0 = _build_model_and_state(cfg)
    eng_proto, dt = _tebd_engine(M_proto, psi_proto, cfg)

    steps = int(cfg.time.steps)
    t_max = float(cfg.time.t_max)
    ts = np.linspace(0.0, t_max, steps + 1)

    obs_names: List[str] = []
    if "energy" in cfg.observables: obs_names.append("energy")
    if "magnetization_z" in cfg.observables: obs_names.append("magnetization_z")
    if "sz_sites" in cfg.observables:
        obs_names.extend([f"sz_{i}" for i in range(cfg.model.N)])
    headers = ["t"] + obs_names

    do_dense = bool(cfg.dense_readout and (cfg.model.N <= int(cfg.dense_readout_N_max)))
    if cfg.store_loschmidt:
        headers.extend(["overlap_re", "overlap_im", "loschmidt"])
    if cfg.entanglement_cut is not None:
        headers.append("SvN_cut")
    prob_cols = []
    if do_dense and cfg.store_probabilities:
        for b in cfg.store_prob_bases:
            prob_cols.extend(_prob_columns_for_basis(cfg.model.N, b))
        headers.extend(prob_cols)
    exp_cols = []
    if do_dense and cfg.measure_pauli_strings:
        exp_cols = [f"exp_{pat}" for pat in cfg.measure_pauli_strings]
        headers.extend(exp_cols)

    K = steps + 1
    acc = {name: np.zeros(K, dtype=float) for name in headers if name != "t"}
    cnt = {name: np.zeros(K, dtype=float) for name in headers if name != "t"}

    for _ in range(n_traj):
        M, psi, _ = _build_model_and_state(cfg)
        eng, dt = _tebd_engine(M, psi, cfg)
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        psi0_dense = _state_to_dense(psi) if (cfg.store_loschmidt or do_dense) else None

        # t=0
        k = 0
        meas0 = _measure_all_basic(M, psi, cfg.observables)
        for name in obs_names:
            if name in meas0:
                acc[name][k] += meas0[name]; cnt[name][k] += 1.0

        if cfg.store_loschmidt:
            if psi0_dense is None:
                ov_re = ov_im = le = float('nan')
            else:
                ov = np.vdot(psi0_dense, psi0_dense)
                ov_re, ov_im, le = float(np.real(ov)), float(np.imag(ov)), float(np.abs(ov)**2)
            for nm, val in zip(["overlap_re","overlap_im","loschmidt"], [ov_re, ov_im, le]):
                if not np.isnan(val):
                    acc[nm][k] += val; cnt[nm][k] += 1.0

        if cfg.entanglement_cut is not None:
            Sv = _compute_entropy_at_cut(psi, cfg.entanglement_cut)
            acc["SvN_cut"][k] += Sv; cnt["SvN_cut"][k] += 1.0

        if do_dense:
            vec = psi0_dense
            if cfg.store_probabilities:
                probs0 = _measure_probabilities_dense(vec, cfg.store_prob_bases)
                for c in prob_cols:
                    acc[c][k] += probs0.get(c, 0.0); cnt[c][k] += 1.0
            if cfg.measure_pauli_strings:
                exps0 = _measure_pauli_strings_dense(vec, cfg.measure_pauli_strings)
                for c in exp_cols:
                    val = exps0.get(c, float('nan'))
                    if not np.isnan(val):
                        acc[c][k] += val; cnt[c][k] += 1.0

        # Steps
        t = 0.0
        for step in range(steps):
            eng.run()
            t += dt
            if (gamma_phi!=0.0) or (gamma_down!=0.0) or (gamma_up!=0.0):
                _sample_and_apply_jumps(psi, dt, gamma_phi, gamma_down, gamma_up, rng)

            k = step + 1
            meas = _measure_all_basic(M, psi, cfg.observables)
            for name in obs_names:
                if name in meas:
                    acc[name][k] += meas[name]; cnt[name][k] += 1.0

            if cfg.store_loschmidt:
                if psi0_dense is None:
                    ov_re = ov_im = le = float('nan')
                else:
                    vec_t = _state_to_dense(psi)
                    ov = np.vdot(psi0_dense, vec_t)
                    ov_re, ov_im, le = float(np.real(ov)), float(np.imag(ov)), float(np.abs(ov)**2)
                for nm, val in zip(["overlap_re","overlap_im","loschmidt"], [ov_re, ov_im, le]):
                    if not np.isnan(val):
                        acc[nm][k] += val; cnt[nm][k] += 1.0

            if cfg.entanglement_cut is not None:
                Sv = _compute_entropy_at_cut(psi, cfg.entanglement_cut)
                acc["SvN_cut"][k] += Sv; cnt["SvN_cut"][k] += 1.0

            if do_dense:
                vec = _state_to_dense(psi)
                if cfg.store_probabilities:
                    probs = _measure_probabilities_dense(vec, cfg.store_prob_bases)
                    for c in prob_cols:
                        acc[c][k] += probs.get(c, 0.0); cnt[c][k] += 1.0
                if cfg.measure_pauli_strings:
                    exps = _measure_pauli_strings_dense(vec, cfg.measure_pauli_strings)
                    for c in exp_cols:
                        val = exps.get(c, float('nan'))
                        if not np.isnan(val):
                            acc[c][k] += val; cnt[c][k] += 1.0

    # Write outputs
    outdir = cfg.outdir
    _ensure_dir(outdir)
    csv_path = os.path.join(outdir, "timeseries.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for k in range(steps + 1):
            row = [float(ts[k])]
            for name in headers[1:]:
                if name in acc:
                    denom = cnt[name][k] if cnt[name][k] > 0 else 1.0
                    row.append(float(acc[name][k] / denom))
                else:
                    row.append(float('nan'))
            w.writerow(row)

    manifest = {
        "run_id": cfg.run_id or f"N{cfg.model.N}_tebd_J{cfg.model.J}_D{cfg.model.Delta}",
        "solver": "tebd2" if cfg.mps.method.upper().startswith("TEBD") else "tdvp",
        "backend": "mps",
        "model": asdict(cfg.model),
        "init": asdict(cfg.init) | {"resolved_basis_string": bitstr0},
        "time": asdict(cfg.time),
        "noise": asdict(cfg.noise),
        "observables": cfg.observables,
        "stored": {
            "probabilities": list(cfg.store_prob_bases) if (do_dense and cfg.store_probabilities) else [],
            "pauli_strings_computed": list(cfg.measure_pauli_strings) if (do_dense and cfg.measure_pauli_strings) else [],
            "loschmidt": bool(cfg.store_loschmidt),
            },
        "mps": asdict(cfg.mps) | {"api": "TeNPy"},
        "entanglement_cut": cfg.entanglement_cut,
        "N": cfg.model.N,
        "trajectories": {"n_traj": n_traj, "seed": seed},
        "notes": "Rescale for Pauli convention (Jxx=4J, Jz=4Δ, hz=2h). Open-system via MCWF; H_eff adds i*(γ↓−γ↑)/2 to hz. SvN in bits. Dense readout for small N.",
        "csv": os.path.abspath(csv_path),
        }
    man_path = os.path.join(outdir, "manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return {"outdir": outdir, "csv": csv_path, "manifest": man_path, "rows": steps+1, "headers": headers}

# -------------------------
# Config loader
# -------------------------
def load_config(json_path: str) -> RunConfig:
    with open(json_path, "r") as f:
        raw = json.load(f)

    if raw.get("backend", "mps") != "mps":
        raise ValueError('This script is the MPS backend; set "backend": "mps".')

    model = ModelConfig(
        N=raw["model"]["N"],
        J=raw["model"].get("J", 1.0),
        Delta=raw["model"].get("Delta", 0.0),
        boundary=raw["model"].get("boundary", "OBC"),
        disorder=DisorderConfig(**raw["model"].get("disorder", {})),
        potential=PotentialConfig(**raw["model"].get("potential", {})),
        )
    init = InitStateConfig(**raw.get("init", {}))
    time_cfg = TimeConfig(**raw.get("time", {}))
    noise = NoiseConfig(**raw.get("noise", {}))

    mps_raw = raw.get("mps", {})
    traj_raw = mps_raw.get("trajectories", {})
    mps_cfg = MPSConfig(**{k: v for k, v in mps_raw.items() if k != "trajectories"})
    mps_cfg.trajectories = TrajectoriesConfig(**traj_raw)

    rc = RunConfig(
        backend="mps",
        model=model,
        init=init,
        time=time_cfg,
        noise=noise,
        observables=raw.get("observables", ["energy", "magnetization_z", "sz_sites"]),
        measure_pauli_strings=raw.get("measure_pauli_strings", []),
        entanglement_cut=raw.get("entanglement_cut", None),
        store_probabilities=raw.get("store_probabilities", False),
        store_prob_bases=raw.get("store_prob_bases", ["Z"]),
        store_loschmidt=raw.get("store_loschmidt", True),
        dense_readout=raw.get("dense_readout", True),
        dense_readout_N_max=raw.get("dense_readout_N_max", 12),
        random_seed=raw.get("random_seed", None),
        pbc_wrap=raw.get("pbc_wrap", False),
        outdir=raw.get("outdir", "experiments/results_mps"),
        run_id=raw.get("run_id", None),
        mps=mps_cfg,
        )
    return rc

def main():
    ap = argparse.ArgumentParser(description="MPS/TEBD backend (closed & MCWF open) + dense readout")
    ap.add_argument("--config", type=str, required=True, help="Path to JSON config")
    ap.add_argument("--outdir", type=str, default=None, help="Override output directory")
    args = ap.parse_args()

    #rc = load_config("experiments/configs/classical/mps_solution/xx_open_mps_traj.json")
    rc = load_config(args.config)
    if args.outdir:
        rc.outdir = args.outdir

    info = run(rc)
    print("Experiment ran successfully.")
    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    main()
