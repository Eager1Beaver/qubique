# src/mps/mps_solution.py
# TEBD backend for XX/XXZ (OBC) with optional dense readout at small N to mirror the exact backend.
# Features:
# - Matches QuTiP "Pauli" convention by rescaling couplings: Jxx=4J, Jz=4Δ, hz=2h
# - Observables: energy, magnetization_z (Pauli norm), sz_i, SvN_cut (base-2)
# - Loschmidt echo (pure state): |<psi(0)|psi(t)>|^2
# - Pauli string expectations: exp_<pattern> (e.g., exp_X0X1X2, exp_Z0Z5)
# - Full probabilities in Z/X/Y bases at small N: pZ_<bitstring>, pX_*, pY_*
#
# Dense readout is enabled only when `dense_readout` is true AND N <= dense_readout_N_max.
# For larger N, TEBD still runs; strings & probabilities are skipped to avoid 2^N blow-up.
#
# CLI:
#   python -m src.mps.mps_solution --config path/to/config.json [--outdir OUT]
#
# JSON fields (superset of exact backend):
#   backend: "mps"
#   model: { N, J, Delta, boundary, disorder, potential }
#   init:  { kind, basis_string }
#   time:  { t_max, steps }
#   noise: (ignored here; closed system only)
#   observables: ["energy","magnetization_z","sz_sites"]
#   entanglement_cut: int | null
#   measure_pauli_strings: ["X0X1", "Z0Z2Z3", ...]
#   store_probabilities: true|false
#   store_prob_bases: ["Z","X","Y"]
#   store_loschmidt: true|false
#   dense_readout: true|false
#   dense_readout_N_max: int (default 12)
#   mps: { method, bond_dim_max, svd_trunc_tol, conserve, two_site }
#
import argparse, json, os, csv, math, itertools, warnings
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Literal, Dict, Any

import numpy as np

# TeNPy imports
try:
    from tenpy.models.xxz_chain import XXZChain
    from tenpy.networks.mps import MPS
    from tenpy.algorithms import tebd
except Exception as e:
    raise SystemExit("Please install TeNPy first: pip install tenpy\n"
                     "Docs: https://tenpy.readthedocs.io/") from e


# -------------------------
# Config
# -------------------------
@dataclass
class NoiseConfig:
    dephasing_rate: float = 0.0
    relaxation_rate: float = 0.0
    thermal_pop: Optional[float] = None

@dataclass
class DisorderConfig:
    kind: Literal["none", "uniform", "normal"] = "none"
    strength: float = 0.0
    seed: Optional[int] = None

@dataclass
class PotentialConfig:
    type: Literal["none", "linear"] = "none"
    slope: float = 0.0    # F in F * i * Z_i  (i starts at 0)

@dataclass
class ModelConfig:
    N: int = 6
    J: float = 1.0        # Pauli convention
    Delta: float = 0.0    # Pauli convention
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
    steps: int = 800

@dataclass
class MPSConfig:
    method: Literal["TEBD2", "TDVP"] = "TEBD2"
    bond_dim_max: int = 256
    svd_trunc_tol: float = 1e-9
    conserve: Literal["Sz", "None"] = "Sz"
    two_site: bool = True

@dataclass
class RunConfig:
    backend: Literal["mps"] = "mps"
    model: ModelConfig = field(default_factory=ModelConfig)
    init: InitStateConfig = field(default_factory=InitStateConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    observables: List[str] = field(default_factory=lambda: ["energy", "magnetization_z", "sz_sites"])
    measure_pauli_strings: List[str] = field(default_factory=list)
    entanglement_cut: Optional[int] = None
    store_probabilities: bool = False
    store_prob_bases: List[str] = field(default_factory=lambda: ["Z"])
    store_loschmidt: bool = True
    dense_readout: bool = True
    dense_readout_N_max: int = 12
    random_seed: Optional[int] = None
    pbc_wrap: bool = False
    outdir: str = "experiments/results/classical/results_mps"
    run_id: Optional[str] = None
    mps: MPSConfig = field(default_factory=MPSConfig)


# -------------------------
# Utilities
# -------------------------
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
    # Match exact backend convention:
    #   neel starts with '1' at site 0: "1010..." (|1>=down/−1)
    if init.kind == "neel":
        return "".join("10"[i % 2] for i in range(N))
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
        raise ValueError("Unknown init.kind")

def _product_state_list(basis_string: str) -> List[str]:
    # Map '0' -> 'up' (Z=+1), '1' -> 'down' (Z=-1) as in exact backend
    return ['up' if ch == '0' else 'down' for ch in basis_string]

def _build_model_and_state(cfg: RunConfig):
    if cfg.noise.dephasing_rate > 0.0 or cfg.noise.relaxation_rate > 0.0 or (cfg.noise.thermal_pop not in (None, 0.0)):
        raise NotImplementedError("Open-system (Lindblad) is not yet implemented for MPS. Use exact backend for small-N open-system validation.")

    N = cfg.model.N
    if cfg.model.boundary != "OBC":
        raise ValueError("XXZChain backend currently supports OBC only (bc_MPS='finite').")

    # Rescale to match Pauli convention of exact backend
    hz_exact = _build_hz_array(cfg.model)
    hz = 2.0 * hz_exact
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
        # Fallback for older signatures
        legacy = dict(tebd_params)
        legacy["delta_t"] = legacy.pop("dt")
        legacy.pop("N_steps", None)
        eng = tebd.TEBDEngine(psi, M, legacy)
    return eng, dt

# To dense (for small N)
def _state_to_dense(psi) -> np.ndarray:
    """
    Dense |psi> as a complex vector (length 2**N), site-0 as MSB.
    Uses TeNPy's public API: psi.get_theta(0, L) -> npc.Array with legs
    ['vL', 'p0', ..., 'p{N-1}', 'vR'] for finite MPS; boundary legs have dim=1.
    """
    try:
        theta = psi.get_theta(0, psi.L)        # npc.Array
    except Exception as e:
        # Fallback: try the ED helper if present in your version
        try:
            from tenpy.algorithms.exact_diag import ExactDiag
            # Some TeNPy versions require an instance; if unavailable, re-raise.
            dense = ExactDiag.mps_to_full(psi)  # may exist in your install
            return np.asarray(dense, dtype=complex).reshape(-1)
        except Exception:
            raise RuntimeError("Cannot densify MPS: get_theta failed and no ED fallback.") from e

    # Convert npc.Array -> ndarray; squeeze 1x boundary legs; flatten with C-order.
    arr = theta.to_ndarray() if hasattr(theta, "to_ndarray") else np.asarray(theta)
    arr = np.squeeze(arr)
    return arr.reshape(-1).astype(complex)

def _apply_local_ops_to_state(vec: np.ndarray, ops: List[np.ndarray]) -> np.ndarray:
    """Apply per-site 2x2 ops to |psi> without building a 2^N x 2^N matrix.
    vec: shape (2**N,)
    ops: list of length N with 2x2 matrices or None for identity.
    """
    N = len(ops)
    psi = vec.reshape([2]*N)
    for i, op in enumerate(ops):
        if op is None:
            continue
        psi = np.tensordot(op, psi, axes=(1, i))
        psi = np.moveaxis(psi, 0, i)  # restore site order
    return psi.reshape(-1)

# Pauli and single-qubit unitaries
_SIGMA_X = np.array([[0, 1],[1, 0]], dtype=complex)
_SIGMA_Y = np.array([[0, -1j],[1j, 0]], dtype=complex)
_SIGMA_Z = np.array([[1, 0],[0, -1]], dtype=complex)
_I2      = np.eye(2, dtype=complex)
_H       = np.array([[1, 1],[1, -1]], dtype=complex) / np.sqrt(2)
_SDG     = np.array([[1, 0],[0, -1j]], dtype=complex)  # S^\dagger

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
    # msb = site 0 (leftmost), matches QuTiP default ordering
    for bits in itertools.product("01", repeat=N):
        yield "".join(bits)

def _prob_columns_for_basis(N: int, base: str):
    prefix = f"p{base.upper()}_"
    return [prefix + b for b in _bitstrings_lex(N)]

def _measure_probabilities_dense(vec: np.ndarray, bases: List[str]) -> Dict[str, float]:
    """Return probabilities for each basis as {col_name: prob}."""
    N = int(round(np.log2(vec.size)))
    out: Dict[str, float] = {}
    psi0 = vec
    for base in bases:
        U = _unitary_for_basis(base)
        ops = [U] * N
        rotated = _apply_local_ops_to_state(psi0, ops)
        probs = np.abs(rotated)**2
        # normalize (guard against tiny round-off)
        s = float(np.sum(probs))
        if s <= 0.0:
            probs = np.zeros_like(probs)
        else:
            probs = probs / s
        cols = _prob_columns_for_basis(N, base)
        for name, p in zip(cols, probs):
            out[name] = float(np.real(p))
    return out

def _parse_pauli_string_token(token: str):
    # e.g., "X0", "Z12"
    if len(token) < 2:
        raise ValueError(f"Bad pauli token: {token}")
    op = token[0].upper()
    idx = int(token[1:])
    if op not in ("X","Y","Z"):
        raise ValueError("Pauli must be X/Y/Z: " + token)
    return op, idx

def _ops_list_for_pauli_pattern(N: int, pattern: str) -> List[np.ndarray]:
    """pattern like 'X0X1X2' or 'Z0Z5'."""
    ops = [None] * N
    # break into tokens op+index
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
    N = int(round(np.log2(vec.size)))
    out: Dict[str, float] = {}
    for pat in patterns:
        ops = _ops_list_for_pauli_pattern(N, pat)
        Opsi = _apply_local_ops_to_state(vec, ops)
        exp = np.vdot(vec, Opsi)  # <psi|O|psi>
        out["exp_" + pat] = float(np.real(exp))
    return out

def _compute_entropy_at_cut(psi: MPS, cut: int) -> float:
    """Return von Neumann entropy across cut (A: [0..cut-1] | B: [cut..N-1]) in bits.
    cut in [1..N-1]."""
    if cut is None:
        return None
    S_e = psi.entanglement_entropy()  # list per bond in nats
    b = int(cut) - 1
    if b < 0 or b >= len(S_e):
        raise ValueError(f"entanglement_cut={cut} is out of range for N={psi.L}. Use 1..{psi.L-1}.")
    return float(S_e[b] / math.log(2.0))

def _measure_all_basic(M, psi, want: List[str]) -> Dict[str, float]:
    """Compute basic observables that don't require dense conversion."""
    out: Dict[str, float] = {}

    # Energy: sum of local bond energies (nearest-neighbor model)
    try:
        E = float(np.real(np.sum(M.bond_energies(psi))))
    except Exception:
        # fallback: MPO self-evaluation for finite chains
        E = float(np.real(M.H_MPO.expectation_value_finite(psi)))
    if "energy" in want:
        out["energy"] = E

    # Pauli-Z normalization: 2*<Sz>
    need_sz = ("magnetization_z" in want) or ("sz_sites" in want)
    if need_sz:
        sz_vals = 2.0 * np.asarray(psi.expectation_value("Sz"), dtype=float)
        if "magnetization_z" in want:
            out["magnetization_z"] = float(sz_vals.mean())
        if "sz_sites" in want:
            for i, v in enumerate(sz_vals):
                out[f"sz_{i}"] = float(v)

    return out


def run(cfg: RunConfig) -> Dict[str, Any]:
    if cfg.pbc_wrap:
        cfg.model.boundary = "PBC"

    if cfg.random_seed is not None:
        np.random.seed(cfg.random_seed)

    # Build model & engine
    M, psi, bitstr0 = _build_model_and_state(cfg)
    eng, dt = _tebd_engine(M, psi, cfg)

    # Dense readout check
    do_dense = bool(cfg.dense_readout and (cfg.model.N <= int(cfg.dense_readout_N_max)))
    if (cfg.store_probabilities or cfg.measure_pauli_strings) and not do_dense:
        warnings.warn("Skipping probabilities/pauli-strings: set dense_readout=true and N<=dense_readout_N_max.")

    # Time grid
    steps = int(cfg.time.steps)
    t_max = float(cfg.time.t_max)
    ts = np.linspace(0.0, t_max, steps + 1)

    # CSV headers
    obs_names: List[str] = []
    if "energy" in cfg.observables:            obs_names.append("energy")
    if "magnetization_z" in cfg.observables:   obs_names.append("magnetization_z")
    if "sz_sites" in cfg.observables:
        obs_names.extend([f"sz_{i}" for i in range(cfg.model.N)])
    headers = ["t"] + obs_names

    if cfg.store_loschmidt:
        headers.extend(["overlap_re", "overlap_im", "loschmidt"])

    if cfg.entanglement_cut is not None:
        headers.append("SvN_cut")

    # dense-derived columns
    prob_cols = []
    if do_dense and cfg.store_probabilities:
        for b in cfg.store_prob_bases:
            prob_cols.extend(_prob_columns_for_basis(cfg.model.N, b))
        headers.extend(prob_cols)

    exp_cols = []
    if do_dense and cfg.measure_pauli_strings:
        exp_cols = [f"exp_{pat}" for pat in cfg.measure_pauli_strings]
        headers.extend(exp_cols)

    # Prepare initial dense state if needed (for loschmidt and dense readouts)
    psi0_dense = _state_to_dense(psi) if (cfg.store_loschmidt or do_dense) else None

    # t=0
    rows: List[List[float]] = []
    meas0 = _measure_all_basic(M, psi, cfg.observables)
    row0 = [0.0] + [meas0.get(name, float('nan')) for name in obs_names]

    # Loschmidt at t=0
    if cfg.store_loschmidt:
        ov = complex(1.0+0j) if psi0_dense is None else np.vdot(psi0_dense, psi0_dense)
        row0.extend([float(np.real(ov)), float(np.imag(ov)), float(np.abs(ov)**2)])

    # Entropy
    if cfg.entanglement_cut is not None:
        row0.append(_compute_entropy_at_cut(psi, cfg.entanglement_cut))

    # Dense readouts
    if do_dense:
        vec = psi0_dense
        if cfg.store_probabilities:
            probs0 = _measure_probabilities_dense(vec, cfg.store_prob_bases)
            row0.extend([probs0.get(c, 0.0) for c in prob_cols])
        if cfg.measure_pauli_strings:
            exps0 = _measure_pauli_strings_dense(vec, cfg.measure_pauli_strings)
            row0.extend([exps0.get(c, float('nan')) for c in exp_cols])

    rows.append(row0)

    # Subsequent times
    t = 0.0
    for k in range(steps):
        eng.run()   # step by dt
        t += dt

        meas = _measure_all_basic(M, psi, cfg.observables)
        row = [float(t)] + [meas.get(name, float('nan')) for name in obs_names]

        if cfg.store_loschmidt:
            if psi0_dense is None:
                row.extend([float('nan'), float('nan'), float('nan')])
            else:
                vec_t = _state_to_dense(psi)
                ov = np.vdot(psi0_dense, vec_t)
                row.extend([float(np.real(ov)), float(np.imag(ov)), float(np.abs(ov)**2)])

        if cfg.entanglement_cut is not None:
            row.append(_compute_entropy_at_cut(psi, cfg.entanglement_cut))

        if do_dense:
            vec = _state_to_dense(psi)
            if cfg.store_probabilities:
                probs = _measure_probabilities_dense(vec, cfg.store_prob_bases)
                row.extend([probs.get(c, 0.0) for c in prob_cols])
            if cfg.measure_pauli_strings:
                exps = _measure_pauli_strings_dense(vec, cfg.measure_pauli_strings)
                row.extend([exps.get(c, float('nan')) for c in exp_cols])

        rows.append(row)

    # Output
    outdir = cfg.outdir
    _ensure_dir(outdir)
    run_id = cfg.run_id or f"N{cfg.model.N}_tebd_J{cfg.model.J}_D{cfg.model.Delta}"
    csv_path = os.path.join(outdir, "timeseries.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow(r)

    manifest = {
        "run_id": run_id,
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
        "notes": "Coupling rescale for Pauli convention: Jxx=4J, Jz=4Δ, hz=2h. SvN in bits. Dense readout only for small N.",
        "csv": os.path.abspath(csv_path),
        }
    man_path = os.path.join(outdir, "manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return {"outdir": outdir, "csv": csv_path, "manifest": man_path, "rows": len(rows), "headers": headers}


# -------------------------
# JSON loader
# -------------------------
def load_config(json_path: str) -> RunConfig:
    with open(json_path, "r") as f:
        raw = json.load(f)

    if raw.get("backend", "mps") != "mps":
        raise ValueError('This script is the MPS backend; set "backend": "mps" in your config or use the exact backend.')

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
    mps_cfg = MPSConfig(**raw.get("mps", {}))

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
    '''ap = argparse.ArgumentParser(description="MPS/TEBD backend for 1D XX/XXZ chains (TeNPy) + dense readout for small N")
    ap.add_argument("--config", type=str, required=True, help="Path to JSON config")
    ap.add_argument("--outdir", type=str, default=None, help="Override output directory")
    args = ap.parse_args()'''

    rc = load_config("experiments/configs/classical/mps_solution/xx_closed_mps_dense.json")
    '''rc = load_config(args.config)
    if args.outdir:
        rc.outdir = args.outdir'''

    info = run(rc)
    print("Experiment ran successfully.")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
