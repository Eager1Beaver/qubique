# src/classical/mps_solution.py
# TEBD/TDVP backend for larger-N 1D spin chains (XX / XXZ with optional Stark field).
# Reads a JSON config and writes
# (1) timeseries.csv and (2) manifest.json to the chosen outdir.
# Observables (closed systems):
#   - "energy": <psi|H|psi>
#   - "magnetization_z": average Pauli-Z over sites (same normalization as exact backend)
#   - "sz_sites": per-site Pauli-Z  (columns: sz_0, ..., sz_{N-1})
#   - "SvN_cut": bipartite entanglement entropy across the chosen cut (if entanglement_cut is set)

import json, os, csv, math
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Literal, Dict, Any

import numpy as np

try:
    from tenpy.models.xxz_chain import XXZChain
    from tenpy.networks.mps import MPS
    from tenpy.algorithms import tebd
except Exception as e:
    raise SystemExit("Please install TeNPy: pip install physics-tenpy") from e


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
    J: float = 1.0        # XX coupling
    Delta: float = 0.0    # ZZ coupling (XXZ)
    boundary: Literal["OBC", "PBC"] = "OBC"
    disorder: DisorderConfig = field(default_factory=DisorderConfig)
    potential: PotentialConfig = field(default_factory=PotentialConfig)

@dataclass
class InitStateConfig:
    kind: Literal["neel", "domain_wall", "product_random_z", "basis_string"] = "neel"
    basis_string: Optional[str] = None

@dataclass
class TimeConfig:
    t_max: float = 20.0
    steps: int = 200

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
    random_seed: Optional[int] = None
    pbc_wrap: bool = False
    outdir: str = "experiments/results/classical/mps_solution"
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
        w = cfg.disorder.strength
        h = rng.uniform(-w, w, size=N)
    elif cfg.disorder.kind == "normal":
        h = rng.normal(0.0, cfg.disorder.strength, size=N)
    else:
        raise ValueError("Unknown disorder.kind")

    if cfg.potential.type == "linear":
        slope = float(cfg.potential.slope)
        h = h + slope * np.arange(N)
    return h

def _basis_string_from_init(init: InitStateConfig, N: int, seed: Optional[int]) -> str:
    if init.kind == "neel":
        # 0101... (0=|0>, 1=|1>) with |0> ≡ spin-up along Z to match Pauli-Z = diag(1,-1)
        return "".join("01"[i % 2] for i in range(N))
    elif init.kind == "domain_wall":
        return "0"*(N//2) + "1"*(N - N//2)
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
    # TeNPy expects labels per site, convention 'up'/'down' for spin-1/2 sites
    # We'll map '0' -> 'up' (Z=+1), '1' -> 'down' (Z=-1)
    return ['up' if ch == '0' else 'down' for ch in basis_string]

def _build_model_and_state(cfg: RunConfig):
    N = cfg.model.N
    hz = _build_hz_array(cfg.model)

    # TEBD with PBC is much heavier
    if cfg.model.boundary != "OBC":
        raise ValueError("XXZChain backend currently supports OBC (bc_MPS='finite') only.")
    bc_mps = "finite"

    model_params = dict(
        L=N,
        Jxx=float(cfg.model.J),       # XX coupling
        Jz=float(cfg.model.Delta),    # ZZ coupling (Δ)
        hz=hz,                        # can be scalar or length-N array
        bc_MPS=bc_mps,
        conserve=cfg.mps.conserve,    # 'Sz' | 'parity' | None
        )
    M = XXZChain(model_params)

    # Initial product state
    bstr = _basis_string_from_init(cfg.init, N, cfg.random_seed)
    prod = _product_state_list(bstr)        # ['up'/'down'] per site
    sites = M.lat.mps_sites()
    psi = MPS.from_product_state(sites, prod, bc=bc_mps)
    psi.canonical_form()
    return M, psi, bstr

def _tebd_engine(M, psi, cfg: RunConfig):
    dt = float(cfg.time.t_max) / int(cfg.time.steps)
    base = {
        "order": 2 if cfg.mps.method.upper().startswith("TEBD2") else 2,
        "dt": dt,
        "N_steps": 1,
        "trunc_params": {"chi_max": int(cfg.mps.bond_dim_max), "svd_min": float(cfg.mps.svd_trunc_tol)},
        }
    try:
        eng = tebd.TEBDEngine(psi, M, base)
    except Exception:
        legacy = dict(base)
        legacy["delta_t"] = legacy.pop("dt")
        legacy.pop("N_steps", None)
        eng = tebd.TEBDEngine(psi, M, legacy)
    return eng, dt

def _measure_all(M, psi, want: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}

    # Energy
    if "energy" in want:
        # Prefer bond_energies for NN chains; fall back to MPO evaluation for finite chains.
        try:
            E = float(np.real(np.sum(M.bond_energies(psi))))
        except Exception:
            E = float(np.real(M.H_MPO.expectation_value_finite(psi)))
        out["energy"] = E

    # σ^z per site and average
    need_sz = ("magnetization_z" in want) or ("sz_sites" in want)
    if need_sz:
        # 'Sz' has eigenvalues ±1/2; multiply by 2.0 for Pauli (±1) parity with QuTiP outputs.
        sz_vals = 2.0 * np.asarray(psi.expectation_value("Sz"), dtype=float)
        if "magnetization_z" in want:
            out["magnetization_z"] = float(sz_vals.mean())
        if "sz_sites" in want:
            for i, v in enumerate(sz_vals):
                out[f"sz_{i}"] = float(v)

    return out

def _compute_entropy_at_cut(psi: MPS, cut: int) -> float:
    """Return von Neumann entropy across cut (A: [0..cut-1] | B: [cut..N-1]).
    TeNPy stores entanglement entropies per bond: bond index b is between sites b and b+1.
    So cut in [1..N-1] maps to bond b = cut - 1.
    """
    if cut is None:
        return None
    psi.entanglement_entropy()  # update cached entanglement
    b = int(cut) - 1
    if b < 0 or b >= len(psi._S):  # len = N-1
        raise ValueError(f"entanglement_cut={cut} is out of range for N={psi.L}. Use 1..{psi.L-1}.")
    # psi.entanglement_entropy() already filled psi._S 'entanglement entropies' (base e); TeNPy uses base e by default.
    # Converting to base-2 to match exact_solution.py (SvN in bits).
    S_e = psi.entanglement_entropy()[b]
    S_2 = float(S_e / math.log(2.0))
    return S_2

def run(cfg: RunConfig) -> Dict[str, Any]:
    # Basic guards
    if cfg.noise.dephasing_rate > 0.0 or cfg.noise.relaxation_rate > 0.0 or (cfg.noise.thermal_pop not in (None, 0.0)):
        raise NotImplementedError("Open-system (Lindblad) evolution is not yet implemented in the MPS backend. "
                                  "Keep closed-system runs here and use the exact (QuTiP) backend for small-N open-system validation.")

    if cfg.pbc_wrap:
        cfg.model.boundary = "PBC"

    if cfg.random_seed is not None:
        np.random.seed(cfg.random_seed)

    # Build model, initial state, and TEBD engine
    M, psi, bitstr0 = _build_model_and_state(cfg)
    eng, dt = _tebd_engine(M, psi, cfg)

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
    if cfg.entanglement_cut is not None:
        headers.append("SvN_cut")

    # Evolve, record
    rows: List[List[float]] = []

    # t=0
    meas0 = _measure_all(M, psi, cfg.observables)
    row0 = [0.0] + [meas0.get(name, float('nan')) for name in obs_names]
    if cfg.entanglement_cut is not None:
        row0.append(_compute_entropy_at_cut(psi, cfg.entanglement_cut))
    rows.append(row0)

    # Subsequent times
    t = 0.0
    for _ in range(steps):
        eng.run()                       # advance by +dt
        t += dt
        meas = _measure_all(M, psi, cfg.observables)
        row = [float(t)] + [meas.get(name, float('nan')) for name in obs_names]
        if cfg.entanglement_cut is not None:
            row.append(_compute_entropy_at_cut(psi, cfg.entanglement_cut))
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
            "probabilities": False,
            "pauli_strings_computed": [],
        },
        "mps": asdict(cfg.mps) | {"api": "TeNPy"},
        "entanglement_cut": cfg.entanglement_cut,
        "N": cfg.model.N,
        "notes": "Pauli strings disabled in v0 (will be added in a follow-up). Entropy is base-2.",
        "csv": os.path.abspath(csv_path),
    }
    man_path = os.path.join(outdir, "manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return {"outdir": outdir, "csv": csv_path, "manifest": man_path, "rows": len(rows), "headers": headers}


# -------------------------
# Config loader
# -------------------------
def load_config(json_path: str) -> RunConfig:
    with open(json_path, "r") as f:
        raw = json.load(f)

    # Allow to omit 'backend' or put 'mps' explicitly
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
        random_seed=raw.get("random_seed", None),
        pbc_wrap=raw.get("pbc_wrap", False),
        outdir=raw.get("outdir", "experiments/results_mps"),
        run_id=raw.get("run_id", None),
        mps=mps_cfg,
        )
    return rc


def main():
    '''ap = argparse.ArgumentParser(description="MPS/TEBD backend for 1D XX/XXZ chains (TeNPy)")
    ap.add_argument("--config", type=str, required=True, help="Path to JSON config")
    ap.add_argument("--outdir", type=str, default=None, help="Override output directory")
    args = ap.parse_args()'''

    # experiments/configs/classical/mps_solution/xx_closed_mps.json
    #experiments/configs/classical/mps_solution/xxz_stark_closed_mps.json
    rc = load_config("experiments/configs/classical/mps_solution/xxz_stark_closed_mps.json")
    '''rc = load_config(args.config)
    if args.outdir:
        rc.outdir = args.outdir'''

    info = run(rc)
    print("Experiment ran successfully.")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
