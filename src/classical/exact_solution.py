# src/classical/exact_solution.py
# Exact small-chain evolution (closed/open) using QuTiP.
# Outputs: CSV (time series), manifest.json, optional trajectories (.npz).
# Ready for benchmarks: Loschmidt echo (pure/mixed), Fisher info (X/Y/Z bases),
# Mermin (Pauli strings), Entanglement (SvN closed / log-negativity open).

import argparse, json, os, csv, math, itertools, random
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Literal, Dict

import numpy as np
try:
    import qutip as qt
except ImportError as e:
    raise SystemExit("Please install QuTiP: pip install qutip") from e


# =========================
# Config schema
# =========================

@dataclass
class NoiseConfig:
    dephasing_rate: float = 0.0          # gamma_phi (Z dephasing)
    relaxation_rate: float = 0.0         # gamma_relax (|1> -> |0>)
    thermal_pop: Optional[float] = None  # excited-state pop for thermal (None => 0)

@dataclass
class DisorderConfig:
    kind: Literal["none", "uniform", "normal"] = "none"
    strength: float = 0.0               # half-width (uniform) or std (normal)
    seed: Optional[int] = None

@dataclass
class PotentialConfig:
    type: Literal["none", "linear"] = "none"
    slope: float = 0.0                  # F in F * i * sz_i (i starts at 0)

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
    seed: Optional[int] = None

@dataclass
class TimeConfig:
    t_max: float = 8.0
    steps: int = 200

@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    init: InitStateConfig = field(default_factory=InitStateConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)

    # Storage / measurements for benchmarks
    store_probabilities: bool = False
    store_prob_bases: List[str] = None          # subset of ["Z","X","Y"], default ["Z"]
    measure_pauli_strings: List[str] = None     # e.g., ["X0X1X2","Z0Z1"]
    entanglement_cut: Optional[int] = None      # bipartition cut index
    store_trajectory: Literal["none", "auto", "ket", "rho"] = "none"
    store_state: bool = False                   # keep final state (legacy; kept for compat)

    observables: List[str] = None               # ["energy","magnetization_z","sz_sites"]
    random_seed: Optional[int] = None
    pbc_wrap: bool = False

    # Output
    outdir: str = "experiments/results"
    run_id: Optional[str] = None


# =========================
# Operators & builders
# =========================

def paulis():
    return qt.sigmax(), qt.sigmay(), qt.sigmaz(), qt.qeye(2)

def embed(op, site, N):
    I = qt.qeye(2)
    ops = [I]*N
    ops[site] = op
    return qt.tensor(ops)

def two_site_coupling(opL, i, opR, j, N):
    return embed(opL, i, N) * embed(opR, j, N)

def build_hamiltonian(cfg: ModelConfig):
    """
    H = sum_i J (Sx_i Sx_{i+1} + Sy_i Sy_{i+1}) + Delta Sz_i Sz_{i+1}
        + sum_i (h_i + F*i) Sz_i
    """
    N, J, Delta = cfg.N, cfg.J, cfg.Delta
    sx, sy, sz, I = paulis()

    pairs = [(i, i+1) for i in range(N-1)]
    if cfg.boundary == "PBC":
        pairs.append((N-1, 0))

    H = 0
    for (i, j) in pairs:
        H += J * (two_site_coupling(sx, i, sx, j, N) + two_site_coupling(sy, i, sy, j, N))
        if abs(Delta) > 0:
            H += Delta * two_site_coupling(sz, i, sz, j, N)

    # Disorder fields
    rng = np.random.default_rng(cfg.disorder.seed) if cfg.disorder.seed is not None else np.random.default_rng()
    if cfg.disorder.kind == "none":
        h = np.zeros(N)
    elif cfg.disorder.kind == "uniform":
        w = cfg.disorder.strength
        h = rng.uniform(-w, w, size=N)
    elif cfg.disorder.kind == "normal":
        h = rng.normal(0.0, cfg.disorder.strength, size=N)
    else:
        raise ValueError("Unknown disorder kind")

    # Linear potential
    if cfg.potential.type == "linear":
        F = cfg.potential.slope
        h = h + F * np.arange(N)

    for i in range(N):
        H += h[i] * embed(qt.sigmaz(), i, N)

    return H, h

def build_initial_state(cfg: InitStateConfig, N: int):
    """
    |0> := |↑z>, |1> := |↓z>. Returns (psi0, bitstr).
    """
    if cfg.kind == "neel":
        bitstr = ''.join('10'[(i % 2)] for i in range(N))  # "1010..."
    elif cfg.kind == "domain_wall":
        half = N // 2
        bitstr = '1'*half + '0'*(N-half)
    elif cfg.kind == "basis_string":
        if cfg.basis_string is None or len(cfg.basis_string) != N or any(c not in "01" for c in cfg.basis_string):
            raise ValueError("basis_string must be 0/1 string of length N")
        bitstr = cfg.basis_string
    elif cfg.kind == "product_random_z":
        rng = random.Random(cfg.seed)
        bitstr = ''.join(rng.choice('01') for _ in range(N))
    else:
        raise ValueError("Unknown init.kind")

    kets = {'0': qt.basis(2, 0), '1': qt.basis(2, 1)}
    psi0 = qt.tensor([kets[c] for c in bitstr])
    # set composite dims for ptrace later
    psi0.dims = [[2]*N, [1]]
    return psi0, bitstr

def build_collapse_ops(noise: NoiseConfig, N: int):
    """
    Dephasing: L = sqrt(gamma_phi) * sz
    Relaxation: L = sqrt(gamma_relax) * sigmam
    Thermal excitation: sqrt(gamma_relax * p_th) * sigmap
    """
    if (noise.dephasing_rate <= 0) and (noise.relaxation_rate <= 0) and (not noise.thermal_pop):
        return []

    c_ops = []
    sqrt = math.sqrt
    for i in range(N):
        if noise.dephasing_rate > 0:
            c_ops.append(sqrt(noise.dephasing_rate) * embed(qt.sigmaz(), i, N))
        if noise.relaxation_rate > 0:
            c_ops.append(sqrt(noise.relaxation_rate) * embed(qt.sigmam(), i, N))
            if noise.thermal_pop and noise.thermal_pop > 0:
                c_ops.append(sqrt(noise.relaxation_rate * noise.thermal_pop) * embed(qt.sigmap(), i, N))
    return c_ops

# ----- Pauli strings -----
PAULI_MAP = {"X": qt.sigmax(), "Y": qt.sigmay(), "Z": qt.sigmaz(), "I": qt.qeye(2)}

def pauli_string_op(spec: str, N: int):
    """
    spec like "X0Y1Y2", "Z0Z1", "X0". Missing sites => identity.
    """
    sites: Dict[int, qt.Qobj] = {}
    i = 0
    while i < len(spec):
        p = spec[i].upper()
        if p not in PAULI_MAP:
            raise ValueError(f"Invalid Pauli '{p}' in {spec}")
        i += 1
        j = i
        while j < len(spec) and spec[j].isdigit():
            j += 1
        if j == i:
            raise ValueError(f"Missing index after {p} in {spec}")
        idx = int(spec[i:j])
        i = j
        sites[idx] = PAULI_MAP[p]
    ops = [sites.get(s, PAULI_MAP["I"]) for s in range(N)]
    return qt.tensor(ops)

def build_pauli_ops(strings: List[str], N: int):
    return {s: pauli_string_op(s, N) for s in strings}

# ----- Observables -----
def build_observables(N: int, want: List[str], H=None):
    sx, sy, sz, I = paulis()
    obs = {}
    if "energy" in want:
        if H is None:
            raise ValueError("Energy requested but Hamiltonian H not provided")
        obs["energy"] = H
    if "magnetization_z" in want:
        obs["magnetization_z"] = sum(embed(sz, i, N) for i in range(N)) / N
    if "sz_sites" in want:
        for i in range(N):
            obs[f"sz_{i}"] = embed(sz, i, N)
    return obs

# ----- Basis rotations -----
def hadamard():
    return qt.Qobj(np.array([[1, 1],[1, -1]], dtype=complex)) / np.sqrt(2)

def s_dagger():
    return qt.Qobj(np.array([[1, 0],[0, -1j]], dtype=complex))

def rotation_unit_for_basis(base: str):
    base = base.upper()
    if base == "Z":
        return qt.qeye(2)
    elif base == "X":
        return hadamard()
    elif base == "Y":
        return s_dagger() * hadamard()
    else:
        raise ValueError("Unknown basis; allowed: Z, X, Y")

def rotate_state_to_basis(state_like: qt.Qobj, base: str, N: int):
    U1 = rotation_unit_for_basis(base)
    U = qt.tensor([U1]*N)
    if state_like.isket:
        out = U * state_like
        out.dims = [[2]*N, [1]]
        return out
    else:
        out = U * state_like * U.dag()
        out.dims = [[2]*N, [2]*N]
        return out

# ----- Entanglement helpers -----
def svn_bipartition_from_ket(psi: qt.Qobj, cut: int):
    """von Neumann entropy across cut (0..cut-1 | cut..N-1) for pure state."""
    N = len(psi.dims[0])
    A = list(range(cut))
    B = list(range(cut, N))
    # Reduced rho_A
    rhoA = qt.ptrace(psi, A)
    return float(qt.entropy_vn(rhoA, base=2))

def log_negativity_from_rho(rho: qt.Qobj, cut: int):
    """
    Log-negativity across cut for a mixed state: log2 || rho^{T_B} ||_1.
    Works across QuTiP versions that differ in partial_transpose signature.
    """
    # Ensure dims are set properly: [[2]*N, [2]*N]
    if not rho.isoper:
        raise ValueError("log_negativity_from_rho expects a density operator (Qobj.isoper == True).")

    if rho.dims is None or len(rho.dims[0]) == 0:
        # Infer N from matrix size as a fallback
        N = int(round(np.log2(rho.shape[0])))
        rho = rho.copy()
        rho.dims = [[2]*N, [2]*N]
    else:
        N = len(rho.dims[0])

    # Build mask: 0 for A (no transpose), 1 for B (partial transpose on these)
    A = set(range(cut))
    mask = [0 if i in A else 1 for i in range(N)]

    # Partial transpose (no 'dims' kwarg; relies on rho.dims)
    try:
        rho_pt = qt.partial_transpose(rho, mask)
    except TypeError:
        # Older/newer variants still use the same call signature without dims
        rho_pt = qt.partial_transpose(rho, mask)

    # Trace norm of rho^{T_B}
    evals = np.linalg.eigvals(rho_pt.full())
    tr_norm = float(np.sum(np.abs(evals)).real)

    # Numerical guard: negativity >= 0
    negativity = max((tr_norm - 1.0) / 2.0, 0.0)

    # Log-negativity base 2
    return float(np.log2(2.0 * negativity + 1.0))


# =========================
# Core run
# =========================

def run(config: RunConfig):
    # normalize
    if config.pbc_wrap:
        config.model.boundary = "PBC"

    if config.random_seed is not None:
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)

    # Hamiltonian & initial state
    H, realized_h = build_hamiltonian(config.model)
    psi0, bitstr0 = build_initial_state(config.init, config.model.N)

    # time grid
    ts = np.linspace(0.0, config.time.t_max, config.time.steps + 1)

    # open vs closed
    c_ops = build_collapse_ops(config.noise, config.model.N)
    closed = (len(c_ops) == 0)

    # observables
    want = config.observables or ["energy", "magnetization_z", "sz_sites"]
    obs_map = build_observables(config.model.N, want, H=H)
    e_ops = list(obs_map.values())
    obs_names = list(obs_map.keys())

    # Pauli strings
    pauli_strings = list(config.measure_pauli_strings or [])
    pauli_ops = build_pauli_ops(pauli_strings, config.model.N)

    # probabilities configuration
    prob_on = bool(config.store_probabilities)
    prob_bases = (config.store_prob_bases or ["Z"])
    prob_bases = [b.upper() for b in prob_bases]
    for b in prob_bases:
        if b not in ("Z","X","Y"):
            raise ValueError("store_prob_bases must be subset of ['Z','X','Y']")

    # basis bitstrings
    basis_bitstrings = None
    if prob_on:
        N = config.model.N
        basis_bitstrings = [''.join(seq) for seq in itertools.product('01', repeat=N)]

    # evolve
    # Decide if we need to store the full state trajectory:
    need_states = True  # we need states for probs/entanglement/trajectory/pauli expectations
    #opts = qt.Options(store_states=need_states, progress_bar=None)
    opts: dict = {"store_states": need_states, "progress_bar": None}

    if closed:
        result = qt.sesolve(H, psi0, ts, e_ops=e_ops if e_ops else None, options=opts)
        states = result.states  # list of kets (may be [])
        rho_final = None
    else:
        rho0 = psi0 * psi0.dag()
        rho0.dims = [[2]*config.model.N, [2]*config.model.N]
        result = qt.mesolve(H, rho0, ts, c_ops=c_ops, e_ops=e_ops if e_ops else None, options=opts)
        states = result.states  # list of density matrices (may be [])
        rho_final = states[-1] if states else None

    # Safety: ensure we actually have states (needed for outputs below)
    if not states or len(states) == 0:
        raise RuntimeError(
            "QuTiP did not return states. Ensure Options(store_states=True) is set "
            "and your QuTiP version supports it. (We set it above, so if this persists, "
            "please check your QuTiP install.)")

    # Output paths
    run_id = config.run_id or f"exact_N{config.model.N}_{config.model.boundary}_{random.randrange(10**8):08d}"
    outdir = os.path.join(config.outdir, run_id)
    os.makedirs(outdir, exist_ok=True)

    # trajectory save mode
    traj_mode = config.store_trajectory
    if traj_mode == "auto":
        traj_mode = "rho" if not closed else "ket"

    if traj_mode in ("ket", "rho"):
        if closed and traj_mode == "ket":
            psi_mat = np.vstack([st.full().reshape(-1) for st in states])   # (T, 2**N)
            np.savez(os.path.join(outdir, "trajectory_ket.npz"),
                     kets=psi_mat, times=ts)
        else:
            rho_stack = np.stack([st.full() for st in states], axis=0)      # (T, 2**N, 2**N)
            np.savez(os.path.join(outdir, "trajectory_rho.npz"),
                     rhos=rho_stack, times=ts)

    # final state save
    if config.store_state:
        if closed:
            psi_final = states[-1].full().reshape(-1)
            np.savez(os.path.join(outdir, "state_final.npz"), ket=psi_final)
        else:
            np.savez(os.path.join(outdir, "state_final.npz"), rho=rho_final.full())

    # =========================
    # CSV: build headers once
    # =========================
    headers = ["t"]
    headers.extend(obs_names)                          # expectations of observables
    if pauli_ops:
        headers.extend([f"exp_{s}" for s in pauli_ops.keys()])
    if config.entanglement_cut is not None:
        headers.append("SvN_cut" if closed else "logneg_cut")
    if prob_on and basis_bitstrings:
        for base in prob_bases:
            headers.extend([f"p{base}_{b}" for b in basis_bitstrings])

    # =========================
    # CSV: compute rows
    # =========================
    rows = []
    for k, t in enumerate(ts):
        row = [float(t)]

        # e_ops expectations (result.expect aligned with e_ops order)
        if e_ops:
            for idx in range(len(obs_names)):
                row.append(float(result.expect[idx][k]))

        # Pauli strings
        if pauli_ops:
            if closed:
                psi = states[k]
                for s, op in pauli_ops.items():
                    row.append(float(qt.expect(op, psi)))
            else:
                rho = states[k]
                for s, op in pauli_ops.items():
                    row.append(float(np.real((op * rho).tr())))

        # Entanglement
        if config.entanglement_cut is not None:
            cut = config.entanglement_cut
            if closed:
                psi = states[k]
                val = svn_bipartition_from_ket(psi, cut)
            else:
                rho = states[k]
                # ensure dims set
                rho.dims = [[2]*config.model.N, [2]*config.model.N]
                val = log_negativity_from_rho(rho, cut)
            row.append(float(val))

        # Probabilities in requested bases
        if prob_on and basis_bitstrings:
            if closed:
                psi_z = states[k]  # already in Z basis
                for base in prob_bases:
                    if base == "Z":
                        probs = np.abs(psi_z.full())**2
                        probs = probs.reshape(-1)
                    else:
                        psi_b = rotate_state_to_basis(psi_z, base, config.model.N)
                        probs = np.abs(psi_b.full())**2
                        probs = probs.reshape(-1)
                    row.extend([float(p) for p in probs])
            else:
                rho_z = states[k]
                rho_z.dims = [[2]*config.model.N, [2]*config.model.N]
                for base in prob_bases:
                    if base == "Z":
                        probs = np.real(np.diag(rho_z.full()))
                    else:
                        rho_b = rotate_state_to_basis(rho_z, base, config.model.N)
                        probs = np.real(np.diag(rho_b.full()))
                    row.extend([float(p) for p in probs])

        rows.append(row)

    # write CSV
    csv_path = os.path.join(outdir, "timeseries.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    # manifest
    manifest = {
        "run_id": run_id,
        "solver": "sesolve" if closed else "mesolve",
        "model": asdict(config.model),
        "init": asdict(config.init) | {"resolved_basis_string": bitstr0},
        "time": asdict(config.time),
        "noise": asdict(config.noise),
        "observables": want,
        "stored": {
            "trajectory": traj_mode,
            "probabilities": bool(prob_on),
            "prob_bases": prob_bases if prob_on else [],
            "pauli_strings": pauli_strings,
            "entanglement_cut": config.entanglement_cut,
            "final_state_npz": bool(config.store_state),
            },
        "realized_disorder_h": list(map(float, realized_h)),
        "csv_columns": headers,
        "versions": {
            "numpy": np.__version__,
            "qutip": qt.__version__,
            },
        "basis_convention": "|0>=|↑z>, |1>=|↓z>"
        }
    
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return {"outdir": outdir, "csv": csv_path}


# =========================
# Config loader & CLI
# =========================

def load_config(path: str) -> RunConfig:
    with open(path) as f:
        raw = json.load(f)

    def parse_dc(dc, d):
        return dc(**{k: v for k, v in d.items()})

    model = parse_dc(ModelConfig, raw.get("model", {}))
    model.disorder = parse_dc(DisorderConfig, raw.get("model", {}).get("disorder", {}))
    model.potential = parse_dc(PotentialConfig, raw.get("model", {}).get("potential", {}))

    init = parse_dc(InitStateConfig, raw.get("init", {}))
    time = parse_dc(TimeConfig, raw.get("time", {}))
    noise = parse_dc(NoiseConfig, raw.get("noise", {}))

    rc = RunConfig(
        model=model,
        init=init,
        time=time,
        noise=noise,
        store_probabilities=raw.get("store_probabilities", False),
        store_prob_bases=raw.get("store_prob_bases", ["Z"]),
        measure_pauli_strings=raw.get("measure_pauli_strings", []),
        entanglement_cut=raw.get("entanglement_cut", None),
        store_trajectory=raw.get("store_trajectory", "none"),
        store_state=raw.get("store_state", False),
        observables=raw.get("observables", ["energy", "magnetization_z", "sz_sites"]),
        random_seed=raw.get("random_seed", None),
        pbc_wrap=raw.get("pbc_wrap", False),
        outdir=raw.get("outdir", "experiments/results"),
        run_id=raw.get("run_id", None),
        )
    return rc

def main():
    '''ap = argparse.ArgumentParser(description="Exact small-chain evolution (QuTiP)")
    ap.add_argument("--config", type=str, required=True, help="Path to JSON config")
    ap.add_argument("--outdir", type=str, default=None, help="Override output directory")
    args = ap.parse_args()'''

    # ..experiments/configs/xx_closed.json
    rc = load_config('experiments/configs/xx_open.json')

    '''rc = load_config(args.config)
    if args.outdir:
        rc.outdir = args.outdir'''

    info = run(rc)
    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    main()
