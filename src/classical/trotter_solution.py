
import numpy as np
import json, os, csv, itertools, random
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Literal, Dict
import argparse

# ============
# Config
# ============

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
    slope: float = 0.0

@dataclass
class ModelConfig:
    N: int = 6
    J: float = 1.0
    Delta: float = 0.0
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

    store_probabilities: bool = False
    store_prob_bases: List[str] = None
    measure_pauli_strings: List[str] = None
    entanglement_cut: Optional[int] = None
    store_trajectory: Literal["none", "auto", "ket", "rho"] = "none"
    store_state: bool = False

    observables: List[str] = None
    random_seed: Optional[int] = None
    pbc_wrap: bool = False

    outdir: str = "experiments/results/classical/trotter_solution"
    run_id: Optional[str] = None

@dataclass
class TrotterConfig:
    order: int = 2               # Only 2 supported for now
    scheme: str = "even-odd"     # reserved for future variants
    dt: Optional[float] = None   # if None -> t_max/steps

# ============
# Linear algebra helpers
# ============

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
Sdg = np.array([[1, 0], [0, -1j]], dtype=complex)
HAD = (1/np.sqrt(2))*np.array([[1, 1],[1, -1]], dtype=complex)

def kronN(ops: List[np.ndarray]) -> np.ndarray:
    out = np.array([[1]], dtype=complex)
    for op in ops:
        out = np.kron(out, op)
    return out

def unitary_from_hermitian(H: np.ndarray, theta: float) -> np.ndarray:
    # U = exp(-i * theta * H)
    w, v = np.linalg.eigh(H)
    phase = np.exp(-1j * theta * w)
    return (v * phase) @ v.conj().T

# ============
# Model construction
# ============

def _disorder_field(model: ModelConfig) -> np.ndarray:
    N = model.N
    rng = np.random.default_rng(model.disorder.seed) if model.disorder.seed is not None else np.random.default_rng()
    if model.disorder.kind == "none":
        h = np.zeros(N, dtype=float)
    elif model.disorder.kind == "uniform":
        w = model.disorder.strength
        h = rng.uniform(-w, w, size=N)
    elif model.disorder.kind == "normal":
        h = rng.normal(0.0, model.disorder.strength, size=N)
    else:
        raise ValueError("Unknown disorder kind")
    if model.potential.type == "linear":
        F = model.potential.slope
        h = h + F * np.arange(N, dtype=float)
    return h

def bond_pairs_even_odd(N: int, boundary: str):
    even = [(i, i+1) for i in range(0, N-1, 2)]
    odd  = [(i, i+1) for i in range(1, N-1, 2)]
    if boundary == "PBC":
        if N % 2 == 0:
            odd.append((N-1, 0))
        else:
            even.append((N-1, 0))
    return even, odd

def two_site_ham(J: float, Delta: float) -> np.ndarray:
    # h = J (X\\otimes X + Y\\otimes Y) + Delta (Z\\otimes Z)
    return J*(np.kron(X, X) + np.kron(Y, Y)) + Delta*np.kron(Z, Z)

def on_site_unitary_z(theta: float) -> np.ndarray:
    # exp(-i theta Z) = diag(e^{-i theta}, e^{+i theta})
    return np.array([[np.exp(-1j*theta), 0],
                     [0, np.exp(1j*theta)]], dtype=complex)

def apply_two_site(U2: np.ndarray, psi: np.ndarray, i: int, N: int) -> np.ndarray:
    # psi is (2^N,), apply 4x4 U2 on qubits (i, i+1)
    psi_nd = psi.reshape([2]*N)
    # handle PBC wrap (N-1,0) specially
    if (i+1) % N == 0 and i == N-1:
        order = [N-1, 0] + [k for k in range(1, N-1)]
        psi_perm = np.transpose(psi_nd, order)
        psi_flat = psi_perm.reshape(4, -1)
        out_flat = U2 @ psi_flat
        out_perm = out_flat.reshape([2, 2] + [2]*(N-2))
        inv = np.argsort(order)
        psi_nd = np.transpose(out_perm, inv)
        return psi_nd.reshape(-1)
    else:
        order = [i, i+1] + [k for k in range(N) if k not in (i, i+1)]
        psi_perm = np.transpose(psi_nd, order)
        psi_flat = psi_perm.reshape(4, -1)
        out_flat = U2 @ psi_flat
        out_perm = out_flat.reshape([2, 2] + [2]*(N-2))
        inv = np.argsort(order)
        psi_nd = np.transpose(out_perm, inv)
        return psi_nd.reshape(-1)

def apply_one_site(U1: np.ndarray, psi: np.ndarray, i: int, N: int) -> np.ndarray:
    psi_nd = psi.reshape([2]*N)
    order = [i] + [k for k in range(N) if k != i]
    psi_perm = np.transpose(psi_nd, order).reshape(2, -1)
    out_flat = U1 @ psi_perm
    out_perm = out_flat.reshape([2] + [2]*(N-1))
    inv = np.argsort(order)
    psi_nd = np.transpose(out_perm, inv)
    return psi_nd.reshape(-1)

def build_dense_H(model: ModelConfig) -> np.ndarray:
    N, J, Delta = model.N, model.J, model.Delta
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)
    # bonds (non-wrapping)
    pairs = [(i, i+1) for i in range(N-1)]
    h2 = two_site_ham(J, Delta)
    for (i, j) in pairs:
        term = np.array([[1]], dtype=complex)
        site = 0
        while site < N:
            if site == i and j == i+1:
                term = np.kron(term, h2)
                site += 2
            else:
                term = np.kron(term, I2)
                site += 1
        H += term
    # PBC wrap term (N-1,0)
    if model.boundary == "PBC":
        if N > 2:
            perm = [N-1, 0] + list(range(1, N-1))
            inv = np.argsort(perm)
            H_wrap = np.zeros_like(H)
            for col in range(dim):
                e_col = np.zeros(dim, dtype=complex); e_col[col] = 1.0
                vec_nd = e_col.reshape([2]*N)
                vec_perm = np.transpose(vec_nd, perm).reshape(4, -1)
                out_perm = (h2 @ vec_perm).reshape([2,2] + [2]*(N-2))
                out_nd = np.transpose(out_perm, inv).reshape(dim)
                H_wrap += np.outer(out_nd, np.conj(e_col))
            H += H_wrap
        else:
            H += h2
    h = _disorder_field(model)
    for i in range(N):
        term = np.array([[1]], dtype=complex)
        for k in range(N):
            term = np.kron(term, Z if k == i else I2)
        H += h[i]*term
    return H

# ============
# Initial state
# ============

def build_initial_state(model: ModelConfig, init: InitStateConfig) -> (np.ndarray, str):
    N = model.N
    if init.kind == "neel":
        bitstr = ''.join('10'[(i % 2)] for i in range(N))  # "1010..."
    elif init.kind == "domain_wall":
        half = N // 2
        bitstr = '1'*half + '0'*(N-half)
    elif init.kind == "basis_string":
        if init.basis_string is None or len(init.basis_string) != N or any(c not in "01" for c in init.basis_string):
            raise ValueError("basis_string must be 0/1 string of length N")
        bitstr = init.basis_string
    elif init.kind == "product_random_z":
        rng = random.Random(init.seed)
        bitstr = ''.join(rng.choice('01') for _ in range(N))
    else:
        raise ValueError("Unknown initial state kind")
    basis = {'0': np.array([1.0, 0.0], dtype=complex), '1': np.array([0.0, 1.0], dtype=complex)}
    psi_nd = None
    for c in bitstr:
        psi_nd = basis[c] if psi_nd is None else np.kron(psi_nd, basis[c])
    return psi_nd.reshape(-1), bitstr

# ============
# Measurements
# ============

def pauli_string_op(spec: str, N: int) -> np.ndarray:
    # e.g., "X0Y1Y2", "Z0Z1", "X0"
    PAULI = {'X': X, 'Y': Y, 'Z': Z, 'I': I2}
    ops = [I2]*N
    i = 0
    L = len(spec)
    while i < L:
        p = spec[i].upper()
        if p not in PAULI:
            raise ValueError(f"Invalid Pauli '{p}' in {spec}")
        i += 1
        j = i
        while j < L and spec[j].isdigit():
            j += 1
        if j == i:
            raise ValueError(f"Missing index after {p} in {spec}")
        idx = int(spec[i:j])
        if not (0 <= idx < N):
            raise ValueError(f"Index {idx} out of range for N={N}")
        ops[idx] = PAULI[p]
        i = j
    return kronN(ops)

def build_observables(N: int, want: List[str], H: np.ndarray = None) -> Dict[str, np.ndarray]:
    obs = {}
    if "energy" in (want or []):
        if H is None:
            raise ValueError("Energy requested but Hamiltonian H not provided")
        obs["energy"] = H
    if "magnetization_z" in (want or []):
        # sum_i Z_i / N
        obs["magnetization_z"] = sum(
            np.kron(np.eye(2**i, dtype=complex), np.kron(Z, np.eye(2**(N-i-1), dtype=complex)))
            for i in range(N)
            ) / N
    if "sz_sites" in (want or []):
        for i in range(N):
            obs[f"sz_{i}"] = np.kron(
                np.eye(2**i, dtype=complex),
                np.kron(Z, np.eye(2**(N-i-1), dtype=complex))
                )
    return obs

def rotate_state_to_basis_numpy(psi: np.ndarray, base: str, N: int) -> np.ndarray:
    base = base.upper()
    if base == "Z":
        return psi
    elif base == "X":
        U1 = HAD
    elif base == "Y":
        U1 = Sdg @ HAD
    else:
        raise ValueError("Unknown basis; allowed: Z, X, Y")
    U = kronN([U1]*N)
    return U @ psi

def entanglement_svn_from_ket(psi: np.ndarray, N: int, cut: int) -> float:
    dL, dR = (1 << cut), (1 << (N - cut))
    M = psi.reshape(dL, dR)
    s = np.linalg.svd(M, compute_uv=False)
    p = (s**2)
    p = p[p > 1e-16]
    p = p / p.sum()
    return float(-np.sum(p * (np.log(p)/np.log(2))))

def load_config_with_trotter(path: str) -> [RunConfig, TrotterConfig, dict]:
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
        model=model, init=init, time=time, noise=noise,
        store_probabilities=raw.get("store_probabilities", False),
        store_prob_bases=raw.get("store_prob_bases", ["Z"]),
        measure_pauli_strings=raw.get("measure_pauli_strings", []),
        entanglement_cut=raw.get("entanglement_cut", None),
        store_trajectory=raw.get("store_trajectory", "none"),
        store_state=raw.get("store_state", False),
        observables=raw.get("observables", ["energy", "magnetization_z", "sz_sites"]),
        random_seed=raw.get("random_seed", None),
        pbc_wrap=raw.get("pbc_wrap", False),
        outdir=raw.get("outdir", "experiments/results/classical/trotter_solution"),
        run_id=raw.get("run_id", None),
        )
    tr = parse_dc(TrotterConfig, raw.get("trotter", {}))
    return rc, tr, raw

def expectation(psi: np.ndarray, op: np.ndarray) -> float:
    v = psi.conj().T @ (op @ psi)
    return float(np.real_if_close(v))

def run_trotter(config: RunConfig, trotter: TrotterConfig):
    if trotter.order != 2:
        raise NotImplementedError("Only 2nd-order (Strang) trotterization is implemented.")
    if config.noise.dephasing_rate > 0 or config.noise.relaxation_rate > 0:
        raise NotImplementedError("Open-system noise is currently not supported.")
    N = config.model.N
    steps = config.time.steps
    t_max = config.time.t_max
    if steps <= 0:
        raise ValueError("time.steps must be > 0")
    dt = trotter.dt if (trotter.dt is not None) else (t_max / steps)
    ts = np.linspace(0.0, t_max, steps + 1)

    # Hamiltonian blocks
    h_vec = _disorder_field(config.model)
    h2 = two_site_ham(config.model.J, config.model.Delta)
    U2_full = unitary_from_hermitian(h2, dt)
    U2_half = unitary_from_hermitian(h2, dt/2)
    U1_half = [on_site_unitary_z(h_i * dt/2) for h_i in h_vec]

    even_pairs, odd_pairs = bond_pairs_even_odd(N, config.model.boundary)

    # Initial state
    psi0, bitstr0 = build_initial_state(config.model, config.init)
    psi = psi0.copy()

    # Observables & Pauli ops for CSV
    want = config.observables or ["energy", "magnetization_z", "sz_sites"]
    H_dense = build_dense_H(config.model) if ("energy" in want) else None
    obs_map = build_observables(N, want, H=H_dense)
    obs_names = list(obs_map.keys())

    pauli_strings = list(config.measure_pauli_strings or [])
    pauli_ops = {s: pauli_string_op(s, N) for s in pauli_strings}

    prob_on = bool(config.store_probabilities)
    prob_bases = [b.upper() for b in (config.store_prob_bases or ["Z"])]
    for b in prob_bases:
        if b not in ("Z","X","Y"):
            raise ValueError("store_prob_bases must be subset of ['Z','X','Y']")
    basis_bitstrings = None
    if prob_on:
        basis_bitstrings = [''.join(seq) for seq in itertools.product('01', repeat=N)]

    # Outputs
    run_id = config.run_id or f"trotter2_N{N}_{config.model.boundary}_{random.randrange(10**8):08d}"
    outdir = os.path.join(config.outdir, run_id)
    os.makedirs(outdir, exist_ok=True)

    traj_mode = config.store_trajectory
    if traj_mode == "auto":
        traj_mode = "ket"

    traj_stack = []

    # CSV header
    headers = ["t"]
    headers.extend(obs_names)
    if pauli_ops:
        headers.extend([f"exp_{s}" for s in pauli_ops.keys()])
    if config.entanglement_cut is not None:
        headers.append("SvN_cut")
    if prob_on and basis_bitstrings:
        for base in prob_bases:
            headers.extend([f"p{base}_{b}" for b in basis_bitstrings])

    rows = []
    def append_row(t_now, psi_now):
        row = [float(t_now)]
        for name in obs_names:
            row.append(expectation(psi_now, obs_map[name]))
        for s, P in pauli_ops.items():
            row.append(expectation(psi_now, P))
        if config.entanglement_cut is not None:
            row.append(entanglement_svn_from_ket(psi_now, N, config.entanglement_cut))
        if prob_on and basis_bitstrings:
            for base in prob_bases:
                psi_b = rotate_state_to_basis_numpy(psi_now, base, N) if base != "Z" else psi_now
                probs = np.abs(psi_b)**2
                row.extend([float(p) for p in probs.reshape(-1)])
        rows.append(row)

    append_row(0.0, psi)

    for step in range(steps):
        # 1) Field half-step
        for i in range(N):
            psi = apply_one_site(U1_half[i], psi, i, N)
        # 2) Odd bonds half-step
        for (i, j) in odd_pairs:
            psi = apply_two_site(U2_half, psi, i, N)
        # 3) Even bonds full-step
        for (i, j) in even_pairs:
            psi = apply_two_site(U2_full, psi, i, N)
        # 4) Odd bonds half-step
        for (i, j) in odd_pairs:
            psi = apply_two_site(U2_half, psi, i, N)
        # 5) Field half-step
        for i in range(N):
            psi = apply_one_site(U1_half[i], psi, i, N)

        if traj_mode == "ket":
            traj_stack.append(psi.copy())

        append_row(ts[step+1], psi)

    # Write CSV
    csv_path = os.path.join(outdir, "timeseries.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    # Save trajectory
    if traj_mode == "ket" and len(traj_stack) > 0:
        ket_arr = np.stack(traj_stack, axis=0)  # shape (steps, 2^N)
        np.savez(os.path.join(outdir, "trajectory_ket.npz"), kets=ket_arr, times=ts[1:])

    # Final state save
    if config.store_state:
        np.savez(os.path.join(outdir, "state_final.npz"), ket=psi)

    # Manifest
    twoq_per_step = len(odd_pairs)*2 + len(even_pairs)*1
    oneq_per_step = 2 * N  # two field half-steps
    manifest = {
        "run_id": run_id,
        "solver": "trotter2",
        "backend": "dense",
        "model": asdict(config.model),
        "init": asdict(config.init),
        "time": asdict(config.time),
        "trotter": {
            "order": trotter.order,
            "dt": float(dt),
            "scheme": trotter.scheme,
            "bond_pairs_even": even_pairs,
            "bond_pairs_odd": odd_pairs,
            "gate_counts": {
                "two_qubit": steps * twoq_per_step,
                "one_qubit": steps * oneq_per_step
                }
                },
        "noise": asdict(config.noise),
        "store_probabilities": config.store_probabilities,
        "store_prob_bases": config.store_prob_bases or ["Z"],
        "measure_pauli_strings": config.measure_pauli_strings or [],
        "observables": config.observables or ["energy","magnetization_z","sz_sites"],
        "pauli_strings_computed": list(pauli_ops.keys()),
        "entanglement_cut": config.entanglement_cut,
        "N": N,
        "csv": csv_path
        }
    
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return {"outdir": outdir, "csv": csv_path, "manifest": os.path.join(outdir, "manifest.json")}

if __name__ == "__main__":
    '''ap = argparse.ArgumentParser(description="Dense 2nd-order Suzuki-Trotter evolution for XX/XXZ chains")
    ap.add_argument("--config", type=str, required=True, help="Path to JSON config (same schema as exact_solution; optional 'trotter' section)")
    ap.add_argument("--outdir", type=str, default=None, help="Override output directory")
    args = ap.parse_args()'''

    # experiments/configs/classical/trotter_solution/xx_closed_trotter.json
    rc, tr, raw = load_config_with_trotter("experiments/configs/classical/trotter_solution/xx_closed_trotter.json")

    '''rc, tr, raw = load_config_with_trotter(args.config)
    if args.outdir:
        rc.outdir = args.outdir'''

    info = run_trotter(rc, tr)
    print(f'Experiment ran successfully\\n{json.dumps(info, indent=2)}')
