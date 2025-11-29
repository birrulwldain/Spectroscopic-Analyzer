
# conv.py — Standalone simulator (no external project imports)
# Run: python src/conv.py  (set BASE_DIR below to your data folder)

import os, json, re
import numpy as np
import pandas as pd
import h5py
from scipy.signal.windows import gaussian
from scipy.signal import find_peaks
from typing import Dict, List, Tuple, Optional


# ===================== Inlined minimal classes =====================

BASE_ELEMENTS = [
    "Ac", "Ag", "Al", "Ar", "B", "Be", "Bi", "C", "Ca", "Cl", "Co", "Cr", "Cs", "Cu",
    "F", "Fe", "Ga", "Ge", "H", "He", "Hf", "Hg", "I", "In", "Ir", "K", "Li", "Mg",
    "Mn", "N", "Na", "Ne", "Ni", "O", "P", "Pd", "Pt", "Ra", "Rh", "Ru", "S", "Sb",
    "Sc", "Si", "Sn", "Sr", "Tc", "Te", "Th", "Ti", "V", "W", "Zn"
]
REQUIRED_ELEMENTS = [f"{elem}_{i}" for elem in BASE_ELEMENTS for i in [1, 2]]

class DataManager:
    """Lightweight paths + loaders (JSON one-hot map and ionization energies)."""
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.data_dir = base_dir
        self.nist_target_path = os.path.join(self.data_dir, "nist_data_hog.h5")
        self.atomic_data_target_path = os.path.join(self.data_dir, "atomic_data1.h5")
        self.json_map_path = os.path.join(self.data_dir, "element-map-2.json")

    def load_element_map(self) -> Dict[str, List[float]]:
        with open(self.json_map_path, 'r') as f:
            element_map = json.load(f)
        # Basic validation
        L = None
        for k, v in element_map.items():
            if not isinstance(v, list) or not all(isinstance(x, (int, float)) for x in v):
                raise ValueError(f"Invalid one-hot for {k}: {v}")
            if L is None: L = len(v)
            elif len(v) != L: raise ValueError("Inconsistent one-hot length")
        return element_map

    def load_ionization_energies(self) -> Dict[str, float]:
        ionization_energies = {}
        with h5py.File(self.atomic_data_target_path, 'r') as f:
            dset = f['elements']
            columns = dset.attrs.get('columns', ['At. num', 'Sp. Name', 'Ion Charge', 'El. Name',
                                                'Prefix', 'Ionization Energy (eV)', 'Suffix'])
            data = [[item[0], item[1].decode('utf-8'), item[2].decode('utf-8'), item[3].decode('utf-8'),
                    item[4].decode('utf-8'), item[5], item[6].decode('utf-8')] for item in dset[:]]
            df = pd.DataFrame(data, columns=columns)

            species_col = None
            ion_energy_col = None
            for col in columns:
                if col.lower() in ['sp.', 'species', 'sp', 'element', 'sp. name']:
                    species_col = col
                if 'ionization' in col.lower() and 'ev' in col.lower():
                    ion_energy_col = col
            if not species_col or not ion_energy_col:
                raise KeyError("Required columns not found in atomic_data1.h5")

            for _, row in df.iterrows():
                try:
                    ionization_energies[row[species_col]] = float(row[ion_energy_col])
                except Exception:
                    ionization_energies[row[species_col]] = 0.0

        # Ensure we have entries for I/II used later
        for elem in REQUIRED_ELEMENTS:
            base_elem, ion = elem.split('_')
            ion_level = 'I' if ion == '1' else 'II'
            sp_name = f"{base_elem} {ion_level}"
            ionization_energies.setdefault(sp_name, 0.0)
        return ionization_energies

class DataFetcher:
    """Read NIST lines from HDF5 store into arrays per (element, ion)."""
    def __init__(self, hdf_path: str, wl_range=(200, 900)):
        self.hdf_path = hdf_path
        self.wl_range = wl_range
        self.delta_E_max = {}

    def get_nist_data(self, element: str, sp_num: int):
        elem_key = f"{element}_{sp_num}"
        try:
            with pd.HDFStore(self.hdf_path, mode='r') as store:
                df = store.get('nist_spectroscopy_data')
                q = df[(df['element'] == element) & (df['sp_num'] == sp_num)]
                req = ['ritz_wl_air(nm)', 'Aki(s^-1)', 'Ek(eV)', 'Ei(eV)', 'g_i', 'g_k']
                if q.empty or not all(c in df.columns for c in req):
                    return [], 0.0
                q = q.dropna(subset=req)
                # Sanitize numeric columns
                q['ritz_wl_air(nm)'] = pd.to_numeric(q['ritz_wl_air(nm)'], errors='coerce')
                for col in ['Ek(eV)', 'Ei(eV)', 'Aki(s^-1)', 'g_i', 'g_k']:
                    q[col] = pd.to_numeric(q[col], errors='coerce')
                q = q.dropna(subset=['ritz_wl_air(nm)', 'Ek(eV)', 'Ei(eV)', 'Aki(s^-1)', 'g_i', 'g_k'])
                q = q[(q['ritz_wl_air(nm)'] >= self.wl_range[0]) & (q['ritz_wl_air(nm)'] <= self.wl_range[1])]
                q['delta_E'] = (q['Ek(eV)'] - q['Ei(eV)']).abs()
                if q.empty: return [], 0.0
                q = q.sort_values(by='Aki(s^-1)', ascending=False)
                dE = float(q['delta_E'].max()) if not pd.isna(q['delta_E'].max()) else 0.0
                self.delta_E_max[elem_key] = dE
                # Return numeric arrays (fast to use)
                return q[['ritz_wl_air(nm)', 'Aki(s^-1)', 'Ek(eV)', 'Ei(eV)', 'g_i', 'g_k']].values.tolist(), dE
        except Exception:
            return [], 0.0

class SpectrumSimulator:
    """Simulate emission spectrum from NIST-like transition table for one species."""
    def __init__(self, nist_data: List[List], element: str, ion: int, ionization_energy: float,
                 config: Dict, element_map_labels: Dict[str, List[float]]):
        self.nist_data = nist_data
        self.element = element
        self.ion = ion
        self.ionization_energy = ionization_energy
        self.resolution = config.get("resolution", 7000)
        self.wl_range = config.get("wl_range", (200, 900))
        self.sigma = config.get("sigma", 0.1)
        self.wavelengths = np.linspace(self.wl_range[0], self.wl_range[1], self.resolution, dtype=np.float32)
        self.element_label = f"{element}_{ion}"
        self.element_map_labels = element_map_labels

    def _partition_function(self, energy_levels: List[float], degeneracies: List[float], temperature: float) -> float:
        k_B = 8.617333262145e-5 # eV/K
        return sum(g * np.exp(-E / (k_B * temperature)) for g, E in zip(degeneracies, energy_levels) if E is not None) or 1.0

    def _calculate_intensity(self, temperature: float, energy: float, degeneracy: float, einstein_coeff: float, Z: float) -> float:
        k_B = 8.617333262145e-5 # eV/K
        return (degeneracy * einstein_coeff * np.exp(-energy / (k_B * temperature))) / Z

    def simulate_single_temp(self, temp: float, atom_percentage: float = 1.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self.nist_data: return None
        # Collect energy levels + degeneracies
        levels = {}
        for wl, Aki, Ek, Ei, gi, gk in self.nist_data:
            if all(v is not None for v in [Ek, Ei, gi, gk]):
                levels[float(Ei)] = float(gi)
                levels[float(Ek)] = float(gk)
        if not levels: return None
        energy_levels = list(levels.keys())
        degeneracies = list(levels.values())
        Z = self._partition_function(energy_levels, degeneracies, temp)

        intensities = np.zeros(self.resolution, dtype=np.float32)
        contrib = np.zeros(self.resolution, dtype=np.float32)

        wavelength_step = (self.wavelengths[-1] - self.wavelengths[0]) / (self.resolution - 1)
        sigma_points = self.sigma / wavelength_step

        for wl, Aki, Ek, _, _, gk in self.nist_data:
            wl = float(wl); Aki = float(Aki); Ek = float(Ek); gk = float(gk)
            I = self._calculate_intensity(temp, Ek, gk, Aki, Z)
            idx = np.searchsorted(self.wavelengths, wl)
            if 0 <= idx < self.resolution:
                # Gaussian kernel as PDF (area=1)
                kernel_size = int(6 * sigma_points) | 1
                half = kernel_size // 2
                kx = np.arange(-half, half+1)
                norm = 1.0 / (sigma_points * np.sqrt(2 * np.pi))
                kernel = norm * np.exp(-0.5 * (kx / sigma_points)**2)

                start = max(0, idx - half)
                end = min(self.resolution, idx + half + 1)
                ks = start - (idx - half)
                ke = ks + (end - start)
                intensities[start:end] += (I * atom_percentage * kernel[ks:ke]).astype(np.float32)
                contrib[start:end]     += (I * atom_percentage * kernel[ks:ke]).astype(np.float32)

        return intensities, contrib

class MixedSpectrumSimulator:
    """Mix spectra from multiple species and return final spectrum + per-wavelength labels (optional)."""
    def __init__(self, simulators: List[SpectrumSimulator], config: Dict,
                 delta_E_max: Dict[str, float], element_map_labels: Dict[str, List[float]]):
        self.simulators = simulators
        self.config = config
        self.resolution = config.get("resolution", 7000)
        self.wl_range = config.get("wl_range", (200, 900))
        self.convolution_sigma = config.get("convolution_sigma", 0.1)
        self.wavelengths = np.linspace(self.wl_range[0], self.wl_range[1], self.resolution, dtype=np.float32)
        self.intensity_threshold = config.get("intensity_threshold", 5e-4)
        self.element_map_labels = element_map_labels
        self.num_labels = len(next(iter(element_map_labels.values())))

    def _normalize_intensity(self, intensity: np.ndarray, target_max: float) -> np.ndarray:
        max_val = float(np.max(intensity))
        if max_val == 0: return intensity
        return (intensity / max_val * target_max).astype(np.float32)

    def _convolve_spectrum(self, spectrum: np.ndarray, sigma_nm: float) -> np.ndarray:
        wavelength_step = (self.wavelengths[-1] - self.wavelengths[0]) / (len(self.wavelengths) - 1)
        sigma_points = sigma_nm / wavelength_step
        ksize = int(6 * sigma_points) | 1
        kernel = gaussian(ksize, sigma_points)
        kernel /= kernel.sum()
        return np.convolve(spectrum, kernel, mode='same').astype(np.float32)

    def set_condition(self, T: float, n_e: float):
        self.current_T = T
        self.current_n_e = n_e

    def mix(self, atom_percentages: Dict[str, float], target_max_intensity: float = 0.8):
        """Return wavelengths, normalized mixed spectrum, labels (per-wl), and percentages used."""
        mixed = np.zeros(self.resolution, dtype=np.float32)
        contrib_all = []  # per-species contributions

        # simulate each active species
        active_sims = [s for s in self.simulators if s.element_label in atom_percentages]
        for sim in active_sims:
            pct = float(atom_percentages.get(sim.element_label, 0.0))
            if pct <= 0: continue
            out = sim.simulate_single_temp(self.current_T, pct)
            if out is None: continue
            spec, contrib = out
            mixed += spec
            contrib_all.append((sim.element_label, contrib))

        if float(np.max(mixed)) == 0.0:
            return self.wavelengths, mixed, None, atom_percentages

        convolved = self._convolve_spectrum(mixed, self.convolution_sigma)
        normalized = self._normalize_intensity(convolved, target_max_intensity)

        # Optional label tensor per wavelength
        if contrib_all:
            labels = np.zeros((self.resolution, self.num_labels), dtype=np.float32)
            for label_name, contrib in contrib_all:
                mask = contrib >= self.intensity_threshold
                vec = np.array(self.element_map_labels[label_name], dtype=np.float32)
                labels[mask] = np.maximum(labels[mask], vec)  # multi-hot set
        else:
            labels = None

        return self.wavelengths, normalized, labels, atom_percentages


# =============================== Main ===============================

if __name__ == "__main__":
    CONFIG = {
      "resolution": 7000, "wl_range": (200, 900),
      "sigma": 0.1, "convolution_sigma": 0.1,
      "intensity_threshold": 5e-4, "target_max_intensity": 0.8
    }
    BASE_DIR = "../data"  # GANTI sesuai lokasi file data Anda

    dm = DataManager(BASE_DIR)
    element_map = dm.load_element_map()
    ionization = dm.load_ionization_energies()

    fetch = DataFetcher(dm.nist_target_path, wl_range=CONFIG["wl_range"])
    # contoh: Ca I, Ca II, Mg I
    nist_data = {f"{e}_{i}": fetch.get_nist_data(e, i)[0] for e,i in [("Ca",1),("Ca",2),("Mg",1)]}

    # buat simulator per spesies
    sims = []
    for key,data in nist_data.items():
        if not data:
            continue
        e,i = key.split("_"); i=int(i)
        ion_suffix = "I" if i==1 else "II"
        sims.append(SpectrumSimulator(
            data, e, i, ionization.get(f"{e} {ion_suffix}", 0.0),
            CONFIG, element_map
        ))

    # campur spektrum
    mixer = MixedSpectrumSimulator(sims, CONFIG, delta_E_max={}, element_map_labels=element_map)
    mixer.set_condition(T=9000, n_e=1e16)
    wl, spec, labels, meta = mixer.mix({"Ca_1":30.0, "Ca_2":20.0, "Mg_1":5.0})

    # === Build per-component (convolved) spectra using the same kernel and a shared normalization ===
    # Recompute each species contribution (unconvolved), convolve, then scale all with the same factor
    # so their amplitudes are comparable with the 'spec' produced by the mixer.
    component_specs = {}  # key: sim.element_label -> dict with 'raw', 'conv', 'norm'
    active_specs = []
    for sim in sims:
        pct = float({"Ca_1":30.0, "Ca_2":20.0, "Mg_1":5.0}.get(sim.element_label, 0.0))
        if pct <= 0.0:
            continue
        out = sim.simulate_single_temp(9000, pct)
        if out is None:
            continue
        raw, _ = out  # raw (before instrument convolution)
        conv = mixer._convolve_spectrum(raw, CONFIG["convolution_sigma"])  # use same kernel as mixer
        component_specs[sim.element_label] = {"raw": raw, "conv": conv}
        active_specs.append(conv)

    # Sum of convolved components (should closely match internal 'spec' before normalization)
    if active_specs:
        mixed_convolved = np.sum(np.stack(active_specs, axis=0), axis=0).astype(np.float32)
        max_val = float(np.max(mixed_convolved)) if np.max(mixed_convolved) > 0 else 1.0
        scale = CONFIG["target_max_intensity"] / max_val
        mixture_norm = mixed_convolved * scale
        for k in component_specs:
            component_specs[k]["norm"] = component_specs[k]["conv"] * scale
    else:
        mixture_norm = spec  # fallback

    # Simpan hasil: campuran + komponen ter-normalisasi
    outdict = {"wavelength": wl, "mixture": mixture_norm.astype(np.float32)}
    for k, v in component_specs.items():
        outdict[f"comp_{k}"] = v["norm"].astype(np.float32)
    np.savez("mixture_and_components.npz", **outdict)

    # === Plot: mixture (black) + components (colored) ===
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        # small vertical offsets for clarity
        offsets = [0.10 * i for i in range(len(component_specs))]  # +0.1 vertical shift per layer
        # place mixture on top (offset above the highest component)
        mixture_offset = (offsets[-1] + 0.1) if offsets else 0.0
        colors = ["tab:red", "tab:green", "tab:orange", "tab:blue", "tab:purple"]
        labels_map = list(component_specs.keys())
        for i, k in enumerate(labels_map):
            y = component_specs[k]["norm"] + (0.0 if i >= len(offsets) else offsets[i])
            plt.plot(wl, y, label=f"{k} component", color=colors[i % len(colors)], linewidth=1.2)
        plt.plot(wl, mixture_norm + mixture_offset, label="Mixture (normalized)", color="black", linewidth=1.8, zorder=10)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Normalized Intensity (a.u.)")
        plt.xlim(380, 410)  # fokus sekitar Ca II; ubah sesuai kebutuhan
        plt.legend()
        plt.tight_layout()
        plt.savefig("components_demo.png", dpi=150)
        print("Saved mixture_and_components.npz and components_demo.png")
    except Exception as e:
        print("Plotting failed:", e)


    # === Deterministic tokenization (pure simulation, no noise): per-line area tokens ===
    # Each token corresponds to a NIST line actually used in the simulation.
    # Area per line is computed analytically since the Gaussian kernel is normalized to area=1.
    try:
        # Active composition used above
        comp_dict = {"Ca_1":30.0, "Ca_2":20.0, "Mg_1":5.0}
        T_used = 9000.0

        # Helper: discretize width bin relative to instrument sigma
        def width_bin_from_sigma(sigma_nm: float) -> str:
            r = sigma_nm / max(CONFIG["convolution_sigma"], 1e-9)
            if r < 0.8: return "W0"
            elif r < 1.2: return "W1"
            elif r < 1.6: return "W2"
            elif r < 2.0: return "W3"
            else: return "W4"

        # First pass: compute raw per-line areas for all active species
        raw_lines = []  # will hold dicts with elem, ion, lambda0, area
        for sim in sims:
            pct = float(comp_dict.get(sim.element_label, 0.0))
            if pct <= 0.0 or not sim.nist_data:
                continue

            # Build levels and partition function Z at T_used
            levels = {}
            for wl0, Aki, Ek, Ei, gi, gk in sim.nist_data:
                if all(v is not None for v in [Ek, Ei, gi, gk]):
                    levels[float(Ei)] = float(gi)
                    levels[float(Ek)] = float(gk)
            if not levels:
                continue
            energy_levels = list(levels.keys())
            degeneracies = list(levels.values())
            Z = sim._partition_function(energy_levels, degeneracies, T_used)

            ion_txt = "I" if sim.ion == 1 else "II" if sim.ion == 2 else str(sim.ion)

            # For each NIST line, the un-convolved kernel has area 1 → line area = I_line * pct
            for wl0, Aki, Ek, Ei, gi, gk in sim.nist_data:
                wl0 = float(wl0); Aki = float(Aki); Ek = float(Ek); gk = float(gk)
                I_line = sim._calculate_intensity(T_used, Ek, gk, Aki, Z)  # peak "mass" pre-convolution
                area = float(I_line * pct)
                raw_lines.append({
                    "elem": sim.element,
                    "ion": ion_txt,
                    "lambda0": wl0,
                    "area": area
                })

        # Optional filter: drop extremely small areas
        # Keep lines that contribute non-negligibly relative to the max area
        if not raw_lines:
            raise RuntimeError("No active lines found for deterministic tokenization.")
        max_area = max(r["area"] for r in raw_lines) or 1.0
        area_eps = max_area * 1e-6  # keep anything above 1e-6 of the strongest line (adjust as needed)
        filtered = [r for r in raw_lines if r["area"] >= area_eps]

        # Define area bins relative to max area (log-space)
        def area_bin(a: float) -> str:
            # a_norm in (0,1]; edges over 6 decades
            a_norm = max(a / max_area, 1e-12)
            lv = np.log10(a_norm)  # in [-inf, 0]
            edges = np.linspace(-6.0, 0.0, 8)  # 7 intervals → A0..A7
            b = int(np.searchsorted(edges, lv, side="right") - 1)
            b = int(np.clip(b, 0, 7))
            return f"A{b}"

        Wbin = width_bin_from_sigma(CONFIG["convolution_sigma"])

        # Build tokens (sorted by wavelength)
        filtered.sort(key=lambda r: r["lambda0"])
        tokens = []
        rows = []
        for r in filtered:
            abin = area_bin(r["area"])
            token = f"LINE|{r['elem']}|{r['ion']}|{r['lambda0']:.3f}|{abin}|{Wbin}|D0|C4"
            tokens.append(token)
            rows.append({
                "elem": r["elem"],
                "ion": r["ion"],
                "lambda0_nm": f"{r['lambda0']:.3f}",
                "area": f"{r['area']:.6e}",
                "A_bin": abin,
                "W_bin": Wbin,
                "shift": "D0",
                "conf": "C4"
            })

        # Save tokens.jsonl
        with open("tokens.jsonl", "w") as fw:
            rec = {
                "sample_id": "SIM_DEMO",
                "meta": {"T": T_used, "ne": 1e16, "sigma_instr": CONFIG['convolution_sigma']},
                "tokens": tokens
            }
            fw.write(json.dumps(rec) + "\n")

        # Save a human-readable CSV with areas
        import csv as _csv
        with open("tokens_area.csv", "w", newline="") as fcsv:
            writer = _csv.DictWriter(fcsv, fieldnames=["elem", "ion", "lambda0_nm", "area", "A_bin", "W_bin", "shift", "conf"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


        print(f"Deterministic tokenization done: {len(tokens)} tokens → tokens.jsonl, tokens_area.csv")

    except Exception as e:
        print("Deterministic tokenization failed:", e)


    # === Seq2Seq windowed export (event tokens) ===
    # Build per-window token sequences with padding & masks for autoregressive training.
    # Outputs:
    #  - vocab.json         : token -> id (with <PAD>=0, <BOS>=1, <EOS>=2)
    #  - seq_windows.npz    : y_in, y_out, loss_mask, pad_mask, window_starts_nm, window_ends_nm
    try:
        # Reuse tokens and rows built above
        # rows: list of dicts with lambda0_nm and token attributes
        # tokens: list of token strings "LINE|Elem|Ion|..."
        if not tokens:
            raise RuntimeError("No tokens to export for seq2seq.")

        # 1) Pair each token with its numeric lambda for windowing
        token_items = []
        for r, t in zip(rows, tokens):
            lam = float(r["lambda0_nm"])
            token_items.append((lam, t))
        token_items.sort(key=lambda x: x[0])  # sort by wavelength

        # 2) Define windows (fixed λ span)
        win_size = 20.0   # nm
        win_stride = 20.0 # nm (non-overlap). Set < win_size to create overlap if desired.
        wl_min, wl_max = float(CONFIG["wl_range"][0]), float(CONFIG["wl_range"][1])

        window_bounds = []
        s = wl_min
        while s < wl_max:
            e = min(s + win_size, wl_max)
            window_bounds.append((s, e))
            s += win_stride

        # 3) Collect token sequences per window
        windows_tokens = []
        windows_lams = []
        for (s,e) in window_bounds:
            seq = [t for lam, t in token_items if (lam >= s and lam < e)]
            lams = [lam for lam, t in token_items if (lam >= s and lam < e)]
            if len(seq) > 0:
                windows_tokens.append(seq)
                windows_lams.append(lams)

        if not windows_tokens:
            raise RuntimeError("No non-empty windows found for seq2seq export.")

        # 4) Build vocabulary (simple, per-file). Reserve special ids.
        specials = ["<PAD>", "<BOS>", "<EOS>"]
        vocab = {sp: i for i, sp in enumerate(specials)}  # <PAD>=0, <BOS>=1, <EOS>=2
        next_id = len(vocab)
        for seq in windows_tokens:
            for tok in seq:
                if tok not in vocab:
                    vocab[tok] = next_id
                    next_id += 1

        # 5) Encode sequences with padding and masks
        # y_in  : [<BOS>, tok1, ..., tokN, <EOS>, <PAD>...]
        # y_out : [tok1, ..., tokN, <EOS>, <PAD>...]
        max_len = max(len(seq) for seq in windows_tokens) + 2  # +2 for BOS/EOS
        num_win = len(windows_tokens)
        y_in = np.full((num_win, max_len), vocab["<PAD>"], dtype=np.int32)
        y_out = np.full((num_win, max_len), vocab["<PAD>"], dtype=np.int32)
        loss_mask = np.zeros((num_win, max_len), dtype=np.float32)  # 1 where loss is computed
        pad_mask = np.ones((num_win, max_len), dtype=np.float32)    # 1 for pad positions (for attention masks)

        for i, seq in enumerate(windows_tokens):
            # Encode
            enc = [vocab[t] for t in seq]
            # Fill y_in and y_out
            y_in[i, 0] = vocab["<BOS>"]
            y_in[i, 1:1+len(enc)] = enc
            y_in[i, 1+len(enc)] = vocab["<EOS>"]

            y_out[i, 0:len(enc)] = enc
            y_out[i, len(enc)] = vocab["<EOS>"]

            # Masks
            valid_len = len(enc) + 1  # include EOS
            loss_mask[i, 0:valid_len] = 1.0
            pad_mask[i, 0:valid_len+1] = 0.0  # mark non-pad (BOS..EOS) as 0; pads remain 1

        # 6) Save artifacts
        import json as _json
        with open("vocab.json", "w") as fv:
            _json.dump(vocab, fv, ensure_ascii=False, indent=2)

        window_starts = np.array([b[0] for b in window_bounds if any((lam >= b[0] and lam < b[1]) for lam,_ in token_items)], dtype=np.float32)
        window_ends   = np.array([b[1] for b in window_bounds if any((lam >= b[0] and lam < b[1]) for lam,_ in token_items)], dtype=np.float32)

        np.savez("seq_windows.npz",
                 y_in=y_in, y_out=y_out,
                 loss_mask=loss_mask, pad_mask=pad_mask,
                 window_starts_nm=window_starts, window_ends_nm=window_ends,
                 vocab_size=np.int32(len(vocab)))

        print(f"Seq2Seq windows exported: {num_win} windows, max_len={max_len}, vocab={len(vocab)} → seq_windows.npz, vocab.json")

    except Exception as e:
        print("Seq2Seq windowed export failed:", e)
