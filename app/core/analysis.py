from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from typing import Any

from app.model import als_baseline_correction  # reuse existing implementation
from app.model import load_assets as _load_assets

# Load assets once for analysis
try:
    _MODEL, ELEMENT_MAP, TARGET_WAVELENGTHS = _load_assets()
except (OSError, ValueError, RuntimeError, ImportError, FileNotFoundError):
    _MODEL, ELEMENT_MAP, TARGET_WAVELENGTHS = None, {}, np.array([], dtype=np.float32)


def run_full_analysis(input_data: dict[str, Any]) -> dict[str, Any]:
    if not input_data.get("asc_content"):
        raise ValueError("ASC content kosong.")

    # Parse ASC text
    rows = []
    for raw in input_data["asc_content"].splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "//", "%", ";")):
            continue
        line = line.replace(",", " ")
        parts = [p for p in line.split() if p]
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0]); y = float(parts[1])
            rows.append((x, y))
        except ValueError:
            continue
    if not rows:
        raise ValueError("Tidak menemukan pasangan data numerik (x y).")
    arr = np.asarray(rows, dtype=float)
    arr = arr[np.argsort(arr[:, 0])]
    wavelengths = arr[:, 0].astype(np.float64)
    intensities = arr[:, 1].astype(np.float64)

    # Wavelength shift
    shift_nm = float(input_data.get("shift_nm") or 0.0)
    if shift_nm != 0.0:
        wavelengths = wavelengths + shift_nm

    use_raw = bool(input_data.get("use_raw_resolution", True))
    if use_raw or TARGET_WAVELENGTHS.size == 0:
        used_wavelengths = wavelengths
        y = intensities.copy()
    else:
        used_wavelengths = TARGET_WAVELENGTHS.astype(np.float64)
        y = np.interp(used_wavelengths, wavelengths, intensities, left=0.0, right=0.0)

    baseline = None
    if input_data.get("apply_baseline_correction"):
        lam = float(input_data.get("lam") or 1e5)
        p = float(input_data.get("p") or 0.01)
        niter = int(input_data.get("niter") or 10)
        try:
            baseline = als_baseline_correction(y, lam=lam, p=p, niter=niter)
        except (ValueError, RuntimeError, TypeError):
            baseline = None

    spectrum_data = y.copy()
    if baseline is not None:
        spectrum_data = spectrum_data - baseline
        spectrum_data = np.clip(spectrum_data, a_min=0.0, a_max=None)

    # Optional smoothing
    if input_data.get("smoothing"):
        try:
            win = int(input_data.get("sg_window") or 11)
            poly = int(input_data.get("sg_poly") or 3)
            if win < 3:
                win = 3
            if win % 2 == 0:
                win += 1
            win = min(win, max(3, (len(spectrum_data) // 2) * 2 + 1))
            if poly >= win:
                poly = max(1, win - 2)
            spectrum_data = savgol_filter(spectrum_data, window_length=win, polyorder=poly)
        except (ValueError, RuntimeError):
            pass

    # Optional normalization
    norm = (input_data.get("normalization") or "None").lower()
    if norm == "max":
        m = float(np.max(spectrum_data)) if len(spectrum_data) else 0.0
        if m > 0:
            spectrum_data = spectrum_data / m
    elif norm == "area":
        area = float(np.trapz(spectrum_data, used_wavelengths)) if len(spectrum_data) else 0.0
        if abs(area) > 0:
            spectrum_data = spectrum_data / area

    prominence = float(input_data.get("prominence") or 0.01)
    distance = int(input_data.get("distance") or 8)
    height = input_data.get("height")
    width = input_data.get("width")
    threshold = float(input_data.get("threshold") or 0.6)
    peak_indices, _ = find_peaks(spectrum_data, prominence=prominence, distance=distance, height=height, width=width)
    peak_wavelengths = used_wavelengths[peak_indices]
    peak_intensities = spectrum_data[peak_indices]

    # Map peaks to elements using nearest index on TARGET_WAVELENGTHS
    all_peaks_list = []
    for i, wl in enumerate(peak_wavelengths):
        intensity = float(peak_intensities[i])
        if intensity < threshold:
            continue
        elements_here = []
        if TARGET_WAVELENGTHS.size and ELEMENT_MAP:
            try:
                nearest_idx = int(np.argmin(np.abs(TARGET_WAVELENGTHS - wl)))
            except (ValueError, TypeError):
                nearest_idx = None
            if nearest_idx is not None:
                for el, grid in ELEMENT_MAP.items():
                    try:
                        weight = float(grid[nearest_idx])
                    except (ValueError, TypeError, IndexError, KeyError):
                        weight = 0.0
                    if weight > 0.5:
                        elements_here.append(el)
        if elements_here:
            all_peaks_list.append({"wavelength": float(wl), "intensity": intensity, "elements": elements_here})

    sorted_peaks = sorted(all_peaks_list, key=lambda p: p['intensity'], reverse=True)

    annotations = []
    element_rank_counter = {}
    predicted_elements_with_locations = {}
    for peak in sorted_peaks:
        label_parts = []
        is_top_peak_overall = False
        for element in sorted(peak["elements"]):
            predicted_elements_with_locations.setdefault(element, []).append(peak["wavelength"])
            current_rank = element_rank_counter.get(element, 0) + 1
            element_rank_counter[element] = current_rank
            if current_rank <= 10:
                label_parts.append(f"{element} ({current_rank})")
                is_top_peak_overall = True
            else:
                label_parts.append(element)
        if label_parts:
            base_text = " ".join(label_parts)
            final_text = f"{base_text} {peak['wavelength']:.2f} nm"
            annotations.append({
                "x": peak["wavelength"], "y": peak["intensity"], "text": final_text,
                "is_top": is_top_peak_overall
            })

    prediction_table = []
    validation_table = []
    summary_metrics = {}
    predicted_elements_set = set(predicted_elements_with_locations.keys())
    for el in sorted(predicted_elements_set):
        locs = predicted_elements_with_locations.get(el, [])
        prediction_table.append({
            "Elemen": el,
            "Jumlah Puncak": len(locs),
            "Lokasi Puncak (nm)": '; '.join(map(lambda x: f"{x:.2f}", sorted(locs)))
        })

    if input_data.get("ground_truth_elements"):
        ground_truth_set = set(input_data["ground_truth_elements"])
        tp_set = predicted_elements_set & ground_truth_set
        fp_set = predicted_elements_set - ground_truth_set
        fn_set = ground_truth_set - predicted_elements_set
        tp = len(tp_set); fp = len(fp_set); fn = len(fn_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        summary_metrics = {"Precision": f"{precision:.2%}", "Recall": f"{recall:.2%}", "F1-Score": f"{f1_score:.2%}", "TP": tp, "FP": fp, "FN": fn}
        for el in sorted(tp_set):
            validation_table.append({"Elemen": el, "Status": "True Positive", "Jumlah Puncak": len(predicted_elements_with_locations.get(el, [])), "Lokasi (nm)": '; '.join(map(lambda x: f"{x:.2f}", sorted(predicted_elements_with_locations.get(el, []))))})
        for el in sorted(fp_set):
            validation_table.append({"Elemen": el, "Status": "False Positive", "Jumlah Puncak": len(predicted_elements_with_locations.get(el, [])), "Lokasi (nm)": '; '.join(map(lambda x: f"{x:.2f}", sorted(predicted_elements_with_locations.get(el, []))))})
        for el in sorted(fn_set):
            validation_table.append({"Elemen": el, "Status": "False Negative", "Jumlah Puncak": 0, "Lokasi (nm)": "-"})

    # Optional Abel inversion
    radial_profile = None
    radial_profile_error = None
    if input_data.get("compute_abel"):
        try:
            import importlib
            abel = importlib.import_module('abel')
            y_for_abel = spectrum_data.astype(float)
            if len(y_for_abel) % 2 == 1:
                y_for_abel = y_for_abel[:-1]
            rp = abel.basex.basex_transform(y_for_abel, direction='inverse')
            radial_profile = rp.tolist() if hasattr(rp, 'tolist') else np.asarray(rp).tolist()
        except (ImportError, ValueError, RuntimeError) as e:
            radial_profile_error = str(e)
            radial_profile = None

    return {
        "spectrum_data": spectrum_data,
        "wavelengths": used_wavelengths,
        "peak_wavelengths": np.array([p['wavelength'] for p in sorted_peaks], dtype=np.float32),
        "peak_intensities": np.array([p['intensity'] for p in sorted_peaks], dtype=np.float32),
        "annotations": annotations,
        "prediction_table": prediction_table,
        "validation_table": validation_table,
        "summary_metrics": summary_metrics,
        "baseline": baseline,
        "radial_profile": radial_profile,
        "radial_profile_error": radial_profile_error,
    }
