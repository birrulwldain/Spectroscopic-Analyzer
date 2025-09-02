from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from typing import Any
import gc  # For memory management

from app.model import als_baseline_correction  # reuse existing implementation
from app.model import load_assets as _load_assets

# Load assets once for analysis - cached globally
try:
    _MODEL, ELEMENT_MAP, TARGET_WAVELENGTHS = _load_assets()
    print(f"Loaded model with {len(ELEMENT_MAP)} elements, {len(TARGET_WAVELENGTHS)} wavelength points")
except (OSError, ValueError, RuntimeError, ImportError, FileNotFoundError):
    _MODEL, ELEMENT_MAP, TARGET_WAVELENGTHS = None, {}, np.array([], dtype=np.float32)
    print("Warning: Model loading failed, using fallback mode")


def run_preprocessing(input_data: dict[str, Any]) -> dict[str, Any]:
    """Preprocessing only: parse ASC, shift wavelengths, resample if needed,
    baseline correction, smoothing, normalization.

    Returns at least: { 'wavelengths', 'spectrum_data' } and also 'baseline' for overlay.
    """
    global _MODEL
    
    if not input_data.get("asc_content"):
        raise ValueError("ASC content kosong.")

    # Parse ASC text
    rows: list[tuple[float, float]] = []
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

    # Resample (optional for preprocess mode, mandatory for predict/validate modes)
    analysis_mode = (input_data.get("analysis_mode") or "preprocess").lower()
    use_raw = bool(input_data.get("use_raw_resolution", True))
    
    # Force downsampling for prediction/validation modes to ensure model compatibility
    if analysis_mode in ["predict", "validate"] or (not use_raw and TARGET_WAVELENGTHS.size > 0):
        # Resample to target grid for ML model compatibility
        used_wavelengths = TARGET_WAVELENGTHS.astype(np.float64)
        left_fill = float(intensities[0]) if intensities.size else 0.0
        right_fill = float(intensities[-1]) if intensities.size else 0.0
        y = np.interp(used_wavelengths, wavelengths, intensities, left=left_fill, right=right_fill)
        print(f"Spektrum di-downsampling dari {len(wavelengths)} ke {len(used_wavelengths)} titik untuk mode {analysis_mode}")
    else:
        # Use original resolution for preprocessing mode only
        used_wavelengths = wavelengths
        y = intensities.copy()
        print(f"Menggunakan resolusi asli: {len(used_wavelengths)} titik untuk mode {analysis_mode}")

    # Baseline correction (optional)
    baseline = None
    baseline_applied = False
    baseline_warning = None
    if input_data.get("apply_baseline_correction"):
        lam = float(input_data.get("lam") or 1e5)
        p = float(input_data.get("p") or 0.01)
        niter = int(input_data.get("niter") or 10)
        try:
            baseline = als_baseline_correction(y, lam=lam, p=p, niter=niter)
        except (ValueError, RuntimeError, TypeError) as e:
            baseline = None
            baseline_warning = f"Baseline gagal: {e}"

    spectrum_data = y.copy()
    if baseline is not None:
        raw = spectrum_data.copy()
        corrected = raw - baseline
        corrected = np.clip(corrected, a_min=0.0, a_max=None)
        # If baseline removal wipes out nearly all signal, fall back to raw
        # (common when params are too aggressive or baseline ~= signal).
        try:
            raw_span = float(np.nanmax(raw) - np.nanmin(raw)) if raw.size else 0.0
            corr_span = float(np.nanmax(corrected) - np.nanmin(corrected)) if corrected.size else 0.0
        except ValueError:
            raw_span, corr_span = 0.0, 0.0
        if raw_span > 0 and corr_span < 1e-6 * raw_span:
            spectrum_data = raw
            baseline_applied = False
            baseline_warning = "Baseline diabaikan: hasil terlalu mendekati nol. Sesuaikan lam/p/niter."
        else:
            spectrum_data = corrected
            baseline_applied = True

    # Smoothing (optional)
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

    # Normalization (optional)
    norm = (input_data.get("normalization") or "None").lower()
    if norm == "max":
        m = float(np.nanmax(spectrum_data)) if len(spectrum_data) else 0.0
        if m > 0:
            spectrum_data = spectrum_data / m
    elif norm == "area":
        # Use positive area to avoid cancellation; guard tiny areas
        try:
            pos = np.clip(spectrum_data, a_min=0.0, a_max=None)
            area = float(np.trapz(pos, used_wavelengths)) if len(spectrum_data) else 0.0
        except (TypeError, ValueError, FloatingPointError):
            area = 0.0
        if abs(area) > 1e-12:
            spectrum_data = spectrum_data / area

    return {
        "wavelengths": used_wavelengths,
        "spectrum_data": spectrum_data,
        "baseline": baseline,
        "baseline_applied": baseline_applied,
        "baseline_warning": baseline_warning,
    }


def run_peak_analysis(input_data: dict[str, Any], preprocessed_results: dict[str, Any]) -> dict[str, Any]:
    """Peak detection, element mapping, annotations, validation, and optional Abel inversion.
    Accepts preprocessed wavelengths & spectrum.
    """
    used_wavelengths = preprocessed_results["wavelengths"]
    spectrum_data = preprocessed_results["spectrum_data"]
    baseline = preprocessed_results.get("baseline")

    prominence = float(input_data.get("prominence") or 0.01)
    distance = int(input_data.get("distance") or 8)
    height = input_data.get("height")
    width = input_data.get("width")
    threshold = float(input_data.get("threshold") or 0.1)
    peak_indices, _ = find_peaks(
        spectrum_data, prominence=prominence, distance=distance, height=height, width=width
    )
    
    # Optimasi: batasi jumlah puncak untuk performa
    max_peaks = 500  # Batasi maksimal 500 puncak
    if len(peak_indices) > max_peaks:
        # Ambil puncak tertinggi saja
        peak_heights = spectrum_data[peak_indices]
        top_indices = np.argsort(peak_heights)[-max_peaks:]
        peak_indices = peak_indices[top_indices]
        peak_indices = np.sort(peak_indices)  # Sort by wavelength
        print(f"DEBUG Optimized: Limited to top {max_peaks} peaks from {len(peak_indices)} total")
    
    # Debug: print total peaks found
    print(f"DEBUG Peak Detection: Found {len(peak_indices)} peaks across spectrum")
    # Debug: print summary info (limited untuk performa)
    if len(peak_indices) > 0:
        print(f"DEBUG Peak wavelengths range: {used_wavelengths[peak_indices[0]]:.2f} - {used_wavelengths[peak_indices[-1]]:.2f} nm")
        # Hanya print 5 puncak pertama dan terakhir untuk menghemat
        print(f"DEBUG First 5 peaks: {[f'{used_wavelengths[i]:.2f}' for i in peak_indices[:5]]}")
        if len(peak_indices) > 5:
            print(f"DEBUG Last 5 peaks: {[f'{used_wavelengths[i]:.2f}' for i in peak_indices[-5:]]}")
    print(f"DEBUG Parameters: prominence={prominence}, distance={distance}")
    
    peak_wavelengths = used_wavelengths[peak_indices]
    peak_intensities = spectrum_data[peak_indices]

    analysis_mode = (input_data.get("analysis_mode") or "predict").lower()
    if analysis_mode == "preprocess":
        return {
            "analysis_mode": analysis_mode,
            "spectrum_data": spectrum_data,
            "wavelengths": used_wavelengths,
            "peak_wavelengths": peak_wavelengths,
            "peak_intensities": peak_intensities,
            "annotations": [],
            "prediction_table": [],
            "validation_table": [],
            "summary_metrics": {},
            "baseline": baseline,
            "radial_profile": None,
            "radial_profile_error": None,
        }

    # Run ML model prediction on the downsampled spectrum
    all_peaks_list = []
    mapped_count = 0
    predictions = None
    
    # Early exit if no peaks to analyze
    if len(peak_wavelengths) == 0:
        print("DEBUG No peaks found for analysis")
        return {
            "analysis_mode": analysis_mode,
            "spectrum_data": spectrum_data,
            "wavelengths": used_wavelengths,
            "peak_wavelengths": np.array([], dtype=np.float32),
            "peak_intensities": np.array([], dtype=np.float32),
            "annotations": [],
            "prediction_table": [],
            "validation_table": [],
            "summary_metrics": {},
            "baseline": baseline,
            "radial_profile": None,
            "radial_profile_error": None,
        }
    
    if analysis_mode in ["predict", "validate"]:
        # Lazy load model untuk menghemat memory
        import torch
        try:
            global _MODEL
            # Load model hanya saat diperlukan
            if '_MODEL' not in globals() or _MODEL is None:
                print("DEBUG Loading INFORMER model...")
                from app.model import InformerModel, MODEL_CONFIG
                
                _MODEL = InformerModel(**MODEL_CONFIG)
                state_dict = torch.load("assets/informer_multilabel_model.pth", map_location='cpu', weights_only=False)
                _MODEL.load_state_dict(state_dict)
                _MODEL.eval()
                print("DEBUG Model loaded successfully")
            
            # Prepare input tensor: [batch_size, seq_length, input_dim]
            input_tensor = torch.from_numpy(spectrum_data[np.newaxis, :, np.newaxis]).float()
            
            with torch.no_grad():
                # Forward pass through INFORMER model (encoder only)
                output_logits = _MODEL(input_tensor)  # Shape: [1, 4096, 18]
                
                # Apply sigmoid for multi-label classification
                predictions = torch.sigmoid(output_logits).squeeze(0).cpu().numpy()  # Shape: [4096, 18]
            
            print(f"DEBUG Model Prediction: Generated predictions for {predictions.shape[0]} wavelength points, {predictions.shape[1]} elements")
            
            # Get element names from ELEMENT_MAP keys
            element_names = list(ELEMENT_MAP.keys()) if ELEMENT_MAP else [f"Element_{i}" for i in range(18)]
            
            # Get prediction threshold from input (default 0.7 if not provided)
            threshold_prob = float(input_data.get("prediction_threshold", 0.7))
            
            for i, wl in enumerate(peak_wavelengths):
                intensity = float(peak_intensities[i])
                if intensity < threshold:
                    continue
                    
                elements_here = []
                try:
                    # Find nearest wavelength index in the downsampled spectrum
                    nearest_idx = int(np.argmin(np.abs(used_wavelengths - wl)))
                    
                    # Get predictions at this wavelength point
                    pred_at_peak = predictions[nearest_idx]  # Shape: [18]
                    
                    # Find elements above threshold
                    detected_indices = np.where(pred_at_peak > threshold_prob)[0]
                    
                    for elem_idx in detected_indices:
                        if elem_idx < len(element_names):
                            element_name = element_names[elem_idx]
                            # Skip background element
                            if element_name.lower() == 'background':
                                continue
                            probability = pred_at_peak[elem_idx]
                            elements_here.append(element_name)
                            
                            # Debug: log predictions for first few peaks
                            if i < 10:
                                print(f"DEBUG Peak {i} at {wl:.2f}nm: {element_name} ({probability:.3f})")
                    
                except (ValueError, TypeError, IndexError) as e:
                    print(f"DEBUG Error mapping peak {i} at {wl:.2f}nm: {e}")
                    continue
                
                if elements_here:
                    all_peaks_list.append({
                        "wavelength": float(wl), 
                        "intensity": intensity, 
                        "elements": elements_here
                    })
                    mapped_count += 1
                    
        except Exception as e:
            print(f"DEBUG Model prediction failed: {e}")
            # Fallback to ELEMENT_MAP if model fails
            predictions = None
    
    # Fallback: Use ELEMENT_MAP for non-prediction modes or if model fails
    if predictions is None and ELEMENT_MAP:
        print("DEBUG Using ELEMENT_MAP fallback for element detection")
        
        # Calculate mapping from 4096 wavelengths to 18 element prediction points
        num_predictions = 18  # ELEMENT_MAP has 18 prediction points
        wavelength_range = TARGET_WAVELENGTHS[-1] - TARGET_WAVELENGTHS[0]  # 900 - 200 = 700nm
        prediction_step = wavelength_range / (num_predictions - 1)  # Step size for predictions
        
        for i, wl in enumerate(peak_wavelengths):
            intensity = float(peak_intensities[i])
            if intensity < threshold:
                continue
            elements_here = []
            try:
                # Find nearest wavelength index in TARGET_WAVELENGTHS
                nearest_wl_idx = int(np.argmin(np.abs(TARGET_WAVELENGTHS - wl)))
                target_wl = TARGET_WAVELENGTHS[nearest_wl_idx]
                
                # Map wavelength to prediction point index (0-17)
                prediction_idx = int(round((target_wl - TARGET_WAVELENGTHS[0]) / prediction_step))
                prediction_idx = max(0, min(17, prediction_idx))  # Clamp to valid range
                
                # Debug: log mapping details for first few peaks
                if i < 10:
                    print(f"DEBUG Fallback mapping peak {i}: {wl:.2f}nm -> wl_idx {nearest_wl_idx} -> pred_idx {prediction_idx} (target: {target_wl:.2f}nm)")
                    
                found_elements = []
                for el, grid in ELEMENT_MAP.items():
                    try:
                        # Skip background element
                        if el.lower() == 'background':
                            continue
                        weight = float(grid[prediction_idx])
                        if weight > 0.5:
                            elements_here.append(el)
                            found_elements.append(f"{el}({weight})")
                    except (ValueError, TypeError, IndexError, KeyError):
                        weight = 0.0
                        
                # Debug: log element mapping for first few peaks
                if i < 10:
                    print(f"DEBUG Fallback elements found at {wl:.2f}nm (pred_idx {prediction_idx}): {found_elements}")
                    
            except (ValueError, TypeError):
                continue
                
            if elements_here:
                all_peaks_list.append({"wavelength": float(wl), "intensity": intensity, "elements": elements_here})
                mapped_count += 1
    
    print(f"DEBUG Element Mapping: {mapped_count} peaks mapped out of {len(peak_wavelengths)} total peaks")

    sorted_peaks = sorted(all_peaks_list, key=lambda p: p['intensity'], reverse=True)

    annotations = []
    element_rank_counter: dict[str, int] = {}
    predicted_elements_with_locations: dict[str, list[float]] = {}
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
    summary_metrics: dict[str, Any] = {}
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
            # Clean up temporary arrays
            del y_for_abel, rp
        except (ImportError, ValueError, RuntimeError) as e:
            radial_profile_error = str(e)
            radial_profile = None

    # Clean up large temporary arrays and force garbage collection for memory efficiency
    if 'predictions' in locals():
        del predictions
    gc.collect()

    return {
        "analysis_mode": analysis_mode,
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
        # Add downsampled data for zoom plot accuracy
        "downsampled_wavelengths": used_wavelengths if analysis_mode in ["predict", "validate"] else None,
        "downsampled_intensities": spectrum_data if analysis_mode in ["predict", "validate"] else None,
    }


def run_full_analysis(input_data: dict[str, Any]) -> dict[str, Any]:
    """Orchestrates preprocessing and peak analysis."""
    pre = run_preprocessing(input_data)
    return run_peak_analysis(input_data, pre)
