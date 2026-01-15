import numpy as np
import pandas as pd
import os
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import label, binary_dilation
from src.metrics.trend_noise import calculate_trend, calculate_noise_from_trend
from src.metrics.spikes import detect_loop_spikes, detect_residual_spikes
from src.metrics.steps import piecewise_improvement
from src.metrics.step_slope import detect_slope_steps
from src.scoring import evaluate_metric, MetricStats

from src.utils import normalize_minmax, normalize_zscore  # Import helpers


def analyze_sensor(
    name,
    t,
    y,
    noise_window,
    spline_window,
    trend_method,
    poly_order,
    trend_iterations,
    apply_spline,
    spline_s,
    loop_window,
    loop_amp_thresh,
    baseline_window,
    baseline_tolerance,
    hampel_sigma,
    use_trend_anom,
    intersect_spikes,
    spike_proximity,  # Added missing args
    step_window,
    step_poly_order,
    thresh_piecewise,
    thresh_slope,
    intersect_steps,
    step_method,
    use_trend_step,
    normalization_method,  # New param
    deriv_noise_weight, # New param
    ref_points, # New param
    grading_params,
):
    """
    Runs the full analysis pipeline for a single sensor.
    Returns: (MetricStats object, and all intermediate series/masks for plotting)
    """
    y_vals = y.values

    # ... (Previous logic for Trend, Noise, SNR, Spikes, Steps)
    # (I will retain existing logic but insert Model Comparison block before Scoring)

    # 1. Trend & Noise
    method_map = {"Median": "median", "Savitzky-Golay": "savgol", "Spline": "spline"}
    smooth_trend = calculate_trend(
        y,
        window=spline_window,
        method=method_map.get(trend_method, "median"),
        poly_order=poly_order,
        iterations=trend_iterations,
        spline_s=spline_s,
    )
    # ... (Post-smoothing optional) ...
    if apply_spline and spline_s is not None:
        x = np.arange(len(smooth_trend))
        try:
            spl = UnivariateSpline(x, smooth_trend.values, s=spline_s)
            smooth_trend = pd.Series(spl(x), index=y.index)
        except:
            pass

    noise_series = calculate_noise_from_trend(y, smooth_trend, window=noise_window)

    trend_diff = pd.Series(np.gradient(smooth_trend), index=y.index)
    trend_diff_smooth = calculate_trend(
        trend_diff, window=spline_window, method="median"
    )
    noise_diff_series = calculate_noise_from_trend(
        trend_diff, trend_diff_smooth, window=noise_window
    )

    snr_series = smooth_trend / (noise_series + 1e-12)

    # 3. Spikes
    if use_trend_anom:
        # Detect on Trend vs Local Trend Baseline
        trend_baseline = (
            pd.Series(smooth_trend)
            .rolling(window=loop_window, center=True, min_periods=1)
            .median()
            .values
        )
        loop_mask = detect_loop_spikes(
            smooth_trend.values,
            noise_series.values,
            smooth=trend_baseline,
            window=loop_window,
            amp_threshold=loop_amp_thresh,
            baseline_window=baseline_window,
            baseline_tolerance=baseline_tolerance,
        )
    else:
        # Detect on Raw vs Trend
        loop_mask = detect_loop_spikes(
            y_vals,
            noise_series.values,
            smooth=smooth_trend.values,
            window=loop_window,
            amp_threshold=loop_amp_thresh,
            baseline_window=baseline_window,
            baseline_tolerance=baseline_tolerance,
        )

    # B. Hampel Spikes (Target for confirmation)
    # Use input based on toggle (Raw or Trend)
    y_anom_input = smooth_trend.values if use_trend_anom else y_vals

    h_win = max(3, int(spline_window))
    if h_win % 2 == 0:
        h_win -= 1

    # Calculate baseline relative to the selected input
    light_smooth = (
        pd.Series(y_anom_input)
        .rolling(window=h_win, center=True, min_periods=1)
        .median()
        .values
    )
    res_mask, residuals, res_thresh = detect_residual_spikes(
        y_anom_input, light_smooth, sigma_factor=hampel_sigma
    )

    # C. Combined Anomalies Logic (Intersection vs Union)

    # C. Combined Anomalies Logic (Intersection vs Union)
    if intersect_spikes:
        # Previous approach using binary dilation (kept for reference)
        # prox = max(0, int(spike_proximity))
        # structure = np.ones(2 * prox + 1, dtype=bool) if prox > 0 else np.array([True], dtype=bool)
        # labeled_loops, num_loop_features = label(loop_mask)
        # labeled_hampel, num_hampel_features = label(res_mask)
        # loop_overlap = loop_mask & binary_dilation(res_mask, structure=structure)
        # valid_loop_labels = np.unique(labeled_loops[loop_overlap])
        # valid_loop_labels = valid_loop_labels[valid_loop_labels != 0]
        # confirmed_loops = np.isin(labeled_loops, valid_loop_labels)
        # hampel_overlap = res_mask & binary_dilation(loop_mask, structure=structure)
        # valid_hampel_labels = np.unique(labeled_hampel[hampel_overlap])
        # valid_hampel_labels = valid_hampel_labels[valid_hampel_labels != 0]
        # confirmed_hampel = np.isin(labeled_hampel, valid_hampel_labels)
        # final_anomalies_mask = confirmed_loops | confirmed_hampel

        # New confirmation logic: keep original indices from both detectors when they agree within proximity.
        prox = max(0, int(spike_proximity))
        n_points = len(loop_mask)
        final_anomalies_mask = np.zeros(n_points, dtype=bool)

        labeled_loops, num_loop_features = label(loop_mask)
        labeled_hampel, num_hampel_features = label(res_mask)
        loop_idx = np.where(loop_mask)[0]
        hampel_idx = np.where(res_mask)[0]

        def iter_spans(labels, count):
            for feature_id in range(1, count + 1):
                idx = np.where(labels == feature_id)[0]
                if idx.size:
                    yield idx[0], idx[-1]

        # Confirm loop events by nearby Hampel points
        for start, end in iter_spans(labeled_loops, num_loop_features):
            s = max(0, start - prox)
            e = min(n_points - 1, end + prox)
            hits = hampel_idx[(hampel_idx >= s) & (hampel_idx <= e)]
            if hits.size:
                final_anomalies_mask[start : end + 1] = True
                final_anomalies_mask[hits] = True

        # Confirm Hampel events by nearby loop points
        for start, end in iter_spans(labeled_hampel, num_hampel_features):
            s = max(0, start - prox)
            e = min(n_points - 1, end + prox)
            hits = loop_idx[(loop_idx >= s) & (loop_idx <= e)]
            if hits.size:
                final_anomalies_mask[start : end + 1] = True
                final_anomalies_mask[hits] = True
    else:
        # Union
        final_anomalies_mask = loop_mask | res_mask

    # 4. Steps
    y_step_input = smooth_trend.values if use_trend_step else y_vals

    metric_piece = piecewise_improvement(
        t.values, y_step_input, window=step_window, poly_order=step_poly_order
    )
    mask_piece = np.zeros_like(y_vals, dtype=bool)
    if metric_piece is not None:
        mask_piece = np.nan_to_num(metric_piece, 0) > thresh_piecewise

    metric_slope, mask_slope = detect_slope_steps(
        y_step_input, window=step_window, slope_threshold=thresh_slope
    )

    if intersect_steps:
        step_mask = mask_piece & mask_slope
        step_metric = metric_piece
    else:
        if step_method == "Slope Pulse":
            step_mask = mask_slope
            step_metric = metric_slope
        else:
            step_mask = mask_piece
            step_metric = metric_piece

    # 5. Model Comparison (New)
    model_rmse = 0.0
    norm_y = y_vals  # Placeholder
    norm_ref = y_vals  # Placeholder
    model_error_msg = None

    try:
        # Load Reference
        ref_path = "src/metrics/Perfect_reference_curve.csv"

        if os.path.exists(ref_path):
            ref_df = pd.read_csv(ref_path)
            # Assume 1st column is Time, 2nd is Signal (or by name if known)
            # User said 'SM_R4C1:Coherent Mass Density' is the ref?
            # Or maybe just take the first data column.
            ref_time_col = [c for c in ref_df.columns if "time" in c.lower()][0]
            ref_val_col = [
                c for c in ref_df.columns if c != ref_time_col and "Unnamed" not in c
            ][0]

            ref_t = ref_df[ref_time_col].values
            ref_y = ref_df[ref_val_col].values

            # Interpolate Reference to Sensor Time
            # Use np.interp
            ref_y_aligned = np.interp(t.values, ref_t, ref_y)

            # Normalize
            if normalization_method == "Z-Score":
                norm_y = normalize_zscore(y_vals)
                norm_ref = normalize_zscore(ref_y_aligned)
            else:  # MinMax
                norm_y = normalize_minmax(y_vals)
                norm_ref = normalize_minmax(ref_y_aligned)

            # Calc RMSE
            model_rmse = np.sqrt(np.nanmean((norm_y - norm_ref) ** 2))
        else:
            model_error_msg = "Reference file 'Perfect_reference_curve.csv' not found."

    except Exception as e:
        model_error_msg = str(e)

    # 5b. Reference Points Calculation
    ref_points_rmse = 0.0
    if ref_points is not None:
         # If DataFrame, convert to records
        if isinstance(ref_points, pd.DataFrame):
            pts_list = ref_points.to_dict("records")
        else:
            pts_list = ref_points

        if pts_list:
            sq_errors = []
            for pt in pts_list:
                # Robust parsing for potentially partial string input
                raw_t = pt.get("Time")
                raw_v = pt.get("Value")
                
                if raw_t is None or raw_v is None:
                    continue

                try:
                    p_t = float(raw_t)
                    p_v = float(raw_v)
                except (ValueError, TypeError):
                    continue
                
                # Interpolate measured (Raw Smoothed Trend) value at p_t
                val_at_t = np.interp(p_t, t.values, smooth_trend.values)
                sq_errors.append((val_at_t - p_v) ** 2)
            
            if sq_errors:
                ref_points_rmse = np.sqrt(np.mean(sq_errors))

    # 6. Scoring
    effective_step_thresh = (
        thresh_slope if step_method == "Slope Pulse" else thresh_piecewise
    )

    # Convert Ratio-based grading params to Counts (per sensor length)
    local_grading = grading_params.copy()
    n_points = len(y_vals)
    
    if "limit_loop_ratio" in local_grading:
        ratio = local_grading.pop("limit_loop_ratio")
        # Ensure we don't accidentally send both if user configured weirdly, but usually app sends ratio
        local_grading["limit_anomalies_count"] = int(ratio * n_points)
    
    if "limit_step_ratio" in local_grading:
        ratio = local_grading.pop("limit_step_ratio")
        local_grading["limit_step_count"] = int(ratio * n_points)

    stats = evaluate_metric(
        name=name,
        values=y_vals,
        noise=noise_series.values,
        snr=snr_series.values,
        residuals=residuals,
        deriv_noise=noise_diff_series.values,
        anomalies_mask=final_anomalies_mask,  # Renamed arg, use final mask
        step_mask=step_mask,
        improvements=step_metric,
        step_threshold=effective_step_thresh,
        model_rmse=model_rmse,
        deriv_noise_weight=deriv_noise_weight,
        ref_points_rmse=ref_points_rmse,
        **local_grading,
    )
    return stats, {
        "noise_series": noise_series,
        "noise_diff_series": noise_diff_series,
        "smooth_trend": smooth_trend,
        "snr_series": snr_series,
        "loop_mask": loop_mask,
        "res_mask": res_mask,
        "residuals": residuals,
        "res_thresh": res_thresh,
        "step_metric": step_metric,
        "step_mask": step_mask,
        "mask_piece": mask_piece,
        "mask_slope": mask_slope,
        "metric_piece": metric_piece,
        "metric_slope": metric_slope,
        "norm_y": norm_y,  # Return for plotting
        "norm_ref": norm_ref,  # Return for plotting
        "model_error_msg": model_error_msg,
        "ref_points_rmse": ref_points_rmse,
    }


def run_global_analysis(df, time_col, sensor_cols, params, grading_params):
    """
    Iterates over all sensors and computes stats.
    Returns a DataFrame summary.
    """
    results = []
    t = df[time_col]

    for sensor in sensor_cols:
        # Basic cleanup/interpolation if needed (similar to example script)
        y = pd.to_numeric(df[sensor], errors="coerce")
        # Simple interpolate for small gaps
        y = y.interpolate(limit_direction="both")
        if y.isna().any():
            y = y.fillna(y.median())

        stats, _ = analyze_sensor(
            name=sensor, t=t, y=y, **params, grading_params=grading_params
        )
        results.append(stats)

    return results
