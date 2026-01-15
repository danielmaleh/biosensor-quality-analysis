from dataclasses import dataclass
import numpy as np

@dataclass
class MetricStats:
    metric: str
    median_signal: float
    abs_median_signal: float
    signal_std: float
    noise_median: float
    noise_ratio: float
    deriv_noise_median: float 
    snr_median: float
    snr_p10: float
    anomalies_count: int 
    step_count: int
    model_rmse: float 
    ref_points_rmse: float 
    noise_pass: bool
    snr_pass: bool
    anomalies_pass: bool 
    steps_pass: bool
    std_pass: bool
    deriv_pass: bool 
    model_pass: bool 
    ref_points_pass: bool 
    overall_score: float
    overall_pass: bool

def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))

def evaluate_metric(
    name: str,
    values: np.ndarray,
    noise: np.ndarray,
    snr: np.ndarray,
    residuals: np.ndarray,
    deriv_noise: np.ndarray,
    anomalies_mask: np.ndarray, 
    step_mask: np.ndarray, 
    improvements: np.ndarray,
    step_threshold: float,
    model_rmse: float = 0.0,
    # Grading Thresholds
    limit_noise_ratio: float = 0.1,
    limit_snr_median: float = 5.0,
    limit_snr_p10: float = 2.0,
    limit_anomalies_count: int = 100,
    limit_step_count: int = 50,
    limit_signal_std: float = 1.0,
    limit_deriv_noise: float = 1.0e-11, 
    limit_model_rmse: float = 0.2, 
    limit_ref_points_rmse: float = 0.2, 
    limit_overall_score: float = 0.7,
    deriv_noise_weight: float = 0.0,
    ref_points_rmse: float = 0.0, 
    **kwargs
) -> MetricStats:
    
    median_signal = float(np.nanmedian(values))
    abs_median = float(np.nanmedian(np.abs(values)))
    
    signal_std = float(np.nanstd(residuals)) 
    
    baseline = abs_median if abs_median > 0 else (abs(median_signal) + 1e-12)
    
    noise_median = float(np.nanmedian(noise))
    if noise_median == 0:
        noise_median = float(np.nanmean(noise))

    deriv_noise_median = float(np.nanmedian(deriv_noise))
    
    # Combined Noise Ratio (Amplitude Noise + Weighted Derivative Noise)
    effective_noise = noise_median + (deriv_noise_weight * deriv_noise_median)
    noise_ratio = float(effective_noise / (baseline + 1e-12))
    
    snr_median = float(np.nanmedian(snr))
    snr_p10 = float(np.nanpercentile(snr, 10))
    
    # Anomalies Count (from mask passed in - Union or Intersection)
    anomalies_count = int(np.sum(anomalies_mask)) # Use the passed mask
    
    valid_impr = improvements[np.isfinite(improvements)]
    step_count = int(np.sum(step_mask))

    # Pass/Fail Criteria
    noise_pass = noise_ratio <= limit_noise_ratio
    snr_pass = snr_median >= limit_snr_median and snr_p10 >= limit_snr_p10
    anomalies_pass = anomalies_count <= limit_anomalies_count
    steps_pass = step_count <= limit_step_count
    std_pass = signal_std <= limit_signal_std
    deriv_pass = deriv_noise_median <= limit_deriv_noise
    model_pass = model_rmse <= limit_model_rmse 
    ref_points_pass = ref_points_rmse <= limit_ref_points_rmse 

    # Scoring (0.0 to 1.0)
    noise_score = clamp(1.0 - noise_ratio / (limit_noise_ratio * 2))
    snr_score = clamp((snr_median - 1.0) / (limit_snr_median + 4.0)) * clamp((snr_p10 - 1.0) / (limit_snr_p10 + 2.0))
    anomalies_score = clamp(1.0 - anomalies_count / (limit_anomalies_count * 2))
    step_score = clamp(1.0 - step_count / (limit_step_count * 2))
    std_score = clamp(1.0 - signal_std / (limit_signal_std * 1.5))
    deriv_score = clamp(1.0 - deriv_noise_median / (limit_deriv_noise * 2))
    model_score = clamp(1.0 - model_rmse / (limit_model_rmse * 2)) 
    ref_points_score = clamp(1.0 - ref_points_rmse / (limit_ref_points_rmse * 2)) 

    # Weighted Sum
    weights = {
        "noise": 0.10,
        "snr": 0.10,
        "anomalies": 0.20, 
        "steps": 0.15,
        "std": 0.10,
        "deriv": 0.10,
        "model": 0.15, 
        "ref_points": 0.10 
    }
    
    weighted_sum = (
        weights["noise"] * noise_score
        + weights["snr"] * snr_score
        + weights["anomalies"] * anomalies_score
        + weights["steps"] * step_score
        + weights["std"] * std_score
        + weights["deriv"] * deriv_score
        + weights["model"] * model_score
        + weights["ref_points"] * ref_points_score
    )
    
    total_weight = sum(weights.values())
    overall_score = float(weighted_sum / total_weight)
    overall_pass = (
        noise_pass and snr_pass and anomalies_pass 
        and steps_pass and std_pass and deriv_pass and model_pass
        and ref_points_pass
        and overall_score >= limit_overall_score
    )

    return MetricStats(
        metric=name,
        median_signal=median_signal,
        abs_median_signal=abs_median,
        signal_std=signal_std,
        noise_median=noise_median,
        noise_ratio=noise_ratio,
        deriv_noise_median=deriv_noise_median,
        snr_median=snr_median,
        snr_p10=snr_p10,
        anomalies_count=anomalies_count,
        step_count=step_count,
        model_rmse=model_rmse,
        ref_points_rmse=ref_points_rmse,
        noise_pass=noise_pass,
        snr_pass=snr_pass,
        anomalies_pass=anomalies_pass,
        steps_pass=steps_pass,
        std_pass=std_pass,
        deriv_pass=deriv_pass,
        model_pass=model_pass,
        ref_points_pass=ref_points_pass,
        overall_score=overall_score,
        overall_pass=overall_pass,
    )