# The Daniel Test - Biosensor Quality Analysis Dashboard

A Streamlit-based interactive dashboard for automated quality analysis of biosensor (mologram) data. This tool provides comprehensive metrics for evaluating chip and individual sensor performance, detecting anomalies, and comparing signals against reference models.

## Overview

This quality analysis system is designed for biosensor measurement validation in production environments. It evaluates sensor signals across multiple dimensions including noise characteristics, signal-to-noise ratio, anomaly detection, step transitions, and model conformance.

## Key Features

### Global Chip Monitoring
- **Chip Pass/Fail Verdict**: Consolidated assessment based on configurable thresholds
- **Global Pass Rate**: Percentage of sensors passing all criteria
- **Chip Uniformity**: Standard deviation of median signals across sensors
- **Visual Analytics**: Distribution plots, trends, and heatmaps

### Per-Sensor Metrics (8 Criteria)

| Metric | Description | Pass Condition |
|--------|-------------|----------------|
| **Noise Ratio** | Normalized amplitude noise (MAD-based) | Ratio ≤ threshold |
| **Derivative Noise** | Rate of change stability | Value ≤ threshold |
| **SNR (Median/P10)** | Signal-to-noise ratio | Median ≥ 5.0, P10 ≥ 2.0 |
| **Anomalies** | Spike detection (Loop + Hampel) | Count ≤ limit |
| **Step Events** | Abrupt transition detection | Count ≤ limit |
| **Residual Spread** | Signal jitter (std of residuals) | Spread ≤ threshold |
| **Model RMSE** | Normalized shape comparison | RMSE ≤ 0.2 |
| **Ref Points RMSE** | User-defined checkpoints | RMSE ≤ threshold |

### Scoring System

```
Overall Score = Σ(wᵢ × Sᵢ)

Weights:
  - Anomalies:    20%
  - Model RMSE:   15%
  - Step Events:  15%
  - Noise/SNR:    10% each
  - Std/Deriv:    10% each
  - Ref Points:   10%

Pass Condition: Score ≥ 0.7 AND all individual metrics pass
```

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │ Data Select │  │ Config Panel│  │ Results Display │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                   Analysis Pipeline                      │
│  ┌─────────────────────────────────────────────────────┐│
│  │ src/analysis.py                                      ││
│  │  └─> analyze_sensor() → run_global_analysis()       ││
│  └─────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────┤
│                    Metrics Modules                       │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐          │
│  │ trend_noise│ │   spikes   │ │   steps    │          │
│  │ - Trend    │ │ - Loop     │ │ - Piecewise│          │
│  │ - MAD Noise│ │ - Hampel   │ │ - Slope    │          │
│  └────────────┘ └────────────┘ └────────────┘          │
├─────────────────────────────────────────────────────────┤
│                   Scoring Engine                         │
│  ┌─────────────────────────────────────────────────────┐│
│  │ src/scoring.py                                       ││
│  │  └─> evaluate_metric() → MetricStats                ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

## Metrics Documentation

### Noise Analysis
- **Rolling MAD**: Median Absolute Deviation over sliding window
- **Derivative Noise**: Stability of signal's rate of change
- **Formula**: `Noise Ratio = (Noise + W × DerivNoise) / Baseline`

### Spike Detection
- **Loop Spikes**: Deviations returning to baseline (single oscillation)
- **Hampel Spikes**: Points deviating > σ standard deviations from local trend
- **Intersection Mode**: Require both methods to agree for confirmation

### Step Detection
- **Piecewise Fit**: Score-based detection using polynomial fitting
- **Slope Pulse**: Derivative-based detection of abrupt changes
- **Intersection Mode**: Require both methods to confirm step events

### Model Comparison
- **Normalization**: MinMax [0,1] or Z-Score standardization
- **Reference Curve**: Comparison against ideal sensor response
- **RMSE Calculation**: Root Mean Square Error on normalized signals

## Installation

### Prerequisites
- Python 3.10+
- uv package manager (recommended)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd biosensor-quality-analysis

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

## Usage

### Run Locally

```bash
# Start the Streamlit application
streamlit run app.py
```

### Docker Deployment

```bash
# Build Docker image
docker build -t biosensor-quality-analysis .

# Run container
docker run -p 8501:8501 biosensor-quality-analysis

# Or use Docker Compose
docker compose up
```

### Data Requirements

Upload CSV files with:
- **Filename Convention**: `YYYY-MM-DD_HH-MM-SS_THE_DANIEL_TEST_HOSTNAME_...csv`
- **Columns**: Time column + sensor columns (pattern: `SM_R#C#`)
- **Format**: Numeric signal values over time

## Configuration

### Hyperparameters (Sidebar)

**Signal & Noise:**
- Smooth Window: Rolling window size for trend calculation
- Trend Method: Median or Savitzky-Golay
- Noise MAD Window: Window for noise calculation

**Spike Detection:**
- Loop Window: Window for loop spike detection
- Hampel Sigma: Standard deviation threshold
- Intersection Mode: Require confirmation from both methods

**Step Detection:**
- Step Window: Window for step detection
- Method: Piecewise Fit or Slope Pulse
- Threshold values for each method

### Grading Rules

All thresholds are configurable:
- Max Noise Ratio: 0.1
- Min SNR (Median): 5.0
- Min SNR (P10): 2.0
- Max Anomalies Ratio: 5%
- Max Step Events: 15%
- Max Residual Spread: 1.0e-8
- Max Derivative Noise: 1.0e-11
- Max Model RMSE: 0.2
- Min Overall Score: 0.7

## Project Structure

```
biosensor-quality-analysis/
├── app.py                      # Main Streamlit application
├── src/
│   ├── __init__.py
│   ├── analysis.py             # Analysis pipeline orchestration
│   ├── scoring.py              # Metric evaluation and scoring
│   ├── utils.py                # Data loading and normalization
│   └── metrics/
│       ├── __init__.py
│       ├── trend_noise.py      # Trend and noise calculations
│       ├── noise.py            # Noise metrics
│       ├── snr.py              # Signal-to-noise ratio
│       ├── spikes.py           # Spike detection algorithms
│       ├── steps.py            # Piecewise step detection
│       ├── step_slope.py       # Slope-based step detection
│       └── Perfect_reference_curve.csv
├── docs/
│   └── The Daniel Test.pdf     # Complete documentation
├── persisted/                  # Runtime data storage
│   ├── Measurements/           # Uploaded CSV files
│   ├── analysis_output/        # Generated plots
│   └── user_config.json        # Saved configuration
├── Dockerfile
├── compose.yml
├── deploy.sh
├── pyproject.toml
├── LICENSE
└── README.md
```

## Features

### Data Management
- **File Upload**: Drag-and-drop CSV upload
- **Persistent Config**: Settings saved to JSON
- **Experiment Log**: Track analysis history
- **Datadog Integration**: Auto-generate log links

### Visualization
- **Interactive Plots**: Plotly-based charts
- **Heatmaps**: Geometric sensor layout visualization
- **Distribution Charts**: Global signal analysis
- **Tab-based Navigation**: Organized metric views

### Export
- **Debug Plots**: Save analysis visualizations
- **PDF Documentation**: Built-in documentation viewer
- **Experiment Log**: CSV export of results

## Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scipy**: Scientific computing (signal processing)
- **plotly**: Interactive visualization
- **pymupdf**: PDF rendering

## Author

**Daniel Abraham Elmaleh**

Developed during internship at Lino Biotech

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
