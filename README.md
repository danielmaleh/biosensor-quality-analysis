# The Daniel Test - Automated Sensor Data Quality Analysis

## Overview

This project is a Streamlit-based interactive dashboard designed for automated quality analysis of sensor (mologram) data from a predefined test. It provides comprehensive tools to evaluate chip and individual sensor performance, detect anomalies, and compare signals against reference models and user-defined criteria.

## Key Features & Improvements

### Global Chip Monitoring
- **Overall Chip Pass/Fail**: A consolidated verdict for the entire experiment (chip) based on configurable thresholds for:
  - **Global Pass Rate**: Percentage of individual sensors that pass all their criteria.
  - **Global Avg Score**: Average `Overall Score` across all sensors.
  - **Chip Uniformity**: Standard deviation of median signals across all sensors, indicating consistency.
- **Visual Global Analysis**: Distribution and trends of signals across all sensors, including aggregated mean and standard deviation plots, and value histograms.

### Enhanced Per-Sensor Metrics & Grading
- **Comprehensive Scoring**: Each sensor (mologram) is individually graded on 8 distinct metrics, contributing to a final `Overall Score` (Pass/Fail).
- **Noise Analysis**:
  - **Noise Ratio**: Normalized measure of amplitude noise, now with a configurable weight to include **Derivative Noise** (slope stability) as a penalty.
  - **Deriv Noise**: Quantifies the stability of the signal's rate of change, indicating signal "wiggliness".
- **Signal-to-Noise Ratio (SNR)**: Evaluates signal strength relative to noise, now considering both Median and 10th Percentile (P10) for robust assessment.
- **Anomalies (Spikes)**: Detection of Loop Spikes (transient deviations) and Hampel Spikes (outliers from local trend). User-defined ratio limits are converted to counts for evaluation.
- **Step Events**: Identification of abrupt signal transitions. Configurable methods (Piecewise Fit, Slope Pulse) and optional intersection logic for stricter detection. User-defined ratio limits are converted to counts.
- **Residual Spread**: Measures overall signal jitter around its smoothed trend.
- **Model RMSE**: Compares the sensor's normalized signal shape against a normalized "Perfect Reference Curve".
- **Ref Points RMSE**: Allows users to define specific (Time, Expected Raw Value) checkpoints. Calculates RMSE between the sensor's smoothed trend and these targets.

### User Experience & Data Management
- **Persistent Configuration**: All hyperparameters, grading rules, and reference points are saved and loaded from `persisted/user_config.json`.
- **Intuitive UI**:
  - Streamlined sidebar layout for data selection, hyperparameters, and grading rules.
  - Dedicated "Documentation" tab provides detailed explanations, formulas, and pass conditions for all metrics.
- **Experiment Metadata**:
  - Input fields in the sidebar for **Machine Software Version** and **Experiment Time (HH:MM)**, linked to individual experiment files.
  - Version displayed prominently on the main dashboard.
- **Datadog Integration**: Automatically generates Datadog log links based on filename timestamp and host. If an "Experiment Time" is provided, the log window is adjusted to +/- 1 hour around that time.
- **Sensor Filtering**: Automatically detects and analyzes only 'SM_R#C#' type sensor columns, ignoring other data (e.g., timestamp, current_injection).

## Installation & Usage

1.  **Prerequisites**: Python 3.10+

2.  **Install Dependencies**:

    ```bash
    uv sync
    ```

3.  **Run the Application**:

    ```bash
    streamlit run app.py
    ```

4.  **Package as Docker File**

    ```bash
    docker build --push -t registry.gitlab.com/linobiotechag/playground/the-daniel-test:1 .
    ```

## Project Structure

-   `app.py`: Main application entry point, Streamlit UI, global analytics.
-   `src/`: Core logic and computation modules.
    -   `metrics/`: Specific algorithms for noise, spikes, SNR, and step detection. Contains `Perfect_reference_curve.csv`.
    -   `analysis.py`: Orchestrates the per-sensor analysis pipeline, including data preparation and feature extraction.
    -   `scoring.py`: Handles metric weighting, individual sensor scoring, and pass/fail logic.
    -   `utils.py`: Utility functions for data loading, normalization, etc.
-   `persisted/`: Stores uploaded data (`Measurements/`), analysis outputs, and user configuration (`user_config.json`).