import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import glob
import base64
import re
import datetime
import json
from dataclasses import asdict
from scipy.ndimage import binary_dilation, label

from src.utils import load_data
from src.analysis import analyze_sensor, run_global_analysis
from src.scoring import evaluate_metric


def parse_filename(filepath):
    """Parse experiment filename into display-friendly components."""
    basename = os.path.basename(filepath)
    name_no_ext = os.path.splitext(basename)[0]

    result = {
        "date": "",
        "test_name": "",
        "reader": "",
        "display_name": basename,
        "suffix": "",
    }

    # Extract date
    date_match = re.match(r"^(\d{4}-\d{2}-\d{2})", name_no_ext)
    if date_match:
        result["date"] = date_match.group(1)

    # Extract test name and reader
    # Reader pattern: 2-4 uppercase letters + 2-3 digits, optionally -N (e.g. SIR06, NOV01, SIR06-3)
    reader_pattern = r"([A-Z]{2,4}\d{2,3}(?:-\d+)?)"

    # Try: DATE_TESTNAME_READER_description
    full_match = re.match(
        r"^(\d{4}-\d{2}-\d{2})_((?:THE_DANIEL_TEST|DE_GENERAL_TEST)(?:-\d+|_\d+)?)_"
        + reader_pattern
        + r"[_\s]",
        name_no_ext,
    )

    if full_match:
        raw_test = full_match.group(2)
        result["test_name"] = raw_test.replace("_", " ").title()
        result["reader"] = full_match.group(3)
    else:
        # Fallback: DATE_TESTNAME_description (no separate reader)
        fallback_match = re.match(
            r"^(\d{4}-\d{2}-\d{2})_((?:THE_DANIEL_TEST|DE_GENERAL_TEST)(?:-\d+|_\d+)?)[_\s]",
            name_no_ext,
        )
        if fallback_match:
            raw_test = fallback_match.group(2)
            result["test_name"] = raw_test.replace("_", " ").title()

    # Detect duplicate suffix like "(1)"
    suffix_match = re.search(r"\((\d+)\)\s*$", basename)
    if suffix_match:
        result["suffix"] = f"({suffix_match.group(1)})"

    # Build compact display name
    parts = []
    if result["date"]:
        parts.append(result["date"])
    if result["reader"]:
        parts.append(result["reader"])
    if result["test_name"]:
        parts.append(result["test_name"])
    elif not result["reader"]:
        parts.append(name_no_ext[:40])
    if result["suffix"]:
        parts.append(result["suffix"])

    result["display_name"] = " | ".join(parts) if parts else basename

    return result


# --- CONFIG ---
OUTPUT_DIR = "persisted/analysis_output"
CONFIG_FILE = "persisted/user_config.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                # Deserialize DataFrame
                if "ref_points" in data:
                    data["ref_points"] = pd.DataFrame(data["ref_points"]).astype(str)
                # Load Metadata
                if "experiment_metadata" in data:
                    st.session_state["experiment_metadata"] = data[
                        "experiment_metadata"
                    ]
                return data
        except:
            pass
    return {}


def save_config_to_file():
    # Keys to save
    keys = [
        "noise_window",
        "spline_window",
        "trend_method",
        "poly_order",
        "trend_iterations",
        "apply_spline",
        "spline_s",
        "vis_noise_low",
        "vis_noise_high",
        "vis_dnoise_low",
        "vis_dnoise_high",
        "loop_window",
        "loop_amp_thresh",
        "baseline_window",
        "baseline_tolerance",
        "hampel_sigma",
        "intersect_spikes",
        "spike_proximity",
        "step_method",
        "step_window",
        "step_poly_order",
        "thresh_piecewise",
        "thresh_slope",
        "intersect_steps",
        "limit_noise_ratio",
        "limit_snr_median",
        "limit_snr_p10",
        "limit_loop_ratio",
        "limit_residual_ratio",
        "limit_step_ratio",
        "limit_signal_std",
        "limit_deriv_noise",
        "limit_model_rmse",
        "limit_ref_points_rmse",
        "limit_overall_score",
        "limit_global_pass_rate",
        "limit_global_avg_score",
        "limit_chip_spread",
        "normalization_method",
        "deriv_noise_weight",
    ]
    data = {k: st.session_state[k] for k in keys if k in st.session_state}

    if "ref_points" in st.session_state:
        data["ref_points"] = st.session_state["ref_points"]

    if "experiment_metadata" in st.session_state:
        data["experiment_metadata"] = st.session_state["experiment_metadata"]

    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=4)
    st.sidebar.success("Settings Saved!")


# --- LOGGING ---
LOG_FILE = "persisted/experiment_log.csv"


def save_experiment_log(
    stats_df, filename, chip_spread, chip_pass, sw_version="not specified"
):
    # Prepare Record
    log_df = stats_df.copy()
    log_df["Filename"] = filename
    log_df["Timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_df["Chip Spread"] = chip_spread
    log_df["Chip Pass"] = chip_pass
    log_df["Machine SW"] = sw_version
    log_df.rename(columns={"metric": "Sensor"}, inplace=True)

    # Load existing
    if os.path.exists(LOG_FILE):
        try:
            existing_log = pd.read_csv(LOG_FILE)
            # Remove entries for this file to update them (Replace Logic)
            existing_log = existing_log[existing_log["Filename"] != filename]
            final_log = pd.concat([existing_log, log_df], ignore_index=True)
        except:
            final_log = log_df
    else:
        final_log = log_df

    final_log.to_csv(LOG_FILE, index=False)
    st.toast(f"Saved {len(log_df)} sensors to Experiment Log!", icon="💾")


@st.dialog("Experiment Log")
def view_experiment_log():
    if not os.path.exists(LOG_FILE):
        st.warning("No log file found.")
        return

    df_log = pd.read_csv(LOG_FILE)

    # Group by Experiment (Filename)
    unique_experiments = df_log["Filename"].unique()
    if len(unique_experiments) > 0:
        selected_experiment = st.selectbox(
            "Select Experiment", unique_experiments, index=0
        )

        # Filter view to this experiment
        filtered_view = df_log[df_log["Filename"] == selected_experiment]

        st.write(f"Sensors in **{selected_experiment}**:")
        selection = st.dataframe(
            filtered_view,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        if selection.selection.rows:
            idx = selection.selection.rows[0]
            row = filtered_view.iloc[idx]  # Use filtered view for row lookup

            sel_file = row["Filename"]
            sel_sensor = row["Sensor"]

            # Verify file exists locally
            full_path = f"./persisted/Measurements/{sel_file}"
            if os.path.exists(full_path):
                st.session_state["load_file_req"] = full_path
                st.session_state["load_sensor_req"] = sel_sensor
                st.session_state["selected_experiment_path"] = full_path
                st.session_state["selected_sensor_name"] = sel_sensor
                st.success(f"Ready to load: {sel_sensor}")
                if st.button("Go to Analysis"):
                    st.rerun()
            else:
                st.error(f"File not found in Measurements: {sel_file}")
    else:
        st.info("Log file is empty.")


@st.dialog("Select Experiment", width="large")
def select_experiment_dialog(csv_files):
    st.write("Choose an experiment to analyze.")

    rows = []
    for f in csv_files:
        parsed = parse_filename(f)
        rows.append(
            {
                "Date": parsed["date"],
                "Reader": parsed["reader"],
                "Test": parsed["test_name"],
                "Variant": parsed["suffix"],
                "_path": f,
            }
        )

    display_df = pd.DataFrame(rows)

    selection = st.dataframe(
        display_df[["Date", "Reader", "Test", "Variant"]],
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    if selection.selection.rows:
        idx = selection.selection.rows[0]
        chosen = rows[idx]
        st.success(
            f"Selected: {chosen['Date']} | {chosen['Reader'] or chosen['Test']}"
        )
        if st.button("Load Experiment", type="primary", use_container_width=True):
            st.session_state["selected_experiment_path"] = chosen["_path"]
            st.session_state["load_file_req"] = chosen["_path"]
            st.rerun()


@st.dialog("Select Sensor", width="large")
def select_sensor_dialog(sensor_cols, summary_df):
    st.write("Select a sensor for detailed analysis.")

    # Parse grid coordinates
    grid_info = {}
    for col_name in sensor_cols:
        match = re.search(r"R(\d+)C(\d+)", col_name)
        if match:
            r, c = int(match.group(1)), int(match.group(2))
            grid_info[(r, c)] = col_name

    if not grid_info:
        # Fallback: simple list if no grid pattern
        chosen = st.selectbox("Select Sensor", sensor_cols)
        if st.button("Select", type="primary"):
            st.session_state["selected_sensor_name"] = chosen
            st.rerun()
        return

    # Build pass/fail + score lookup
    status_map = {}
    score_map = {}
    for _, row in summary_df.iterrows():
        status_map[row["metric"]] = row["overall_pass"]
        score_map[row["metric"]] = row["overall_score"]

    all_rows = sorted(set(r for r, c in grid_info.keys()))
    all_cols = sorted(set(c for r, c in grid_info.keys()))

    current_sensor = st.session_state.get("selected_sensor_name", "")

    # Render grid
    header_cols = st.columns([0.3] + [1] * len(all_cols))
    header_cols[0].write("")
    for i, c in enumerate(all_cols):
        header_cols[i + 1].caption(f"**C{c}**")

    for r in all_rows:
        row_cols = st.columns([0.3] + [1] * len(all_cols))
        row_cols[0].caption(f"**R{r}**")
        for i, c in enumerate(all_cols):
            sensor_name = grid_info.get((r, c))
            if sensor_name:
                passed = status_map.get(sensor_name, True)
                score = score_map.get(sensor_name, 0)
                icon = "+" if passed else "X"
                label = f"{icon} R{r}C{c}\n{score:.2f}"
                is_current = sensor_name == current_sensor

                if row_cols[i + 1].button(
                    label,
                    key=f"sensor_btn_{r}_{c}",
                    use_container_width=True,
                    type="primary" if is_current else "secondary",
                ):
                    st.session_state["selected_sensor_name"] = sensor_name
                    st.rerun()


defaults = load_config()
# Initialize Session State with defaults if not already set
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.set_page_config(page_title="THE DANIEL TEST", layout="wide")

st.title("THE DANIEL TEST")

# --- NAVIGATION ---
page = st.sidebar.radio("Navigation", ["Analysis Dashboard", "Experiment Description"])

# --- SIDEBAR: DATA SELECTION (Always Visible) ---
st.sidebar.header("Data Selection")

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload New Experiment (CSV)", type=["csv"])

if uploaded_file is not None:
    fname = uploaded_file.name

    # Check if this file was just processed to avoid infinite loop
    if (
        "last_uploaded_file" not in st.session_state
        or st.session_state["last_uploaded_file"] != fname
    ):
        # Validate Naming Convention
        # 1. Starts with Date (YYYY-MM-DD)
        if not re.match(r"\d{4}-\d{2}-\d{2}", fname):
            st.sidebar.warning(
                "⚠️ Note: Filename usually starts with Date (YYYY-MM-DD)."
            )
        # 2. Contains Project Identifier
        elif not re.search(r"_(THE_DANIEL_TEST|DE_GENERAL_TEST-[0-9]+)_", fname):
            st.sidebar.warning(
                "⚠️ Note: Filename usually contains '_THE_DANIEL_TEST_' or '_DE_GENERAL_TEST-X_'."
            )

        # Sanitize filename
        fname = os.path.basename(uploaded_file.name)

        save_path = os.path.join("./persisted/Measurements", fname)

        try:
            os.makedirs(
                os.path.dirname(save_path), exist_ok=True
            )  # Ensure directory exists
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        except Exception as e:
            st.error(f"Error saving file to {save_path}: {e}")
            st.stop()

        st.session_state["last_uploaded_file"] = fname
        # Auto-select the newly uploaded file
        st.session_state["load_file_req"] = save_path
        st.session_state["selected_experiment_path"] = save_path

        st.sidebar.success(f"✅ Saved: {fname}")
        if page == "Experiment Description":
            st.sidebar.info(
                "👉 Switch to 'Analysis Dashboard' to view/analyze this file."
            )
        st.rerun()
else:
    # Reset tracking if uploader is cleared so the same file can be re-uploaded if needed
    if "last_uploaded_file" in st.session_state:
        del st.session_state["last_uploaded_file"]

csv_files = [
    f for f in glob.glob("./persisted/Measurements/*") if f.lower().endswith(".csv")
]
if not csv_files:
    st.error("No CSV files found in 'persisted/Measurements' folder.")
    # If no files, we can't do much, but we still might want to show Description
    if page != "Experiment Description":
        st.stop()
else:
    # Initialize selected experiment from load_file_req or default
    if "load_file_req" in st.session_state:
        req_path = st.session_state["load_file_req"]
        if req_path in csv_files:
            st.session_state["selected_experiment_path"] = req_path

    if (
        "selected_experiment_path" not in st.session_state
        or st.session_state["selected_experiment_path"] not in csv_files
    ):
        st.session_state["selected_experiment_path"] = csv_files[0]

    selected_file = st.session_state["selected_experiment_path"]

    # 1. Placeholders for inputs (appear above file selector)
    meta_placeholder = st.sidebar.container()

    # 2. File Selector (compact display + dialog button)
    current_parsed = parse_filename(selected_file)
    st.sidebar.markdown(f"**Experiment:** {current_parsed['display_name']}")

    if st.sidebar.button("Select Experiment", use_container_width=True):
        select_experiment_dialog(csv_files)

    # 3. Handle File Change Logic
    if "last_selected_file_meta" not in st.session_state:
        st.session_state["last_selected_file_meta"] = None

    if st.session_state["last_selected_file_meta"] != selected_file:
        # File Changed! Update inputs to match new file's metadata
        st.session_state["last_selected_file_meta"] = selected_file

        # Get saved meta for this new file
        new_fname = os.path.basename(selected_file)
        new_meta = st.session_state.get("experiment_metadata", {}).get(new_fname, {})

        # Force update widget state
        st.session_state["sw_version_input"] = new_meta.get("version", "not specified")
        st.session_state["exp_time_input"] = new_meta.get("time", "")

    # 4. Render Inputs in Placeholder (Top)
    with meta_placeholder:
        # Ensure metadata store exists
        if "experiment_metadata" not in st.session_state:
            st.session_state["experiment_metadata"] = {}

        # Inputs (Widgets will use values from session_state automatically)
        sw_version = st.text_input("Machine Software Version", key="sw_version_input")
        exp_time = st.text_input("Experiment Time (HH:MM)", key="exp_time_input")

        # Update Metadata Store with current input
        current_fname = os.path.basename(selected_file)
        st.session_state["experiment_metadata"][current_fname] = {
            "version": sw_version,
            "time": exp_time,
        }

    filename = os.path.basename(selected_file)
    curr_meta = st.session_state.get("experiment_metadata", {}).get(filename, {})
    ver_display = curr_meta.get("version", "not specified")
    st.caption(f"**Machine SW Version:** {ver_display}")

# Placeholder for Sensor Selection (to appear at top, below file select)
sensor_placeholder = st.sidebar.empty()

# --- DATADOG LINK ---
try:
    filename = os.path.basename(selected_file)

    # 1. Extract Timestamp (Date + Optional Time)
    # Check for User Override first
    curr_meta = st.session_state.get("experiment_metadata", {}).get(filename, {})
    user_time = curr_meta.get("time", "")

    start_ts = 0
    end_ts = 0
    label_time = ""

    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    date_str = date_match.group(1) if date_match else ""

    if user_time and date_str:
        try:
            # Try parsing HH:MM or HH
            if ":" in user_time:
                dt = datetime.datetime.strptime(
                    f"{date_str} {user_time}", "%Y-%m-%d %H:%M"
                )
            else:
                dt = datetime.datetime.strptime(
                    f"{date_str} {user_time}", "%Y-%m-%d %H"
                )

            # Window: +/- 1 hour
            start_ts = int(
                (dt - datetime.timedelta(hours=1) - datetime.timedelta(minutes=10))
                .replace(tzinfo=datetime.timezone.utc)
                .timestamp()
                * 1000
            )
            end_ts = int(
                (dt + datetime.timedelta(minutes=10))
                .replace(tzinfo=datetime.timezone.utc)
                .timestamp()
                * 1000
            )
            label_time = f"{date_str} {user_time}"
        except:
            pass  # Fallback

    if start_ts == 0:
        # Fallback to Filename logic
        # Try finding YYYY-MM-DD_HH-MM-SS
        time_match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", filename)
        if time_match:
            date_str = time_match.group(1)
            time_str = time_match.group(2)
            dt_str = f"{date_str} {time_str.replace('-', ':')}"
            dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")

            # Precise Window: Start to Start+4h
            start_ts = int(dt.replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)
            end_ts = start_ts + (4 * 3600 * 1000)  # +4h
            label_time = f"{date_str} {time_str}"
        elif date_str:
            # Date Only
            dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            start_ts = int(dt.replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)
            end_ts = start_ts + 86400000  # +24h
            label_time = date_str

    # 2. Extract Host
    # Regex: Matches prefix, then captures alphanumeric host (stopping at - or _)
    host_match = re.search(
        r"_(THE_DANIEL_TEST|DE_GENERAL_TEST-[0-9]+)_([A-Za-z0-9]+)", filename
    )

    if host_match:
        host_name = host_match.group(2).lower()

        # Datadog URL (EU domain per example)
        base_dd = "https://app.datadoghq.eu/logs"
        params = [
            f"query=host:{host_name}",
            "agg_m=count",
            "agg_m_source=base",
            "agg_t=count",
            "cols=host,service,image_tag",
            "fromUser=true",
            "messageDisplay=inline",
            "refresh_mode=paused",
            "storage=hot",
            "stream_sort=desc",
            "viz=stream",
            f"from_ts={start_ts}",
            f"to_ts={end_ts}",
            "live=false",
        ]
        dd_url = f"{base_dd}?{'&'.join(params)}"

        st.sidebar.link_button(f"🐶 Logs: {host_name} ({label_time})", dd_url)
    else:
        # Debug: Show why it failed if user expects it
        # st.sidebar.caption(f"No host found in {filename}")
        pass

except Exception as e:
    pass


# --- SIDEBAR: HYPERPARAMETERS (Always Visible) ---
# Sidebar Params (Need these before running global analysis)
st.sidebar.header("Hyperparameters")

with st.sidebar.expander("Analysis Configuration", expanded=False):
    st.markdown("### Signal & Noise")

    c1, c2 = st.columns(2)
    spline_window = c1.number_input("Smooth Window", value=20, key="spline_window")
    trend_iterations = c2.number_input(
        "Trend Iterations", value=1, min_value=1, max_value=5, key="trend_iterations"
    )

    c3, c4 = st.columns(2)
    trend_method = c3.selectbox(
        "Trend Method", ["Median", "Savitzky-Golay"], key="trend_method"
    )
    poly_order = 3
    if trend_method == "Savitzky-Golay":
        poly_order = c4.number_input(
            "SG Poly Order", value=3, min_value=1, max_value=5, key="poly_order"
        )

    # Spline Post-Processing
    st.markdown("#### Post-Processing")
    apply_spline = st.checkbox(
        "Apply Spline Fit to Trend", value=False, key="apply_spline"
    )
    spline_s = None
    if apply_spline:
        spline_s = st.number_input(
            "Spline Factor (s)",
            value=1e-16,
            format="%.1e",
            key="spline_s",
            help="Smoothing factor. Applied to the already smoothed trend.",
        )

    st.divider()
    noise_window = st.number_input("Noise MAD Window", value=20, key="noise_window")

    vis_noise_low = st.number_input(
        "Vis: Low Noise", value=1e-10, format="%.2e", key="vis_noise_low"
    )
    vis_noise_high = st.number_input(
        "Vis: High Noise", value=0.5e-9, format="%.2e", key="vis_noise_high"
    )

    c_dvis1, c_dvis2 = st.columns(2)
    vis_dnoise_low = c_dvis1.number_input(
        "Vis: Deriv Low", value=1e-10, format="%.2e", key="vis_dnoise_low"
    )
    vis_dnoise_high = c_dvis2.number_input(
        "Vis: Deriv High", value=0.5e-9, format="%.2e", key="vis_dnoise_high"
    )

    deriv_noise_weight = st.number_input(
        "Deriv Noise Weight",
        value=0.0,
        format="%.1f",
        key="deriv_noise_weight",
        help="Weight to add Derivative Noise to the Noise Ratio metric (Noise + W * Deriv).",
    )

    st.markdown("#### Model Comparison")

    # Reference Curve Upload
    ref_curve_path = "src/metrics/Perfect_reference_curve.csv"
    if os.path.exists(ref_curve_path):
        st.caption("Reference curve loaded.")
    else:
        st.warning("No reference curve found.")

    ref_curve_upload = st.file_uploader(
        "Upload Reference Curve (CSV)",
        type=["csv"],
        key="ref_curve_uploader",
        help="Replaces the current Perfect Reference Curve used for model comparison.",
    )
    if ref_curve_upload is not None:
        try:
            # Validate: must have a time column and at least one data column
            test_df = pd.read_csv(ref_curve_upload)
            time_cols = [c for c in test_df.columns if "time" in c.lower()]
            data_cols = [
                c
                for c in test_df.columns
                if c not in time_cols and "Unnamed" not in c
            ]
            if not time_cols or not data_cols:
                st.error(
                    "Invalid CSV: needs a 'time' column and at least one data column."
                )
            else:
                ref_curve_upload.seek(0)
                with open(ref_curve_path, "wb") as f:
                    f.write(ref_curve_upload.getbuffer())
                st.success("Reference curve updated!")
                st.cache_data.clear()
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    normalization_method = st.selectbox(
        "Normalization", ["MinMax [0,1]", "Z-Score"], key="normalization_method"
    )

    st.write("Reference Points (Time, Value)")

    # Initialize as list of dicts for button-based input
    if "ref_points" not in st.session_state:
        st.session_state["ref_points"] = []

    # Conversion from old DataFrame format if exists
    if isinstance(st.session_state["ref_points"], pd.DataFrame):
        st.session_state["ref_points"] = st.session_state["ref_points"].to_dict(
            "records"
        )

    # Render Inputs
    points_to_remove = []
    for i, pt in enumerate(st.session_state["ref_points"]):
        c1, c2, c3 = st.columns([0.4, 0.4, 0.2])
        with c1:
            pt["Time"] = st.text_input(
                f"Time #{i + 1}", value=str(pt.get("Time", "")), key=f"ref_t_{i}"
            )
        with c2:
            pt["Value"] = st.text_input(
                f"Value #{i + 1}", value=str(pt.get("Value", "")), key=f"ref_v_{i}"
            )
        with c3:
            if st.button("🗑️", key=f"del_ref_{i}"):
                points_to_remove.append(i)

    # Handle Removal
    for i in sorted(points_to_remove, reverse=True):
        del st.session_state["ref_points"][i]
        st.rerun()

    if st.button("➕ Add Reference Point"):
        st.session_state["ref_points"].append({"Time": "", "Value": ""})
        st.rerun()

    st.divider()
    st.markdown("### Spike Detection")
    loop_window = st.number_input("Loop Window", value=40, key="loop_window")
    loop_amp_thresh = st.number_input(
        "Loop Amp Thresh", value=5e-10, format="%.0e", key="loop_amp_thresh"
    )

    c_base1, c_base2 = st.columns(2)
    baseline_window = c_base1.number_input(
        "Base Window", value=3, key="baseline_window"
    )
    baseline_tol = c_base2.number_input("Base Tol", value=2.0, key="baseline_tolerance")

    hampel_sigma = st.number_input(
        "Hampel Sigma", value=4.0, step=0.5, key="hampel_sigma"
    )

    intersect_spikes = st.checkbox(
        "Confirmed Spikes Only (Intersection)", key="intersect_spikes"
    )
    spike_proximity = 5
    if intersect_spikes:
        spike_proximity = st.number_input(
            "Intersection Proximity (pts)", value=5, min_value=0, key="spike_proximity"
        )

    st.divider()
    st.markdown("### Step Detection")

    step_window = st.number_input("Step Window", value=30, key="step_window")
    step_method = st.selectbox(
        "Primary Method", ["Piecewise Fit", "Slope Pulse"], key="step_method"
    )

    st.caption("Method Parameters")
    c_p, c_s = st.columns(2)

    # Piecewise Params
    with c_p:
        st.markdown("**Piecewise**")
        step_poly_order = st.number_input(
            "Fit Order", value=1, min_value=1, max_value=3, key="step_poly_order"
        )
        thresh_piecewise = st.number_input(
            "Thresh (Score)", value=0.65, format="%.2f", key="thresh_piecewise"
        )

    # Slope Params
    with c_s:
        st.markdown("**Slope Pulse**")
        st.write("")  # Spacer for alignment
        st.write("")
        thresh_slope = st.number_input(
            "Thresh (Raw)", value=1e-10, format="%.2e", key="thresh_slope"
        )

    intersect_steps = st.checkbox(
        "Confirmed Steps (Intersection)", key="intersect_steps"
    )

    # Logic mapping
    step_threshold = (
        thresh_piecewise if step_method == "Piecewise Fit" else thresh_slope
    )

with st.sidebar.expander("Grading Rules", expanded=False):
    st.caption("Pass/Fail Thresholds")
    grading_params = {
        "limit_noise_ratio": st.number_input(
            "Max Noise Ratio", value=0.1, format="%.2f", key="limit_noise_ratio"
        ),
        "limit_snr_median": st.number_input(
            "Min Median SNR", value=5.0, format="%.1f", key="limit_snr_median"
        ),
        "limit_snr_p10": st.number_input(
            "Min P10 SNR", value=2.0, format="%.1f", key="limit_snr_p10"
        ),
        "limit_loop_ratio": st.number_input(
            "Max Anomalies Ratio", value=0.05, format="%.2f", key="limit_loop_ratio"
        ),
        "limit_step_ratio": st.number_input(
            "Max Step Events", value=0.15, format="%.2f", key="limit_step_ratio"
        ),
        "limit_signal_std": st.number_input(
            "Max Residual Spread (Std)",
            value=1.0e-8,
            format="%.2e",
            key="limit_signal_std",
        ),
        "limit_deriv_noise": st.number_input(
            "Max Derivative Noise (Median)",
            value=1.0e-11,
            format="%.2e",
            key="limit_deriv_noise",
        ),
        "limit_model_rmse": st.number_input(
            "Max Model Error (RMSE)", value=0.2, format="%.2f", key="limit_model_rmse"
        ),
        "limit_ref_points_rmse": st.number_input(
            "Max Point RMSE", value=1.0e-9, format="%.2e", key="limit_ref_points_rmse"
        ),
        "limit_overall_score": st.number_input(
            "Min Overall Score", value=0.7, format="%.2f", key="limit_overall_score"
        ),
        "limit_global_pass_rate": st.number_input(
            "Min Global Pass Rate",
            value=0.90,
            format="%.2f",
            key="limit_global_pass_rate",
        ),
        "limit_global_avg_score": st.number_input(
            "Min Global Avg Score",
            value=0.80,
            format="%.2f",
            key="limit_global_avg_score",
        ),
        "limit_chip_spread": st.number_input(
            "Max Chip Uniformity (Std)",
            value=1.0e-8,
            format="%.2e",
            key="limit_chip_spread",
        ),
    }

if st.sidebar.button("💾 Save Configuration"):
    save_config_to_file()


# --- PAGE ROUTING ---
if page == "Experiment Description":
    st.header("Description")
    pdf_file = "The Daniel Test.pdf"

    if os.path.exists(pdf_file):
        with open(pdf_file, "rb") as f:
            pdf_bytes = f.read()

        # 1. Download Button (Always available)
        st.download_button(
            label="📥 Download / Print PDF",
            data=pdf_bytes,
            file_name="The_Daniel_Test_Description.pdf",
            mime="application/pdf",
        )

        st.divider()

        # 2. Render as Images using PyMuPDF (fitz)
        # This is the most robust way to "preview" a PDF if browser plugins fail
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            st.caption(f"Showing {len(doc)} pages:")

            for i, page in enumerate(doc):
                # Render page to image (pixmap)
                pix = page.get_pixmap(dpi=150)  # 150 DPI is good for screen
                img_bytes = pix.tobytes("png")

                st.image(img_bytes, caption=f"Page {i + 1}", use_container_width=True)

        except ImportError:
            st.error(
                "Library 'pymupdf' is missing. Please run 'uv sync' to install it."
            )
        except Exception as e:
            st.error(f"Error rendering PDF: {e}")

    else:
        st.error(f"PDF file '{pdf_file}' not found.")
    st.stop()

else:
    # --- ANALYSIS DASHBOARD CONTENT ---

    if not csv_files:
        st.stop()  # Should have stopped earlier but double check

    # Load Data
    df, time_col, sensor_cols = load_data(selected_file)
    if df is None:
        st.error("Error loading file.")
        st.stop()

    # --- GLOBAL ANALYSIS MONITOR (Pre-calc for Sensor Selector default) ---
    # Cache the global analysis to avoid re-running on every interaction unless params change
    @st.cache_data
    def get_global_results_v7(file_path, _params, _grading):
        return run_global_analysis(df, time_col, sensor_cols, _params, _grading)

    # Package params for analysis function
    analysis_params = {
        "noise_window": noise_window,
        "spline_window": spline_window,
        "trend_method": trend_method,
        "poly_order": poly_order,
        "trend_iterations": trend_iterations,
        "apply_spline": apply_spline,
        "spline_s": spline_s,
        "loop_window": loop_window,
        "loop_amp_thresh": loop_amp_thresh,
        "baseline_window": baseline_window,
        "baseline_tolerance": baseline_tol,
        "hampel_sigma": hampel_sigma,
        "step_window": step_window,
        "step_poly_order": step_poly_order,
        "thresh_piecewise": thresh_piecewise,
        "thresh_slope": thresh_slope,
        "intersect_steps": intersect_steps,
        "intersect_spikes": intersect_spikes,
        "spike_proximity": spike_proximity,
        "use_trend_anom": False,  # Default for global analysis
        "use_trend_step": False,  # Default for global analysis
        # If intersecting, step_method might be ambiguous for UI labels, but we'll default to Piecewise or handle it.
        # We still need 'step_method' if not intersecting.
        "step_method": step_method,
        "normalization_method": normalization_method,
        "deriv_noise_weight": deriv_noise_weight,
        "ref_points": st.session_state.get("ref_points"),
    }

    # Run Global Analysis for defaults
    global_stats_list = get_global_results_v7(
        selected_file, analysis_params, grading_params
    )
    summary_df = pd.DataFrame([asdict(s) for s in global_stats_list])

    st.sidebar.divider()
    if st.sidebar.button("📝 Save Experiment to Log"):
        # Calculate Chip Stats for Logging
        chip_spread = summary_df["median_signal"].std()
        chip_pass = chip_spread <= grading_params["limit_chip_spread"]

        # Get Version
        curr_meta = st.session_state.get("experiment_metadata", {}).get(filename, {})
        ver_to_save = curr_meta.get("version", "not specified")

        save_experiment_log(summary_df, filename, chip_spread, chip_pass, ver_to_save)

    if st.sidebar.button("📂 View/Load Experiment Log"):
        view_experiment_log()

    save_plots = st.sidebar.checkbox("Save Debug Plots", key="save_plots")

    # Determine Default Sensor (First Failed, or First Overall, or Loaded from Log)
    failed_list = summary_df[~summary_df["overall_pass"]]["metric"].tolist()

    # Check for Log Load Request
    if (
        "load_sensor_req" in st.session_state
        and st.session_state.get("load_file_req") == selected_file
    ):
        req_sensor = st.session_state["load_sensor_req"]
        if req_sensor in sensor_cols:
            st.session_state["selected_sensor_name"] = req_sensor
    # Default to first failure if no explicit selection for this file
    elif st.session_state.get("selected_sensor_name") not in sensor_cols:
        if failed_list:
            st.session_state["selected_sensor_name"] = failed_list[0]
        else:
            st.session_state["selected_sensor_name"] = sensor_cols[0]

    # Guard: ensure selection is valid
    if st.session_state.get("selected_sensor_name") not in sensor_cols:
        st.session_state["selected_sensor_name"] = sensor_cols[0]

    selected_sensor = st.session_state["selected_sensor_name"]

    # Render Sensor selector in the placeholder at the top
    with sensor_placeholder.container():
        # Show current sensor with pass/fail badge
        sensor_row = summary_df[summary_df["metric"] == selected_sensor]
        if not sensor_row.empty:
            s_passed = sensor_row.iloc[0]["overall_pass"]
            s_score = sensor_row.iloc[0]["overall_score"]
            s_badge = "PASS" if s_passed else "FAIL"
            st.markdown(
                f"**Sensor:** `{selected_sensor}` ({s_score:.2f} {s_badge})"
            )
        else:
            st.markdown(f"**Sensor:** `{selected_sensor}`")

        if st.button(
            "Select Sensor", key="btn_select_sensor", use_container_width=True
        ):
            select_sensor_dialog(sensor_cols, summary_df)

# --- GLOBAL ANALYSIS MONITOR UI ---


st.subheader("Global Monitor (Chip)")


# Metrics Calculation

total_sensors = len(summary_df)

passed_sensors = summary_df["overall_pass"].sum()

failed_sensors = total_sensors - passed_sensors

pass_rate_val = (passed_sensors / total_sensors) if total_sensors > 0 else 0.0

avg_score_val = summary_df["overall_score"].mean()

chip_spread_val = summary_df["median_signal"].std()


# Thresholds

lim_pass_rate = grading_params.get("limit_global_pass_rate", 0.90)

lim_avg_score = grading_params.get("limit_global_avg_score", 0.80)

lim_spread = grading_params.get("limit_chip_spread", 1e-8)


# Pass Checks

pass_rate_ok = pass_rate_val >= lim_pass_rate

avg_score_ok = avg_score_val >= lim_avg_score

spread_ok = chip_spread_val <= lim_spread


global_pass = pass_rate_ok and avg_score_ok and spread_ok


# 1. Main Banner

if global_pass:
    st.success(f"### ✅ CHIP PASSED\nAll global criteria met.")

else:
    st.error(f"### ❌ CHIP FAILED\nOne or more global criteria failed.")


# 2. Detailed Breakdown (Merged 5 columns)

c1, c2, c3, c4, c5 = st.columns(5)


with c1:
    st.metric(
        "Pass Rate",
        f"{pass_rate_val:.1%}",
        delta=f"Target: > {lim_pass_rate:.1%}",
        delta_color="normal" if pass_rate_ok else "inverse",
    )

    st.caption(f"{passed_sensors} / {total_sensors} Passed")


with c2:
    st.metric(
        "Global Avg Score",
        f"{avg_score_val:.2f}",
        delta=f"Target: > {lim_avg_score:.2f}",
        delta_color="normal" if avg_score_ok else "inverse",
    )


with c3:
    st.metric(
        "Chip Uniformity (Std)",
        f"{chip_spread_val:.2e}",
        delta=f"Target: < {lim_spread:.2e}",
        delta_color="normal" if spread_ok else "inverse",
    )

    if spread_ok:
        st.caption("✅ Uniformity OK")

    else:
        st.caption("❌ Non-Uniform")


with c4:
    st.metric("Total Sensors", total_sensors)


with c5:
    st.metric("Failed Sensors", failed_sensors, delta="Failures", delta_color="inverse")


# --- GLOBAL SIGNAL ANALYSIS (Moved Up) ---

with st.expander("Signal Distribution & Trends", expanded=False):
    st.markdown("Analysis of the signal distribution across all sensors.")

    # Prepare Data

    # df[sensor_cols] is the data

    numeric_df = df[sensor_cols].apply(pd.to_numeric, errors="coerce")

    # 1. Aggregated Trend (Mean +/- STD)

    mean_signal = numeric_df.mean(axis=1)

    std_signal = numeric_df.std(axis=1)

    upper_bound = mean_signal + std_signal

    lower_bound = mean_signal - std_signal

    fig_agg = go.Figure()

    # Std Bound (filled)

    fig_agg.add_trace(
        go.Scatter(
            x=pd.concat([df[time_col], df[time_col][::-1]]),
            y=pd.concat([upper_bound, lower_bound[::-1]]),
            fill="toself",
            fillcolor="rgba(0,100,80,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Std Dev",
        )
    )

    # Mean Line

    fig_agg.add_trace(
        go.Scatter(
            x=df[time_col],
            y=mean_signal,
            line=dict(color="rgb(0,100,80)"),
            name="Mean Signal",
        )
    )

    fig_agg.update_layout(
        title="Average Signal Across All Sensors (± 1 SD)",
        xaxis_title="Time",
        yaxis_title="Signal",
    )

    st.plotly_chart(fig_agg, use_container_width=True)

    # 2. Histogram of Values

    # Flatten data for histogram

    st.write("**Y-Axis Value Distribution (All Data)**")

    # Sample if too large?

    flat_values = numeric_df.values.flatten()

    flat_values = flat_values[~np.isnan(flat_values)]

    fig_hist = px.histogram(x=flat_values, nbins=100, title="Global Signal Density")

    fig_hist.update_layout(xaxis_title="Signal Value", yaxis_title="Count")

    st.plotly_chart(fig_hist, use_container_width=True)

    # 3. All Traces (Optional)

    if st.checkbox("Show All Individual Traces"):
        fig_all = go.Figure()

        for col in sensor_cols:
            fig_all.add_trace(
                go.Scatter(
                    x=df[time_col],
                    y=numeric_df[col],
                    mode="lines",
                    name=col,
                    opacity=0.5,
                )
            )

        fig_all.update_layout(
            title="All Sensors Overlay", xaxis_title="Time", yaxis_title="Signal"
        )

        st.plotly_chart(fig_all, use_container_width=True)


# Summary Table Dialog
@st.dialog("Global Summary Table")
def show_summary_dialog():
    view_type = st.radio("View Mode", ["Table", "Heatmap"], horizontal=True)

    if view_type == "Table":
        st.write("Overview of all sensors in the current file.")
        # Formatting only (no background colors)
        st.dataframe(
            summary_df[
                [
                    "metric",
                    "overall_score",
                    "overall_pass",
                    "noise_ratio",
                    "snr_median",
                    "anomalies_count",
                    "signal_std",  # residual ratio was removed
                ]
            ].style.format(
                {
                    "overall_score": "{:.2f}",
                    "noise_ratio": "{:.2e}",
                    "snr_median": "{:.1f}",
                    "anomalies_count": "{:d}",
                    "signal_std": "{:.2e}",
                }
            ),
            use_container_width=True,
            height=600,
        )
    else:
        st.write("Geometric Heatmap of Sensor Metrics (Row x Col)")

        # 1. Select Metric
        metric_options = [
            "overall_score",
            "noise_ratio",
            "snr_median",
            "anomalies_count",
            "signal_std",
        ]
        metric_to_plot = st.selectbox("Select Metric to Visualize", metric_options)

        # 2. Parse Coordinates
        # Expecting pattern like "SM_R4C1:..." -> R=4, C=1
        heatmap_data = []
        for _, row in summary_df.iterrows():
            name = row["metric"]
            match = re.search(r"R(\d+)C(\d+)", name)
            if match:
                r = int(match.group(1))
                c = int(match.group(2))
                heatmap_data.append(
                    {"Row": r, "Col": c, "Value": row[metric_to_plot], "Name": name}
                )

        if not heatmap_data:
            st.warning(
                "No sensor names matched the R{row}C{col} pattern (e.g., 'SM_R4C1'). Cannot generate heatmap."
            )
        else:
            hm_df = pd.DataFrame(heatmap_data)
            # Pivot: Index=Row, Columns=Col, Values=Value
            pivot_df = hm_df.pivot(index="Row", columns="Col", values="Value")

            fig_hm = go.Figure(
                data=go.Heatmap(
                    z=pivot_df.values,
                    x=pivot_df.columns,
                    y=pivot_df.index,
                    colorscale="Viridis",
                    text=pivot_df.values,
                    texttemplate="%{text:.2f}"
                    if metric_to_plot == "overall_score"
                    else "%{text:.2e}",
                    xgap=2,  # Gap for visual separation
                    ygap=2,
                )
            )
            fig_hm.update_layout(
                title=f"Heatmap: {metric_to_plot}",
                xaxis_title="Column",
                yaxis_title="Row",
                xaxis=dict(dtick=1, side="top"),  # Integers, labels on top like a plate
                yaxis=dict(
                    dtick=1, autorange="reversed", scaleanchor="x"
                ),  # Integers, reversed, square aspect
                width=700,
                height=700,
            )
            st.plotly_chart(
                fig_hm, use_container_width=False
            )  # Fixed width/height for aspect ratio to hold


if st.button("📄 Open Summary Table"):
    show_summary_dialog()


st.divider()


# --- DETAILED ANALYSIS ---


st.header("Detailed Analysis (Mologram)")

# Run Detail Analysis (Default Raw)
t = df[time_col]
y = df[selected_sensor]
y = pd.to_numeric(y, errors="coerce").interpolate(limit_direction="both")
if y.isna().any():
    y = y.fillna(y.median())


def status_badge(passed):
    return "✅ PASS" if passed else "❌ FAIL"


# Initial Full Analysis (Raw Basis)
stats, details = analyze_sensor(
    selected_sensor, t, y, grading_params=grading_params, **analysis_params
)

# Unpack details
noise_series = details["noise_series"]
noise_diff_series = details["noise_diff_series"]
smooth_trend = details["smooth_trend"]
snr_series = details["snr_series"]
loop_mask = details["loop_mask"]
res_mask = details["res_mask"]
residuals = details["residuals"]
res_thresh = details["res_thresh"]
step_metric = details["step_metric"]
step_mask = details["step_mask"]
norm_y = details["norm_y"]
norm_ref = details["norm_ref"]
model_error_msg = details["model_error_msg"]

# Preserve Raw Detection Masks for Confirmation Logic
raw_res_mask = res_mask.copy()
raw_loop_mask = loop_mask.copy()

# Placeholders for Top Elements (rendered later to reflect tab toggles)
metrics_container = st.container()
plot_container = st.container()

# Initialize Toggles (will be updated inside tabs)
use_trend_anom = False
use_trend_step = False

# Imports for dynamic recalc
from src.metrics.spikes import detect_loop_spikes, detect_residual_spikes
from src.metrics.steps import piecewise_improvement
from src.metrics.step_slope import detect_slope_steps
from src.scoring import evaluate_metric

# --- TABS ---
tab_doc, tab_noise, tab_snr, tab_anom, tab_step, tab_deriv, tab_model = st.tabs(
    [
        "Documentation",
        "Noise Analysis",
        "SNR",
        "Anomalies",
        "Step Detection",
        "Derivatives",
        "Model Comparison",
    ]
)

with tab_doc:
    st.markdown(r"""
    ### 1. Global Metrics (Chip Health)
    These metrics evaluate the entire experiment (chip) as a whole.
    
    *   **Global Pass Rate:** The percentage of individual sensors that passed all their specific criteria. 
        *   *Pass Condition:* Rate $\ge$ `Min Global Pass Rate` (default 90%).
    *   **Global Avg Score:** The average `Overall Score` of all sensors on the chip.
        *   *Pass Condition:* Average $\ge$ `Min Global Avg Score` (default 0.8).
    *   **Chip Uniformity:** Measures how consistent the signal levels are across different sensors. Calculated as the Standard Deviation of the median signal of all sensors.
        *   *Pass Condition:* StdDev $\le$ `Max Chip Uniformity`.

    ---

    ### 2. Per-Sensor Metrics
    Each sensor is graded individually on these 8 criteria.

    **1. Noise Ratio**
    A normalized measure of signal noise.
    *   **Calculation:** Uses the **Rolling Median Absolute Deviation (MAD)** of the residuals (Raw - Trend). If the signal is flat (Median=0), it falls back to the Mean.
    *   **Formula:** $\text{Ratio} = \frac{\text{Noise} + (W \cdot \text{DerivNoise})}{\text{Baseline Amplitude}}$
    *   *Note:* Includes a penalty for derivative noise if `Deriv Noise Weight` ($W$) > 0.

    **2. Deriv Noise (Slope Stability)**
    Measures the stability of the signal's rate of change (how "wiggly" the slope is).
    *   **Calculation:** The Rolling MAD of the *residuals* of the 1st Derivative of the smoothed trend ($ \text{Derivative Residuals} = | \frac{d\text{Trend}}{dt} - \text{Smoothed}(\frac{d\text{Trend}}{dt}) | $).
    *   *Pass Condition:* Value $\le$ `Max Derivative Noise`.

    **3. Signal-to-Noise Ratio (SNR)**
    Measures signal strength relative to background noise.
    *   **Formula:** $\text{SNR}(t) = \frac{\text{Trend}(t)}{\text{Local Noise}(t)}$
    *   **Metrics:** We check both the **Median SNR** (typical performance) and the **P10 SNR** (worst-case 10% performance).
    *   *Pass Condition:* Median $\ge$ `Min Median SNR` AND P10 $\ge$ `Min P10 SNR`.

    **4. Anomalies (Spikes)**
    Percentage of data points flagged as artifacts/glitches.
    *   **Loop Spike:** A deviation that returns to baseline (max 1 oscillation).
    *   **Hampel Spike:** A point deviating from the local trend by $> \sigma$ standard deviations.
    *   *Pass Condition:* Total Anomalies (calculated from `Max Anomalies Ratio`) $\le$ actual count.

    **5. Step Events**
    Percentage of data points involved in step-like transitions (sudden jumps).
    *   **Methods:** "Piecewise Fit" (score-based) or "Slope Pulse" (derivative-based). You can require *Intersection* (both must agree) for stricter detection.
    *   *Pass Condition:* Total Steps (calculated from `Max Step Events`) $\le$ actual count.

    **6. Residual Spread**
    Measures the overall "wobbliness" or standard deviation of the residuals (Raw - Trend).
    *   *Pass Condition:* Spread $\le$ `Max Residual Spread`.

    **7. Model RMSE (Normalized)**
    Compares the sensor's shape against a "Perfect Reference Curve".
    *   **Process:** Both signals are normalized (0 to 1 or Z-Score) and aligned in time.
    *   **Metric:** Root Mean Square Error (RMSE) between them.
    *   *Pass Condition:* RMSE $\le$ `Max Model Error`.

    **8. Ref Points RMSE (Raw)**
    Checks if the sensor passes through specific user-defined checkpoints.
    *   **Process:** Compares the sensor's **Smoothed Trend** (Raw values) against your list of (Time, Expected Value) points.
    *   **Metric:** RMSE of the differences at those specific times.
    *   *Pass Condition:* RMSE $\le$ `Max Point RMSE`.

    **9. Overall Score**
    A weighted average of all the above scores (normalized 0-1).
    $$ S_{\text{total}} = \sum (w_i \cdot S_i) $$
    *   **Weights:** Anomalies (20%), Model (15%), Steps (15%), Noise/SNR/Std/Deriv/Ref (10% each).
    *   *Pass Condition:* $S_{\text{total}} \ge 0.7$ **AND** all individual metrics must pass.
    """)

with tab_noise:
    st.subheader("Noise Analysis")
    st.info(f"Trend Method Used: **{trend_method}**")

    # Signal with Noise Highlights
    # (Checkbox moved below)

    fig_noise_overlay = go.Figure()
    fig_noise_overlay.add_trace(
        go.Scatter(x=t, y=y, name="Signal", line=dict(color="lightgray", width=1))
    )

    # Target Noise Zone (Green)
    target_noise_mask = (noise_series <= vis_noise_high) & (
        noise_series >= vis_noise_low
    )
    if target_noise_mask.any():
        fig_noise_overlay.add_trace(
            go.Scatter(
                x=t[target_noise_mask],
                y=y[target_noise_mask],
                mode="markers",
                marker=dict(color="green", size=2),
                name="Target Noise Zone",
            )
        )

    # Highlight Target Derivative Noise (Orange)
    # User request: Keep only noise between low and high thresholds (like signal noise logic)
    show_deriv_on_signal = st.checkbox(
        "Highlight Target Derivative Noise on Signal", key="chk_highlight_deriv"
    )

    if show_deriv_on_signal:
        # Between Low and High
        target_deriv_mask = (noise_diff_series >= vis_dnoise_low) & (
            noise_diff_series <= vis_dnoise_high
        )

        if target_deriv_mask.any():
            fig_noise_overlay.add_trace(
                go.Scatter(
                    x=t[target_deriv_mask],
                    y=y[target_deriv_mask],
                    mode="markers",
                    marker=dict(color="orange", symbol="x", size=4),
                    name="Target Deriv Noise",
                )
            )
        else:
            st.caption(
                f"No derivative noise points in range [{vis_dnoise_low:.2e}, {vis_dnoise_high:.2e}]."
            )

    fig_noise_overlay.update_layout(
        title="Signal with Noise Highlights", xaxis_title="Time", yaxis_title="Signal"
    )
    st.plotly_chart(fig_noise_overlay, use_container_width=True)

    st.write("Noise Timeline (Rolling MAD)")

    # Plot 1: Signal Noise
    fig_mad = px.line(x=t, y=noise_series, title="Signal Noise (Amplitude Stability)")
    fig_mad.update_traces(line_color="blue")

    fig_mad.add_hline(
        y=stats.noise_median,
        line_dash="dash",
        annotation_text=f"Median: {stats.noise_median:.2e}",
        line_color="blue",
        annotation_position="top left",
    )
    fig_mad.add_hline(
        y=vis_noise_low,
        line_dash="dot",
        annotation_text="Low Thresh",
        line_color="green",
        annotation_position="bottom right",
    )
    fig_mad.add_hline(
        y=vis_noise_high,
        line_dash="dot",
        annotation_text="High Thresh",
        line_color="red",
        annotation_position="top right",
    )
    st.plotly_chart(fig_mad, use_container_width=True)

    # Plot 2: Derivative Noise
    show_deriv_noise = st.checkbox(
        "Show Derivative Noise Analysis", key="chk_show_deriv_plot"
    )
    if show_deriv_noise:
        fig_dmad = px.line(
            x=t,
            y=noise_diff_series,
            title="Derivative Noise (Rate of Change Stability)",
        )
        fig_dmad.update_traces(line_color="orange")

        fig_dmad.add_hline(
            y=vis_dnoise_high,
            line_dash="dot",
            annotation_text="Deriv High",
            line_color="orange",
            annotation_position="top right",
        )
        fig_dmad.add_hline(
            y=vis_dnoise_low,
            line_dash="dot",
            annotation_text="Deriv Low",
            line_color="green",
            annotation_position="bottom right",
        )
        st.plotly_chart(fig_dmad, use_container_width=True)
with tab_snr:
    st.subheader("Signal-to-Noise Ratio")
    st.info(f"Trend Method Used: **{trend_method}**")
    fig_snr = px.line(x=t, y=snr_series, title="SNR Evolution")
    target_snr = grading_params["limit_snr_median"]
    fig_snr.add_hline(
        y=target_snr,
        line_color="green",
        line_dash="dash",
        annotation_text=f"Target ({target_snr})",
        annotation_position="top left",
    )
    st.plotly_chart(fig_snr, use_container_width=True)

with tab_anom:
    st.subheader("Anomaly Details")
    use_trend_anom = st.toggle(
        "Analyze Smoothed Trend",
        value=False,
        key="use_trend_anom",
        help="Run Spike detection on Smoothed Trend instead of Raw Data",
    )

    if use_trend_anom:
        # Calculate a local baseline for the trend (to detect bumps IN the trend)
        # Use the same window logic as the raw detection's implicit baseline concept
        trend_baseline = (
            pd.Series(smooth_trend)
            .rolling(window=analysis_params["loop_window"], center=True, min_periods=1)
            .median()
            .values
        )

        # Re-run spike detection on smooth_trend vs its local baseline
        loop_mask = detect_loop_spikes(
            smooth_trend.values,
            noise_series.values,
            smooth=trend_baseline,
            window=analysis_params["loop_window"],
            amp_threshold=analysis_params["loop_amp_thresh"],
            baseline_window=analysis_params["baseline_window"],
            baseline_tolerance=analysis_params["baseline_tolerance"],
        )
        # Re-calc Hampel on Trend
        h_win = max(3, int(analysis_params["spline_window"]))
        if h_win % 2 == 0:
            h_win -= 1
        light_smooth_trend = (
            pd.Series(smooth_trend)
            .rolling(window=h_win, center=True, min_periods=1)
            .median()
            .values
        )
        res_mask, residuals, res_thresh = detect_residual_spikes(
            smooth_trend.values,
            light_smooth_trend,
            sigma_factor=analysis_params["hampel_sigma"],
        )

    c1, c2 = st.columns(2)
    c1.info(f"Loop Spikes: {loop_mask.sum()} points")
    c2.info(f"Hampel Spikes: {res_mask.sum()} points")

    st.write("**Hampel (Residual) Deviation**")
    fig_res = px.line(x=t, y=residuals, title="Signal Residuals (Input - Baseline)")
    fig_res.add_hline(y=res_thresh, line_color="red", line_dash="dash")
    fig_res.add_hline(y=-res_thresh, line_color="red", line_dash="dash")
    st.plotly_chart(fig_res, use_container_width=True)

    st.write("**Loop Spike Deviation**")
    # Plot deviation from median (as in reference script)
    # Use input based on toggle
    y_anom_input = smooth_trend if use_trend_anom else y
    loop_dev = np.abs(
        y_anom_input
        - y_anom_input.rolling(window=loop_window, center=True, min_periods=1).median()
    )
    fig_loop = px.line(x=t, y=loop_dev, title="Deviation from Local Median")
    fig_loop.add_hline(
        y=loop_amp_thresh,
        line_color="gray",
        line_dash="dash",
        annotation_text="Amp Threshold",
    )
    st.plotly_chart(fig_loop, use_container_width=True)

with tab_step:
    st.subheader(
        f"Step Detection: {step_method if not intersect_steps else 'Intersection Mode'}"
    )
    use_trend_step = st.toggle(
        "Analyze Smoothed Trend",
        value=False,
        key="use_trend_step",
        help="Run Step detection on Smoothed Trend instead of Raw Data",
    )

    # Dynamic Recalculation on Trend
    if use_trend_step:
        # Recalculate based on mode
        if intersect_steps:
            # Run both on trend
            m_p = piecewise_improvement(
                t.values,
                smooth_trend.values,
                window=analysis_params["step_window"],
                poly_order=analysis_params["step_poly_order"],
            )
            msk_p = np.nan_to_num(m_p, 0) > analysis_params["thresh_piecewise"]

            m_s, msk_s = detect_slope_steps(
                smooth_trend.values,
                window=analysis_params["step_window"],
                slope_threshold=analysis_params["thresh_slope"],
            )

            step_mask = msk_p & msk_s
            step_metric = m_p  # Visualization preference
        else:
            # Run selected on trend
            if analysis_params["step_method"] == "Slope Pulse":
                step_metric, step_mask = detect_slope_steps(
                    smooth_trend.values,
                    window=analysis_params["step_window"],
                    slope_threshold=analysis_params["thresh_slope"],
                )
            else:
                step_metric = piecewise_improvement(
                    t.values,
                    smooth_trend.values,
                    window=analysis_params["step_window"],
                    poly_order=analysis_params["step_poly_order"],
                )
                step_mask = np.zeros_like(y, dtype=bool)
                if step_metric is not None:
                    step_mask = (
                        np.nan_to_num(step_metric, 0)
                        > analysis_params["thresh_piecewise"]
                    )

    if intersect_steps:
        st.info("Displaying Steps confirmed by **Piecewise Fit** AND **Slope Pulse**.")

        # Plot 1: Piecewise
        st.write("**Metric 1: Piecewise Fit Score**")
        fig_p = go.Figure()
        fig_p.add_trace(
            go.Scatter(
                x=t,
                y=details["metric_piece"],
                mode="markers",
                marker=dict(
                    size=6,
                    color=details["metric_piece"],
                    colorscale="Viridis",
                    showscale=True,
                    cmin=0,
                    cmax=1,
                ),
                name="Piecewise Score",
            )
        )
        fig_p.add_hline(
            y=analysis_params["thresh_piecewise"],
            line_dash="dash",
            line_color="red",
            annotation_text="Thresh",
        )
        st.plotly_chart(fig_p, use_container_width=True)

        # Plot 2: Slope Pulse
        st.write("**Metric 2: Slope Pulse Height**")
        fig_s = go.Figure()
        fig_s.add_trace(
            go.Scatter(
                x=t,
                y=details["metric_slope"],
                mode="markers",
                marker=dict(
                    size=6,
                    color=details["metric_slope"],
                    colorscale="Magma",
                    showscale=True,
                    cmax=analysis_params["thresh_slope"] * 2,
                ),
                name="Slope Metric",
            )
        )
        fig_s.add_hline(
            y=analysis_params["thresh_slope"],
            line_dash="dash",
            line_color="red",
            annotation_text="Thresh",
        )
        st.plotly_chart(fig_s, use_container_width=True)

    else:
        desc = "Score/Metric"
        st.write(f"Metric: **{desc}**. Higher values indicate likelihood of a step.")

        fig_steps = go.Figure()
        fig_steps.add_trace(
            go.Scatter(
                x=t,
                y=step_metric,
                mode="markers",
                marker=dict(
                    size=6, color=step_metric, colorscale="Viridis", showscale=True
                ),
                name="Step Metric",
            )
        )
        # Show threshold line(s)
        thresh = (
            analysis_params["thresh_slope"]
            if step_method == "Slope Pulse"
            else analysis_params["thresh_piecewise"]
        )
        fig_steps.add_hline(
            y=thresh, line_dash="dash", line_color="red", annotation_text="Threshold"
        )

        st.plotly_chart(fig_steps, use_container_width=True)

with tab_deriv:
    st.subheader("Signal Derivatives")
    st.write("Derivatives calculated on the **Smoothed Trend** (to suppress noise).")

    # Calculate Derivatives
    dy = np.gradient(smooth_trend)
    ddy = np.gradient(dy)

    fig_deriv = go.Figure()
    fig_deriv.add_trace(
        go.Scatter(
            x=t,
            y=dy,
            name="1st Derivative (Slope)",
            line=dict(color="orange", width=1.5),
        )
    )
    fig_deriv.add_trace(
        go.Scatter(
            x=t,
            y=ddy,
            name="2nd Derivative (Curvature)",
            line=dict(color="purple", width=1.5),
            visible="legendonly",
        )
    )
    fig_deriv.update_layout(
        title="Derivatives of Smoothed Trend",
        xaxis_title="Time",
        yaxis_title="Derivative Value",
    )
    st.plotly_chart(fig_deriv, use_container_width=True)

    st.write("**Derivative Noise (Stability)**")
    fig_dnoise = px.line(
        x=t, y=noise_diff_series, title="Rolling Noise of 1st Derivative"
    )
    st.plotly_chart(fig_dnoise, use_container_width=True)

    st.divider()
    if st.checkbox("Show Raw 1st Derivative (Noisy)"):
        raw_dy = np.gradient(y)
        fig_raw_dy = px.line(x=t, y=raw_dy, title="1st Derivative of Raw Signal")
        fig_raw_dy.update_traces(line_color="lightblue")
        st.plotly_chart(fig_raw_dy, use_container_width=True)

with tab_model:
    st.subheader("Reference Model Comparison")

    if model_error_msg:
        st.error(f"Model Comparison Failed: {model_error_msg}")
        st.warning(
            "Ensure 'Perfect_reference_curve.csv' is in the folder and has valid 'time' and data columns."
        )
    else:
        st.markdown(f"**Normalization Method:** {normalization_method}")
        st.info(
            f"Model RMSE: **{stats.model_rmse:.4f}** {status_badge(stats.model_pass)}"
        )

        fig_model = go.Figure()
        fig_model.add_trace(
            go.Scatter(
                x=t,
                y=norm_y,
                mode="lines",
                name="Measured (Norm)",
                line=dict(color="blue"),
            )
        )
        fig_model.add_trace(
            go.Scatter(
                x=t,
                y=norm_ref,
                mode="lines",
                name="Reference (Norm)",
                line=dict(color="red", dash="dash"),
            )
        )

        fig_model.update_layout(
            title=f"Model Comparison ({normalization_method})",
            xaxis_title="Time",
            yaxis_title="Normalized Value",
        )
        st.plotly_chart(fig_model, use_container_width=True)

        st.divider()
        st.subheader("Reference Points Check")

        # Explicit Metric Display
        ref_rmse_val = stats.ref_points_rmse
        ref_pass = stats.ref_points_pass
        st.info(f"Ref Points RMSE: **{ref_rmse_val:.2e}** {status_badge(ref_pass)}")

        # Plot Raw Trend vs User Targets
        fig_ref = go.Figure()
        fig_ref.add_trace(
            go.Scatter(
                x=t,
                y=smooth_trend,
                mode="lines",
                name="Smoothed Trend",
                line=dict(color="lime", width=2),
            )
        )

        if "ref_points" in st.session_state and isinstance(
            st.session_state["ref_points"], list
        ):
            # Parse list of dicts
            rp_x = []
            rp_y = []
            for pt in st.session_state["ref_points"]:
                try:
                    rp_x.append(float(pt.get("Time")))
                    rp_y.append(float(pt.get("Value")))
                except:
                    pass

            if rp_x:
                fig_ref.add_trace(
                    go.Scatter(
                        x=rp_x,
                        y=rp_y,
                        mode="markers",
                        name="User Targets",
                        marker=dict(
                            color="magenta", symbol="x", size=12, line=dict(width=2)
                        ),
                    )
                )

        fig_ref.update_layout(
            title="Smoothed Trend vs Reference Points",
            xaxis_title="Time",
            yaxis_title="Raw Signal",
        )
        st.plotly_chart(fig_ref, use_container_width=True)

    st.markdown("""
    **Methodology:**
    1.  **Interpolation:** Reference curve aligned to sensor timepoints.
    2.  **Normalization:** Both signals transformed using the selected method.
    3.  **Metric:** RMSE calculated on the difference.
    """)

# --- RE-CALCULATE METRICS & RENDER TOP PLOTS ---

# 1. Calculate Anomalies Mask (Union or Intersection)
if intersect_spikes:
    # Intersection Logic (Robust Window Search)
    # Use currently active Hampel mask (res_mask) for confirmation
    # This ensures we confirm against what is visually shown as "Hampel Spikes"
    target_mask = res_mask
    confirmed_mask = np.zeros_like(loop_mask)
    labeled_loops, num_features = label(loop_mask)
    n_points = len(loop_mask)

    for i in range(1, num_features + 1):
        event_indices = np.where(labeled_loops == i)[0]
        if len(event_indices) == 0:
            continue
        start_idx, end_idx = event_indices[0], event_indices[-1]
        s = max(0, start_idx - spike_proximity)
        e = min(n_points, end_idx + spike_proximity + 1)
        if np.any(target_mask[s:e]):
            confirmed_mask[event_indices] = True

    final_anomalies_mask = confirmed_mask
else:
    # Union Logic (Sum of both)
    final_anomalies_mask = loop_mask | res_mask

# Recalculate stats with updated masks
eff_thresh = (
    analysis_params["thresh_piecewise"]
    if (intersect_steps or step_method == "Piecewise Fit")
    else analysis_params["thresh_slope"]
)

stats = evaluate_metric(
    name=selected_sensor,
    values=y.values,  # Stats usually based on Raw Signal properties
    noise=noise_series.values,
    snr=snr_series.values,
    residuals=residuals,  # Pass residuals
    deriv_noise=noise_diff_series.values,  # Pass derivative noise
    anomalies_mask=final_anomalies_mask,  # Renamed arg, use final mask
    residual_mask=res_mask,  # Updated
    step_mask=step_mask,
    improvements=step_metric,  # Updated
    step_threshold=eff_thresh,
    model_rmse=stats.model_rmse,  # Preserve RMSE from initial analysis
    ref_points_rmse=stats.ref_points_rmse,  # Preserve Ref RMSE from initial analysis
    **grading_params,
)

# Render Metrics
with metrics_container:
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(9)
    c1.markdown(
        f"**Overall Score**<br>{stats.overall_score:.2f} {status_badge(stats.overall_pass)}",
        unsafe_allow_html=True,
        help="Weighted average of all health metrics. Pass if > 0.7.",
    )
    c2.markdown(
        f"**Noise Ratio**<br>{stats.noise_ratio:.2e} {status_badge(stats.noise_pass)}",
        unsafe_allow_html=True,
        help="Median Noise (MAD) divided by Signal Baseline amplitude.",
    )
    c3.markdown(
        f"**Median SNR**<br>{stats.snr_median:.1f} {status_badge(stats.snr_pass)}",
        unsafe_allow_html=True,
        help="Signal-to-Noise Ratio (Trend / Noise). Target > 5.0.",
    )
    c4.markdown(
        f"**Anomalies**<br>{(stats.anomalies_count / len(y)) * 100:.1f}% {status_badge(stats.anomalies_pass)}",
        unsafe_allow_html=True,
        help="Percentage of data points flagged as Spikes (Loop or Hampel).",
    )
    val_steps = f"{(stats.step_count / len(y)) * 100:.1f}%" if len(y) > 0 else "N/A"
    c5.markdown(
        f"**Step Events**<br>{val_steps} {status_badge(stats.steps_pass)}",
        unsafe_allow_html=True,
        help="Percentage of data points identified as Step transitions.",
    )
    c6.markdown(
        f"**Residual Spread**<br>{stats.signal_std:.2e} {status_badge(stats.std_pass)}",
        unsafe_allow_html=True,
        help="Standard Deviation of (Raw Signal - Trend). Measures total jitter.",
    )
    c7.markdown(
        f"**Deriv Noise**<br>{stats.deriv_noise_median:.2e} {status_badge(stats.deriv_pass)}",
        unsafe_allow_html=True,
        help="Stability of the signal's rate of change (slope noise).",
    )
    c8.markdown(
        f"**Model RMSE**<br>{stats.model_rmse:.2f} {status_badge(stats.model_pass)}",
        unsafe_allow_html=True,
        help="Root Mean Square Error vs Normalized Reference Curve.",
    )
    c9.markdown(
        f"**Ref Points RMSE**<br>{stats.ref_points_rmse:.2e} {status_badge(stats.ref_points_pass)}",
        unsafe_allow_html=True,
        help="Root Mean Square Error vs User-defined Raw Reference Points.",
    )
# Render Main Plot
with plot_container:
    fig_main = go.Figure()

    # Always show Raw (Blue) and Trend (Green)
    fig_main.add_trace(
        go.Scatter(
            x=t,
            y=y,
            mode="lines",
            name="Raw Signal",
            line=dict(color="royalblue", width=1),
        )
    )
    fig_main.add_trace(
        go.Scatter(
            x=t,
            y=smooth_trend,
            mode="lines",
            name="Trend",
            line=dict(color="lime", width=2),
        )
    )

    # Determine Y-values for markers
    y_anom_plot = smooth_trend if use_trend_anom else y
    y_step_plot = smooth_trend if use_trend_step else y

    if intersect_spikes:
        # Plot ONLY the Confirmed Spikes (Intersection)
        # Reduce to single peaks for cleaner plot
        if final_anomalies_mask.any():
            confirmed_peaks_mask = np.zeros_like(final_anomalies_mask)
            labeled_confirmed, n_confirmed = label(final_anomalies_mask)

            for i in range(1, n_confirmed + 1):
                event_idx = np.where(labeled_confirmed == i)[0]
                event_dev = np.abs(y.values[event_idx] - smooth_trend.values[event_idx])
                peak_local_idx = np.argmax(event_dev)
                peak_global_idx = event_idx[peak_local_idx]
                confirmed_peaks_mask[peak_global_idx] = True

            # Single Trace for all confirmed spikes
            fig_main.add_trace(
                go.Scatter(
                    x=t[confirmed_peaks_mask],
                    y=y_anom_plot[confirmed_peaks_mask],
                    mode="markers",
                    name="Confirmed Spikes",
                    marker=dict(
                        color="cyan",
                        symbol="star",
                        size=14,
                        line=dict(width=1, color="black"),
                    ),
                )
            )

            # Debug Target (optional, maybe remove for clean look?)
            # fig_main.add_trace(go.Scatter(x=t[raw_res_mask], y=y[raw_res_mask], mode='markers', marker=dict(color='rgba(255,165,0,0.2)', symbol='x'), name='(Debug) Hampel'))

    else:
        # Plot BOTH (Union) Separately
        if loop_mask.any():
            fig_main.add_trace(
                go.Scatter(
                    x=t[loop_mask],
                    y=y_anom_plot[loop_mask],
                    mode="markers",
                    name="Loop Spikes",
                    marker=dict(
                        color="red", symbol="circle-open", size=8, line=dict(width=2)
                    ),
                )
            )
        if res_mask.any():
            fig_main.add_trace(
                go.Scatter(
                    x=t[res_mask],
                    y=y_anom_plot[res_mask],
                    mode="markers",
                    name="Hampel Spikes",
                    marker=dict(color="orange", symbol="x", size=6),
                )
            )

    if step_mask.any():
        fig_main.add_trace(
            go.Scatter(
                x=t[step_mask],
                y=y_step_plot[step_mask],
                mode="markers",
                name="Step Detect",
                marker=dict(color="magenta", symbol="diamond", size=6),
            )
        )

    fig_main.update_layout(
        title=f"Overview: {selected_sensor}",
        xaxis_title="Time",
        yaxis_title="Signal",
        height=500,
    )
    st.plotly_chart(fig_main, use_container_width=True)

    if save_plots:
        fig_main.write_html(f"{OUTPUT_DIR}/{selected_sensor}_main.html")
