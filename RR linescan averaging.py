import streamlit as st
import pandas as pd
import numpy as np
import base64
import re
import matplotlib.pyplot as plt

def round_to_nearest(value, step):
    """Round value to the nearest multiple of 'step'."""
    return round(value / step) * step

def extract_scan_number(filename):
    """Extract the line scan number from the filename, assuming it's after 'LS'."""
    match = re.search(r'LS(\d+)', filename)
    if match:
        return f"scan_{match.group(1)}"
    return filename  # Default to the filename if no match is found

def main():
    st.title("Line Scan Averager")
    st.write(
        "Upload multiple CSV files containing two columns: Distance (X) and KAM value (Y). "
        "The app will align the scans by X values, leave missing data as blank, and compute an average using only existing points."
    )

    uploaded_files = st.file_uploader(
        "Select one or more CSV files", type="csv", accept_multiple_files=True
    )
    if not uploaded_files:
        st.info("Awaiting CSV file uploads...")
        return

    scans = []
    all_x_values = set()
    scan_labels = []  # List to store scan labels

    for uploaded_file in uploaded_files:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read {uploaded_file.name}: {e}")
            return

        if df.shape[1] < 2:
            st.error(f"File {uploaded_file.name} does not have at least two columns.")
            return

        # Extract the scan label from the filename
        scan_label = extract_scan_number(uploaded_file.name)
        scan_labels.append(scan_label)

        # Assume first column is X, second is Y
        df = df.iloc[:, :2].dropna()
        df.columns = ["X", "Y"]

        # Replace 0 values in Y with NaN
        df["Y"] = df["Y"].replace(0, np.nan)

        # Round the X values to the nearest 0.2 µm
        df["X"] = df["X"].apply(lambda x: round_to_nearest(x, 0.2))

        # Drop duplicates for the current scan (keep first occurrence)
        df = df.drop_duplicates(subset="X", keep="first")

        # Add the X values to the global set of X values
        all_x_values.update(df["X"])

        # Sort and set X as the index
        df = df.sort_values("X").set_index("X")

        scans.append(df["Y"])

    # Create the unified list of X values from all files
    all_x = sorted(list(all_x_values))

    # Reindex each scan onto the unified X grid, leaving missing values as NaN
    reindexed = [scan.reindex(all_x) for scan in scans]

    # Combine into DataFrame for easier averaging
    combined = pd.concat(reindexed, axis=1)
    combined.columns = scan_labels  # Use scan labels for the columns

    # Compute mean across rows, ignoring NaNs
    combined["Y_mean"] = combined.mean(axis=1, skipna=True)

    result_df = combined.reset_index().rename(columns={"index": "X"})

    # --- Baseline and Recovery Point Calculation ---
    st.subheader("Baseline and Recovery Analysis")
    tail_section = result_df[result_df["X"] >= result_df["X"].max() - 50]  # Last 50 µm
    baseline_mean = tail_section["Y_mean"].mean()
    baseline_std = tail_section["Y_mean"].std()

    threshold = baseline_mean + baseline_std
    recovery_df = result_df[result_df["Y_mean"] <= threshold]
    recovery_point = recovery_df["X"].min() if not recovery_df.empty else np.nan

    st.markdown(f"**Baseline Mean:** {baseline_mean:.4f}")
    st.markdown(f"**Baseline Std Dev:** {baseline_std:.4f}")
    st.markdown(f"**Recovery Point (X when Y_mean <= baseline + 1 std):** {recovery_point:.2f} µm")

    # --- Plot with Annotations ---
    st.subheader("Averaged Line Scan")
    fig, ax = plt.subplots()
    ax.plot(result_df["X"], result_df["Y_mean"], label="Y_mean", color='blue')
    ax.axhline(baseline_mean, color='green', linestyle='--', label="Baseline")
    ax.axhline(threshold, color='red', linestyle='--', label="Baseline + 1 Std Dev")
    if not np.isnan(recovery_point):
        ax.axvline(recovery_point, color='orange', linestyle=':', label="Recovery Point")
    ax.set_xlabel("Distance (X)")
    ax.set_ylabel("KAM Value (Y_mean)")
    ax.set_title("Averaged Line Scan with Baseline and Recovery Point")
    ax.legend()
    st.pyplot(fig)

    # --- Table and Download ---
    st.dataframe(result_df)

    csv = result_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = (
        f'<a href="data:file/csv;base64,{b64}" '
        'download="averaged_line_scan.csv">Download averaged CSV</a>'
    )
    st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
