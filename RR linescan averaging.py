import streamlit as st
import pandas as pd
import numpy as np
import base64
import re
import altair as alt

def round_to_nearest(value, step):
    return round(value / step) * step

def extract_scan_number(filename):
    match = re.search(r'LS(\d+)', filename)
    if match:
        return f"scan_{match.group(1)}"
    return filename

def main():
    st.title("Line Scan Averager")
    st.write(
        "Upload multiple CSV files containing two columns: Distance (X) and KAM value (Y). "
        "The app will align the scans by X values, trim to average scan length, and compute an average."
    )

    uploaded_files = st.file_uploader(
        "Select one or more CSV files", type="csv", accept_multiple_files=True
    )
    if not uploaded_files:
        st.info("Awaiting CSV file uploads...")
        return

    scans = []
    all_x_values = set()
    scan_labels = []
    max_x_values = []

    for uploaded_file in uploaded_files:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read {uploaded_file.name}: {e}")
            return

        if df.shape[1] < 2:
            st.error(f"File {uploaded_file.name} does not have at least two columns.")
            st.write("Detected columns:", df.columns.tolist())
            return

        scan_label = extract_scan_number(uploaded_file.name)
        scan_labels.append(scan_label)

        df = df.iloc[:, :2].dropna()
        df.columns = ["X", "Y"]

        # Ensure X is numeric
        df["X"] = pd.to_numeric(df["X"], errors="coerce")
        df = df.dropna(subset=["X"])

        # Replace 0 in Y with NaN
        df["Y"] = df["Y"].replace(0, np.nan)

        # Round X values
        df["X"] = df["X"].apply(lambda x: round_to_nearest(x, 0.2))

        # Drop duplicates
        df = df.drop_duplicates(subset="X", keep="first")

        # Sort and set index
        df = df.sort_values("X").set_index("X")

        # Get max X value
        max_x = df.index.max()
        if pd.notna(max_x):
            max_x_values.append(max_x)
        else:
            st.warning(f"Could not determine max X for {uploaded_file.name}")
            continue

        scans.append(df["Y"])
        all_x_values.update(df.index)

    if not max_x_values:
        st.error("No valid X data found in any file.")
        return

    average_max_x = round_to_nearest(np.mean(max_x_values), 0.2)
    st.markdown(f"**Average Maximum X (Trim Limit):** {average_max_x:.2f} µm")

    all_x = sorted([x for x in all_x_values if x <= average_max_x])
    reindexed = [scan.reindex(all_x) for scan in scans]

    combined = pd.concat(reindexed, axis=1)
    combined.columns = scan_labels
    combined["Y_mean"] = combined.mean(axis=1, skipna=True)
    result_df = combined.reset_index().rename(columns={"index": "X"})

    # --- Deformation Analysis ---
    st.subheader("Deformation Zone Analysis")
    tail_section = result_df[result_df["X"] >= result_df["X"].max() - 30]
    baseline_mean = tail_section["Y_mean"].mean()
    baseline_std = tail_section["Y_mean"].std()
    threshold = baseline_mean + baseline_std

    recovery_df = result_df[result_df["Y_mean"] <= threshold]
    recovery_point = recovery_df["X"].min() if not recovery_df.empty else np.nan

    st.markdown(f"**Baseline Mean:** {baseline_mean:.4f}")
    st.markdown(f"**Baseline Std Dev:** {baseline_std:.4f}")
    st.markdown(f"**Deformation Zone Size:** {recovery_point:.2f} µm")

    # --- Plot ---
    st.subheader("Averaged Line Scan")
    base = alt.Chart(result_df).mark_line(color='blue').encode(
        x=alt.X('X', title='Distance (µm)'),
        y=alt.Y('Y_mean', title='KAM (°)')
    )

    baseline_line = alt.Chart(pd.DataFrame({'Y': [baseline_mean]})).mark_rule(color='green', strokeDash=[5,5]).encode(y='Y')
    threshold_line = alt.Chart(pd.DataFrame({'Y': [threshold]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='Y')
    recovery_marker = alt.Chart(pd.DataFrame({'X': [recovery_point]})).mark_rule(color='orange', strokeDash=[2,2]).encode(x='X')

    chart = (base + baseline_line + threshold_line + recovery_marker).properties(
        width=700,
        height=400,
        title="Averaged Line Scan with Baseline, Std Dev and Recovery Point"
    )

    st.altair_chart(chart, use_container_width=True)

    # --- Output ---
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
