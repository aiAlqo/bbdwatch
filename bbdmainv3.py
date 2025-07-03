# inventory_status_colored.py (with forecast runout simulation)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from inventory_utils import (
    get_row_fill_color, generate_excel, assess_status, suggest_reorder,
    highlight_row, simulate_runout, highlight_forecast
)

st.set_page_config(page_title="Inventory Health Check", layout="wide")
st.title("📦 SOH Watch")

# ---- Configurable forecast column suffix
FORECAST_SUFFIX = "-25"  # Change this if your forecast columns use a different suffix

st.info(f"Forecast simulation uses columns ending with '{FORECAST_SUFFIX}'. Adjust your file or the app setting if needed.")

# ---- File Upload
uploaded_file = st.file_uploader("Upload your inventory file", type=["csv", "xlsx"])

# Add Purchase Order Report uploader
uploaded_po_file = st.file_uploader("Upload your Purchase Order Report (optional)", type=["csv", "xlsx"], key="po_report")

# Display uploaded file names for reference
if uploaded_file:
    st.success(f"Inventory file uploaded: {uploaded_file.name}")
    if uploaded_po_file:
        st.info(f"Purchase Order Report uploaded: {uploaded_po_file.name}")
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # 🟢 Fix column name spacing
    df.columns = df.columns.str.strip()

    required_cols = [
        "SKU Code", "SKU Description", "SKU Category", "Site", "Source",
        "SOH", "Safety Stock", "Min Qty", "Max Qty",
        "MOQ", "Max Order Qty", "Minor Order Multiple", "Major Order Multiple"
    ]

    string_cols = ["SKU Code", "SKU Description", "SKU Category", "Site", "Source"]
    numeric_cols = ["SOH", "Safety Stock", "Min Qty", "Max Qty", "MOQ", "Max Order Qty", "Minor Order Multiple", "Major Order Multiple"]

    if not all(col in df.columns for col in required_cols):
        st.error("❌ Missing one or more required columns.")
        st.stop()

    for col in string_cols:
        df[col] = df[col].astype(str).fillna("").str.strip()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ---- Filters
    st.sidebar.header("🔍 Filter Options")
    selected_site = st.sidebar.multiselect("Site", df['Site'].unique(), default=df['Site'].unique())
    selected_cat = st.sidebar.multiselect("SKU Category", df['SKU Category'].unique(), default=df['SKU Category'].unique())
    selected_source = st.sidebar.multiselect("Source", df['Source'].unique(), default=df['Source'].unique())

    df = df[(df['Site'].isin(selected_site)) &
            (df['SKU Category'].isin(selected_cat)) &
            (df['Source'].isin(selected_source))]

    # ---- Business Logic
    df["Status"] = df.apply(assess_status, axis=1)
    df["Suggested Reorder Qty"] = df.apply(suggest_reorder, axis=1)

    # Base display columns without PO info
    display_cols = ["SKU Code", "SKU Description", "SKU Category", "Site", "Source", "SOH", "Status", "Suggested Reorder Qty"]

    # ---- Purchase Order Integration ----
    po_info = None
    if uploaded_po_file:
        if uploaded_po_file.name.endswith('.csv'):
            df_po = pd.read_csv(uploaded_po_file)
        else:
            df_po = pd.read_excel(uploaded_po_file)
        # Clean up PO columns
        df_po.columns = df_po.columns.str.strip()  # Remove leading/trailing spaces from column names
        df_po['SKU Code'] = df_po['SKU Code'].astype(str).str.strip()
        # Parse Expected Delivery Date with explicit format, fallback to generic if needed
        try:
            df_po['Expected Delivery Date'] = pd.to_datetime(df_po['Expected Delivery Date'], format='%d/%m/%Y', errors='coerce')
        except Exception:
            df_po['Expected Delivery Date'] = pd.to_datetime(df_po['Expected Delivery Date'], errors='coerce')
        df_po['Order Qty'] = pd.to_numeric(df_po['Order Qty'], errors='coerce')
        # Drop rows with missing key info
        df_po = df_po.dropna(subset=['SKU Code', 'Expected Delivery Date'])
        # Only keep relevant columns
        df_po = df_po[['SKU Code', 'Order Qty', 'Expected Delivery Date']]
        # For each SKU, find the earliest PO arrival
        po_next_arrival = df_po.groupby('SKU Code')['Expected Delivery Date'].min().to_dict()
        po_next_qty = df_po.groupby('SKU Code')['Order Qty'].sum().to_dict()
        # Estimate runout date using forecast columns if available
        forecast_cols = [col for col in df.columns if col.endswith(FORECAST_SUFFIX)]
        def estimate_runout(row):
            if not forecast_cols or row['SOH'] <= 0:
                return np.nan
            remaining = row['SOH']
            for i, col in enumerate(forecast_cols):
                usage = row.get(col, 0)
                if pd.isna(usage):
                    continue
                remaining -= usage
                if remaining <= 0:
                    # Assume each forecast col is 1 period (e.g., week)
                    return float(i)  # periods until runout as float
            return np.nan
        df['Runout Period'] = df.apply(estimate_runout, axis=1)
        # Map PO info to inventory
        def get_next_po(row):
            return po_next_arrival.get(row['SKU Code'], None)
        def get_next_po_qty(row):
            return po_next_qty.get(row['SKU Code'], None)
        df['Next PO Arrival'] = df.apply(get_next_po, axis=1)
        df['Next PO Qty'] = df.apply(get_next_po_qty, axis=1)
        # Determine if PO mitigates OOS
        def mitigates_oos(row):
            if pd.isna(row['Runout Period']) or pd.isna(row['Next PO Arrival']):
                return 'N/A'
            # Assume forecast period is 1 week, and first forecast col is next week
            from datetime import datetime, timedelta
            today = pd.Timestamp.today().normalize()
            runout_date = today + pd.Timedelta(weeks=float(row['Runout Period']))
            if row['Next PO Arrival'] <= runout_date:
                return 'Yes'
            else:
                return 'No'
        df['PO Mitigates OOS?'] = df.apply(mitigates_oos, axis=1)
        
        # Add PO columns to display only if PO file is uploaded
        display_cols.extend(['Next PO Arrival', 'PO Mitigates OOS?'])

    # ---- Pie Chart
    status_colors = {
        "🔴 Critical!!! Below Min Qty": "#D44444",
        "🟠 Reorder Level": "#FF9148",
        "🕣 Overstocked": "#7B4FB6",
        "✅ Healthy": "#8CDF8C",
        "❓ Missing SOH": "#B0B0B0"  # Changed to gray for clarity
    }

    st.subheader("📊 Inventory Status Distribution")
    fig = px.pie(df, names="Status", title="Status Summary", color="Status", color_discrete_map=status_colors)
    st.plotly_chart(fig, use_container_width=True)

    # ---- Table Styling
    df_display = df[display_cols]

    st.subheader("📦 Inventory Table with Color Coding")
    table_height = min(600, 40 + 30 * len(df_display))  # Dynamic height
    st.dataframe(
        df_display.style.apply(highlight_row, axis=1),
        use_container_width=True,
        height=table_height
    )

    # ---- Forecast Coverage Simulation ----
    forecast_cols = [col for col in df.columns if col.endswith(FORECAST_SUFFIX)]
    if forecast_cols:
        df_coverage = df[["SKU Code", "SOH"] + forecast_cols].copy()
        df_coverage[forecast_cols] = df_coverage.apply(lambda row: simulate_runout(row, forecast_cols), axis=1)
        st.subheader("📆 Forecast Coverage Simulation")
        st.dataframe(
            df_coverage.style.applymap(highlight_forecast, subset=forecast_cols),
            use_container_width=True
        )
    else:
        st.warning(f"No forecast columns found ending with '{FORECAST_SUFFIX}'. Forecast simulation is skipped.")

    # ---- Downloads ----
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📅 Download as CSV",
            data=df_display.to_csv(index=False),
            file_name="inventory_status.csv",
            mime="text/csv"
        )
    with col2:
        excel_file = generate_excel(df_display)
        st.download_button(
            label="📊 Download as Excel",
            data=excel_file,
            file_name="inventory_status_colored.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("👆 Upload a file to begin. Ensure all required columns are present.")
    # Welcome screen with instructions
    st.markdown("""
    ## Welcome to SOH Watch Dashboard! 📦
    This dashboard helps you monitor and analyze SOH status to ensure continuity and optimize inventory management.
    ### 📋 Required CSV Format:
    Your CSV file should contain the following columns:
    - **SKU Code**: Stock Keeping Unit Code
    - **SKU Description**: Stock Keeping Unit Description
    - **SKU Category**: Stock Keeping Unit category
    - **Site**: Site name
    - **Source**: Source name
    - **SOH**: Qty On Hand
    - **Safety Stock**: Safety Stock
    - **Min Qty**: Minimum Quantity
    - **Max Qty**: Maximum Quantity
    - **MOQ**: Minimum Order Quantity
    - **Max Order Qty**: Maximum Order Quantity
    - **Minor Order Multiple**: Minor Order Multiple
    - **Major Order Multiple**: Major Order Multiple
    """)
    # Add sample data structure
    st.subheader("📄 Sample Data Structure")
    sample_data = pd.DataFrame({
        'SKU Code': ['SKU001', 'SKU002', 'SKU003'],
        'SKU Description': ['Sample Item 1', 'Sample Item 2', 'Sample Item 3'],
        'SKU Category': ['Category A', 'Category B', 'Category A'],
        'Site': ['Factory Name 1', 'Distribution Center 1', 'Factory Name 2'],
        'Source': ['Supplier 1', 'Supplier 2', 'Supplier 3'],
        'SOH': [100, 50, 200],
        'Safety Stock': [20, 10, 30],
        'Min Qty': [50, 30, 100],
        'Max Qty': [200, 100, 300],
        'MOQ': [10, 10, 20],
        'Max Order Qty': [500, 200, 600],
        'Minor Order Multiple': [5, 5, 10],
        'Major Order Multiple': [20, 20, 40]
    })
    st.dataframe(sample_data, use_container_width=True)
