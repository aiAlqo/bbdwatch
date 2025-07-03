import pandas as pd
import numpy as np
import io

def assess_status(row):
    """
    Determine the stock status of an SKU based on SOH, Min Qty, and Max Qty.
    """
    soh = row.get('SOH', np.nan)
    min_qty = row.get('Min Qty', np.nan)
    max_qty = row.get('Max Qty', np.nan)

    if pd.isna(soh):
        return "❓ Missing SOH"
    elif soh < min_qty:
        return "🔴 Critical!!! Below Min Qty"
    elif soh < max_qty:
        return "🟠 Reorder Level"
    elif soh > max_qty:
        return "🕣 Overstocked"
    else:
        return "✅ Healthy"


def suggest_reorder(row):
    """
    Suggest reorder qty if stock is below Min Qty.
    """
    soh = row.get('SOH', np.nan)
    min_qty = row.get('Min Qty', np.nan)
    max_qty = row.get('Max Qty', np.nan)
    moq = row.get('MOQ', 0)
    minor_mult = row.get('Minor Order Multiple', 1)
    major_mult = row.get('Major Order Multiple', 1)
    max_order_qty = row.get('Max Order Qty', np.nan)

    if pd.isna(soh) or pd.isna(min_qty) or pd.isna(max_qty):
        return np.nan

    if soh >= min_qty:
        return 0

    reorder_qty = max_qty - soh

    # Round up to match Minor/ Major Order Multiples
    reorder_qty = max(reorder_qty, moq)
    reorder_qty = int(np.ceil(reorder_qty / minor_mult) * minor_mult)
    reorder_qty = int(np.ceil(reorder_qty / major_mult) * major_mult)

    if not pd.isna(max_order_qty):
        reorder_qty = min(reorder_qty, max_order_qty)

    return reorder_qty


def highlight_row(row):
    """
    Return CSS style for row based on Status column.
    """
    status = row.get("Status", "")
    color_map = {
        "🔴 Critical!!! Below Min Qty": "background-color: #F08080;",  # light red
        "🟠 Reorder Level": "background-color: #FFD580;",             # light orange
        "🕣 Overstocked": "background-color: #D8BFD8;",               # light purple
        "✅ Healthy": "background-color: #90EE90;",                   # light green
        "❓ Missing SOH": "background-color: #D3D3D3;",               # light gray
    }
    return [color_map.get(status, "")] * len(row)


def simulate_runout(row, forecast_cols):
    """
    For each forecast period, calculate remaining stock after usage.
    """
    remaining = row.get('SOH', 0)
    runout = []
    for col in forecast_cols:
        usage = row.get(col, 0)
        if pd.isna(usage):
            usage = 0
        remaining -= usage
        runout.append(max(remaining, 0))
    return pd.Series(runout, index=forecast_cols)


def highlight_forecast(val):
    """
    Highlight forecast cells where stock runs out.
    """
    if val <= 0:
        return 'background-color: #FFB6C1;'  # light pink
    return ''


def get_row_fill_color(status):
    """
    Optional helper if you want to add fill colors elsewhere.
    """
    color_map = {
        "🔴 Critical!!! Below Min Qty": "#D44444",
        "🟠 Reorder Level": "#FF9148",
        "🕣 Overstocked": "#7B4FB6",
        "✅ Healthy": "#8CDF8C",
        "❓ Missing SOH": "#B0B0B0",
    }
    return color_map.get(status, "#FFFFFF")


def generate_excel(df):
    """
    Generate an Excel file in memory with DataFrame content.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Inventory')
        workbook = writer.book
        worksheet = writer.sheets['Inventory']
        # Example: you could apply some workbook formatting here
    output.seek(0)
    return output
