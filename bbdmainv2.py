# streamlit_app_prototype.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Best Before Date Analysis", layout="wide")
st.title("📦 BBD Watch Dashboard")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Settings")
wri_threshold_1 = st.sidebar.slider("Low Risk Threshold (Days)", 0, 365, 180)
wri_threshold_2 = st.sidebar.slider("Medium Risk Threshold (Days)", 0, 365, 90)

uploaded_file = st.file_uploader("Upload your CSV file (including BBD, Qty, Cost)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    with st.expander("✅ File uploaded successfully!"):
        st.write(df.head())

    try:
        df['BBD'] = pd.to_datetime(df['BBD'], errors='coerce', dayfirst=True)
        df['Qty On Hand'] = df['Qty On Hand'].astype(str).str.replace(',', '').astype(float)
        df['UnitCost'] = pd.to_numeric(df['UnitCost'], errors='coerce')
        df['Value'] = df['Qty On Hand'] * df['UnitCost']
        today = pd.to_datetime(datetime.today())
        df['DaysToExpiry'] = (df['BBD'] - today).dt.days
    except KeyError as e:
        st.error(f"Missing expected column: {e}")

    def stage(days):
        if pd.isna(days): return "Unknown"
        elif days > wri_threshold_1: return "Stage 1"
        elif days > wri_threshold_2: return "Stage 2"
        elif days >= 0: return "Stage 3"
        else: return "Expired"

    df['Stage'] = df['DaysToExpiry'].apply(stage)

    # Waste Risk Index
    def compute_wri(row):
        if pd.isna(row['DaysToExpiry']) or pd.isna(row['Qty On Hand']):
            return 0
        decay_factor = np.exp(-row['DaysToExpiry'] / 90) if row['DaysToExpiry'] >= 0 else 1.5
        return round(min(100, decay_factor * 100), 1)

    df['WRI'] = df.apply(compute_wri, axis=1)

    def risk_category(wri):
        if wri > 75: return "High"
        elif wri > 50: return "Medium"
        else: return "Low"

    df['Risk Level'] = df['WRI'].apply(risk_category)

    # --- KPIs ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    total_value = df['Value'].sum()
    expired_value = df[df['Stage'] == 'Expired']['Value'].sum()
    high_risk_count = df[df['Risk Level'] == 'High'].shape[0]
    avg_days = df['DaysToExpiry'].mean()

    kpi1.metric("💰 Total Stock Value", f"K {total_value:,.0f}")
    kpi2.metric("⚠️ Expired Stock Value", f"K {expired_value:,.0f}")
    kpi3.metric("🔥 High Risk SKUs", f"{high_risk_count}")
    kpi4.metric("📅 Avg Days to Expiry", f"{avg_days:.0f} days")

    # --- Stage Summary Table ---
    stage_colors = {'Stage 1': '#2e7d32', 'Stage 2': '#a5d6a7', 'Stage 3': '#ffb74d', 'Expired': '#e57373'}
    stage_order = ['Stage 1', 'Stage 2', 'Stage 3', 'Expired']
    stage_summary = df.groupby('Stage')['Value'].sum().reindex(stage_order, fill_value=0).reset_index()

    st.subheader("Shelf life stage analysis report")
    with st.expander("View shelf life stage analysis"):
        stages = {"Stage 1": "Stage 1 - Stocks beyond 6 month shelf life\n181+ days", "Stage 2": "Stage 2 - Stocks between 3 to 6 months shelf life\n91 - 180 days", "Stage 3": "Stage 3 - Stocks below 3 months Shelf life\n0 - 90 days", "Expired": "Stage 4 - Stocks passed BBD Shelf life\n - 0 < days"}

        for stage_key, stage_label in stages.items():
            st.markdown(f"### {stage_label}")
            filtered = df[df['Stage'] == stage_key]

            if not filtered.empty:
                summary = filtered.groupby('Category')[['Qty On Hand', 'Value']].sum().reset_index()
                total = summary[['Qty On Hand', 'Value']].sum().to_frame().T
                total['Category'] = 'Total'
                summary = pd.concat([summary, total], ignore_index=True)

                # Format numbers
                summary['Qty On Hand'] = summary['Qty On Hand'].round(0).astype(int).map('{:,}'.format)
                summary['Value'] = summary['Value'].round(0).astype(int).map('{:,}'.format)

                summary = summary.rename(columns={
                    'Category': 'Category',
                    'Qty On Hand': 'Qty',
                    'Value': 'Value'
                })[['Category', 'Qty', 'Value']]

                st.dataframe(summary, use_container_width=True)
            else:
                st.write("No data in this shelf life stage.")

    col1, col2 = st.columns(2)
    with col1:
        fig_pie = px.pie(stage_summary, values='Value', names='Stage', title='Stock Value by Stage', color='Stage', color_discrete_map=stage_colors)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_bar = px.bar(stage_summary, x='Stage', y='Value', text_auto='.2s', title='Total Value by Expiry Stage', color='Stage', color_discrete_map=stage_colors)
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- WRI Trend Forecast ---
    trend = df[df['DaysToExpiry'] >= 0].copy()
    trend['Month'] = trend['BBD'].dt.to_period('M').dt.to_timestamp()
    trend_summary = trend.groupby('Month')['Value'].sum().reset_index()

    st.subheader("📈 Expiry Forecast Value by Month")
    fig_trend = px.line(trend_summary, x='Month', y='Value', markers=True)
    st.plotly_chart(fig_trend, use_container_width=True)

    # --- WRI Summary Table ---
    st.subheader("📊 Waste Risk Index Summary")
    risk_summary = df.groupby('Risk Level')[['Qty On Hand', 'Value']].sum().reset_index()
    st.dataframe(risk_summary, use_container_width=True)

    # --- Detailed View ---
    st.subheader("📋 Expiry Detail Table with WRI")
    display_cols = ['Stock Code', 'Description', 'Category', 'Whse', 'Lot', 'BBD', 'Qty On Hand', 'UnitCost', 'Value', 'DaysToExpiry', 'Stage', 'WRI', 'Risk Level']
    st.dataframe(df[display_cols].sort_values(by='DaysToExpiry'), use_container_width=True)

    # --- Materials Transitioning ---
    st.subheader("⏳ Upcoming Stage Transitions")
    def days_to_transition(days):
        if pd.isna(days): return None
        if days > wri_threshold_1 and days <= wri_threshold_1 + 30: return wri_threshold_1
        elif days > wri_threshold_2 and days <= wri_threshold_2 + 30: return wri_threshold_2
        elif days >= 0 and days <= 30: return 0
        else: return None

    transition_df = df[df['DaysToExpiry'].between(0, wri_threshold_1 + 30)].copy()
    transition_df['NextStageBoundary'] = transition_df['DaysToExpiry'].apply(days_to_transition)
    transition_df = transition_df[~transition_df['NextStageBoundary'].isna()]
    transition_df['DaysToNextStage'] = transition_df['DaysToExpiry'] - transition_df['NextStageBoundary']

    def get_next_stage_label(stage):
        if stage == 'Stage 1': return '→ Stage 2'
        elif stage == 'Stage 2': return '→ Stage 3'
        elif stage == 'Stage 3': return '→ Expired'
        else: return '—'

    transition_df['Upcoming Stage'] = transition_df['Stage'].apply(get_next_stage_label)

    with st.expander("📋 View Materials Transitioning Soon"):
        transition_cols = ['Stock Code', 'Description', 'Whse', 'BBD', 'Qty On Hand', 'UnitCost', 'Value', 'DaysToExpiry', 'Stage', 'Upcoming Stage', 'DaysToNextStage']
        st.dataframe(transition_df[transition_cols].sort_values(by='DaysToNextStage'), use_container_width=True)

else:
    st.warning("⚠️ Please upload a CSV file to proceed.")
