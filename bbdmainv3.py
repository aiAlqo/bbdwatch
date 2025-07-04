# streamlit_app_prototype.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import json
import requests
import os
import pickle
from pathlib import Path
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Best Before Date Analysis", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #8a90a0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert {
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Constants and configuration
STAGE_COLORS = {
    'Stage 1': '#2e7d32', 
    'Stage 2': '#a5d6a7', 
    'Stage 3': '#ffb74d', 
    'Expired': '#e57373',
}

STAGE_ORDER = ['Stage 1', 'Stage 2', 'Stage 3', 'Expired']

REQUIRED_COLUMNS = ['BBD', 'Qty On Hand', 'UnitCost', 'Stock Code', 'Description', 'Category']
OPTIONAL_COLUMNS = ['Batch', 'Batch No', 'Location', 'Site', 'MFG Date', 'Manufacture Date']

# LLM Configuration
LLM_PROVIDERS = {
    "OpenAI GPT": "openai",
    "Local Ollama": "ollama", 
    "Hugging Face": "huggingface",
    "Custom API": "custom"
}

@st.cache_data
def load_and_validate_data(uploaded_file):
    """Load and validate uploaded CSV data"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check for required columns
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            st.info("Required columns: BBD, Qty On Hand, UnitCost, Stock Code, Description, Category")
            return None
            
        return df
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        return None

@st.cache_data
def process_data(df, wri_threshold_1, wri_threshold_2):
    """Process and calculate all derived fields"""
    try:
        # Data cleaning and type conversion
        df_processed = df.copy()
        
        # Handle BBD dates with multiple formats
        df_processed['BBD'] = pd.to_datetime(df_processed['BBD'], errors='coerce', dayfirst=True)
        
        # Clean quantity data
        df_processed['Qty On Hand'] = (
            df_processed['Qty On Hand']
            .astype(str)
            .str.replace(',', '')
            .str.replace(' ', '')
            .astype(float)
        )
        
        # Clean cost data
        df_processed['UnitCost'] = pd.to_numeric(df_processed['UnitCost'], errors='coerce')
        
        # Calculate derived fields
        df_processed['Value'] = df_processed['Qty On Hand'] * df_processed['UnitCost']
        today = pd.to_datetime(datetime.today())
        df_processed['DaysToExpiry'] = (df_processed['BBD'] - today).dt.days
        
        # Calculate stage
        def stage(days):
            if pd.isna(days): return "Unknown"
            elif days > wri_threshold_1: return "Stage 1"
            elif days > wri_threshold_2: return "Stage 2"
            elif days >= 0: return "Stage 3"
            else: return "Expired"
        
        df_processed['Stage'] = df_processed['DaysToExpiry'].apply(stage)
        
        # Calculate WRI (Waste Risk Index)
        def compute_wri(row):
            if pd.isna(row['DaysToExpiry']) or pd.isna(row['Qty On Hand']):
                return 0
            decay_factor = np.exp(-row['DaysToExpiry'] / 90) if row['DaysToExpiry'] >= 0 else 1.5
            return round(min(100, decay_factor * 100), 1)
        
        df_processed['WRI'] = df_processed.apply(compute_wri, axis=1)
        
        # Risk categorization
        def risk_category(wri):
            if wri > 75: return "High"
            elif wri > 50: return "Medium"
            else: return "Low"
        
        df_processed['Risk Level'] = df_processed['WRI'].apply(risk_category)
        
        return df_processed
        
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        return None

def format_currency(value):
    """Format currency values with proper formatting"""
    if pd.isna(value):
        return "N/A"
    return f"K {value:,.0f}"

def format_number(value):
    """Format numbers with proper separators"""
    if pd.isna(value):
        return "N/A"
    return f"{value:,.0f}"

def create_kpi_metrics(df):
    """Create enhanced KPI metrics with better formatting"""
    total_value = df['Value'].sum()
    expired_value = df[df['Stage'] == 'Expired']['Value'].sum()
    high_risk_count = df[df['Risk Level'] == 'High'].shape[0]
    avg_days = df['DaysToExpiry'].mean()
    
    # Additional metrics
    total_skus = len(df)
    expired_skus = len(df[df['Stage'] == 'Expired'])
    critical_skus = len(df[df['DaysToExpiry'].between(0, 30)])
    
    return {
        'total_value': total_value,
        'expired_value': expired_value,
        'high_risk_count': high_risk_count,
        'avg_days': avg_days,
        'total_skus': total_skus,
        'expired_skus': expired_skus,
        'critical_skus': critical_skus
    }

@st.cache_data
def generate_llm_insights(chart_data, chart_type, df_summary):
    """Generate intelligent insights using LLM"""
    
    # Get LLM configuration from session state or sidebar
    llm_provider = st.session_state.get('llm_provider', 'openai')
    api_key = st.session_state.get('api_key', '')
    
    if not api_key:
        return "Please configure your LLM API key in the sidebar to generate AI insights."
    
    # Prepare context for LLM
    context = {
        "chart_type": chart_type,
        "data_summary": df_summary,
        "chart_data": chart_data.to_dict('records') if hasattr(chart_data, 'to_dict') else str(chart_data),
        "total_items": len(df_summary) if 'df' in df_summary else 0,
        "total_value": df_summary.get('total_value', 0),
        "expired_value": df_summary.get('expired_value', 0),
        "high_risk_count": df_summary.get('high_risk_count', 0)
    }
    
    # Create prompt for LLM
    prompt = f"""
    You are a data analyst specializing in inventory and expiry date analysis. 
    
    Analyze the following chart data and provide 3-5 actionable business insights:
    
    Chart Type: {context['chart_type']}
    Data Summary: {context['data_summary']}
    Chart Data: {context['chart_data']}
    
    Please provide insights that are:
    1. Specific and actionable
    2. Business-focused
    3. Include specific numbers and percentages where relevant
    4. Suggest next steps or recommendations
    5. Written in a professional, concise manner
    
    Format your response as a bulleted list with clear, actionable insights.
    """
    
    try:
        if llm_provider == 'openai':
            return call_openai_api(prompt, api_key)
        elif llm_provider == 'ollama':
            return call_ollama_api(prompt)
        elif llm_provider == 'huggingface':
            return call_huggingface_api(prompt, api_key)
        elif llm_provider == 'custom':
            return call_custom_api(prompt, api_key)
        else:
            return "Unsupported LLM provider selected."
            
    except Exception as e:
        return f"Error generating insights: {str(e)}"

def call_openai_api(prompt, api_key):
    """Call OpenAI API for insights"""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a data analyst providing business insights."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"API Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"OpenAI API Error: {str(e)}"

def call_ollama_api(prompt):
    """Call local Ollama API for insights"""
    try:
        data = {
            "model": "llama2",
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['response']
        else:
            return f"Ollama API Error: {response.status_code}"
            
    except Exception as e:
        return f"Ollama API Error: {str(e)}"

def call_huggingface_api(prompt, api_key):
    """Call Hugging Face API for insights"""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "inputs": prompt,
            "parameters": {
                "max_length": 500,
                "temperature": 0.7
            }
        }
        
        response = requests.post(
            "https://api-inference.huggingface.co/models/gpt2",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result[0]['generated_text']
        else:
            return f"Hugging Face API Error: {response.status_code}"
            
    except Exception as e:
        return f"Hugging Face API Error: {str(e)}"

def call_custom_api(prompt, api_key):
    """Call custom API endpoint for insights"""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "prompt": prompt,
            "max_tokens": 500
        }
        
        # Replace with your custom API endpoint
        response = requests.post(
            "https://your-custom-api.com/generate",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', 'No response from custom API')
        else:
            return f"Custom API Error: {response.status_code}"
            
    except Exception as e:
        return f"Custom API Error: {str(e)}"

# Data Persistence Functions
def ensure_data_directory():
    """Ensure the data directory exists"""
    data_dir = Path("historical_data")
    data_dir.mkdir(exist_ok=True)
    return data_dir

def save_analysis_data(df, filename, upload_date):
    """Save processed data with metadata"""
    try:
        data_dir = ensure_data_directory()
        
        # Create metadata
        metadata = {
            'filename': filename,
            'upload_date': upload_date.isoformat(),
            'analysis_date': datetime.now().isoformat(),
            'total_items': len(df),
            'total_value': df['Value'].sum(),
            'expired_count': len(df[df['Stage'] == 'Expired']),
            'high_risk_count': len(df[df['Risk Level'] == 'High'])
        }
        
        # Save data and metadata
        data_file = data_dir / f"{upload_date.strftime('%Y%m%d_%H%M%S')}_{filename}.pkl"
        metadata_file = data_dir / f"{upload_date.strftime('%Y%m%d_%H%M%S')}_{filename}_metadata.json"
        
        with open(data_file, 'wb') as f:
            pickle.dump(df, f)
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(data_file)
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")
        return None

def load_historical_data():
    """Load all historical data files"""
    try:
        data_dir = ensure_data_directory()
        historical_data = []
        
        # Find all data files
        for data_file in data_dir.glob("*.pkl"):
            # Build corresponding metadata filename
            metadata_file = data_file.parent / f"{data_file.stem}_metadata.json"
            
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                with open(data_file, 'rb') as f:
                    df = pickle.load(f)
                
                historical_data.append({
                    'data': df,
                    'metadata': metadata,
                    'file_path': str(data_file)
                })
        
        # Sort by upload date (newest first)
        historical_data.sort(key=lambda x: x['metadata']['upload_date'], reverse=True)
        return historical_data
    except Exception as e:
        st.error(f"Error loading historical data: {str(e)}")
        return []

def compare_datasets(current_df, historical_df, current_metadata, historical_metadata):
    """Compare current data with historical data"""
    try:
        comparison = {}
        
        # Basic metrics comparison
        comparison['total_items'] = {
            'current': len(current_df),
            'historical': len(historical_df),
            'change': len(current_df) - len(historical_df),
            'change_pct': ((len(current_df) - len(historical_df)) / len(historical_df)) * 100 if len(historical_df) > 0 else 0
        }
        
        comparison['total_value'] = {
            'current': current_df['Value'].sum(),
            'historical': historical_df['Value'].sum(),
            'change': current_df['Value'].sum() - historical_df['Value'].sum(),
            'change_pct': ((current_df['Value'].sum() - historical_df['Value'].sum()) / historical_df['Value'].sum()) * 100 if historical_df['Value'].sum() > 0 else 0
        }
        
        comparison['expired_value'] = {
            'current': current_df[current_df['Stage'] == 'Expired']['Value'].sum(),
            'historical': historical_df[historical_df['Stage'] == 'Expired']['Value'].sum(),
            'change': current_df[current_df['Stage'] == 'Expired']['Value'].sum() - historical_df[historical_df['Stage'] == 'Expired']['Value'].sum(),
            'change_pct': ((current_df[current_df['Stage'] == 'Expired']['Value'].sum() - historical_df[historical_df['Stage'] == 'Expired']['Value'].sum()) / historical_df[historical_df['Stage'] == 'Expired']['Value'].sum()) * 100 if historical_df[historical_df['Stage'] == 'Expired']['Value'].sum() > 0 else 0
        }
        
        comparison['high_risk_count'] = {
            'current': len(current_df[current_df['Risk Level'] == 'High']),
            'historical': len(historical_df[historical_df['Risk Level'] == 'High']),
            'change': len(current_df[current_df['Risk Level'] == 'High']) - len(historical_df[historical_df['Risk Level'] == 'High']),
            'change_pct': ((len(current_df[current_df['Risk Level'] == 'High']) - len(historical_df[historical_df['Risk Level'] == 'High'])) / len(historical_df[historical_df['Risk Level'] == 'High'])) * 100 if len(historical_df[historical_df['Risk Level'] == 'High']) > 0 else 0
        }
        
        # Stage comparison
        current_stages = current_df.groupby('Stage')['Value'].sum()
        historical_stages = historical_df.groupby('Stage')['Value'].sum()
        
        comparison['stages'] = {}
        for stage in STAGE_ORDER:
            if stage in current_stages.index and stage in historical_stages.index:
                comparison['stages'][stage] = {
                    'current': current_stages[stage],
                    'historical': historical_stages[stage],
                    'change': current_stages[stage] - historical_stages[stage],
                    'change_pct': ((current_stages[stage] - historical_stages[stage]) / historical_stages[stage]) * 100 if historical_stages[stage] > 0 else 0
                }
        
        # Category comparison
        current_categories = current_df.groupby('Category')['Value'].sum()
        historical_categories = historical_df.groupby('Category')['Value'].sum()
        
        comparison['categories'] = {}
        all_categories = set(current_categories.index) | set(historical_categories.index)
        for category in all_categories:
            current_val = current_categories.get(category, 0)
            historical_val = historical_categories.get(category, 0)
            comparison['categories'][category] = {
                'current': current_val,
                'historical': historical_val,
                'change': current_val - historical_val,
                'change_pct': ((current_val - historical_val) / historical_val) * 100 if historical_val > 0 else 0
            }
        
        return comparison
    except Exception as e:
        st.error(f"Error comparing datasets: {str(e)}")
        return {}

def calculate_write_off_risk(current_df, historical_df):
    """Calculate potential write-off risk based on movement patterns"""
    try:
        write_off_analysis = {}
        
        # Items that have moved to expired or high risk
        current_expired = current_df[current_df['Stage'] == 'Expired']
        historical_expired = historical_df[historical_df['Stage'] == 'Expired']
        
        # New expired items (weren't expired in historical data)
        new_expired = current_expired[~current_expired['Stock Code'].isin(historical_expired['Stock Code'])]
        
        write_off_analysis['new_expired_items'] = {
            'count': len(new_expired),
            'value': new_expired['Value'].sum(),
            'items': new_expired[['Stock Code', 'Description', 'Value', 'DaysToExpiry']].to_dict('records')
        }
        
        # Items moving towards expiry (Stage 3 or high risk)
        current_high_risk = current_df[current_df['Risk Level'] == 'High']
        historical_high_risk = historical_df[historical_df['Risk Level'] == 'High']
        
        # Items that became high risk
        became_high_risk = current_high_risk[~current_high_risk['Stock Code'].isin(historical_high_risk['Stock Code'])]
        
        write_off_analysis['became_high_risk'] = {
            'count': len(became_high_risk),
            'value': became_high_risk['Value'].sum(),
            'items': became_high_risk[['Stock Code', 'Description', 'Value', 'DaysToExpiry', 'Stage']].to_dict('records')
        }
        
        # Items with declining days to expiry
        common_items = current_df[current_df['Stock Code'].isin(historical_df['Stock Code'])]
        if not common_items.empty:
            merged = common_items.merge(
                historical_df[['Stock Code', 'DaysToExpiry']], 
                on='Stock Code', 
                suffixes=('_current', '_historical')
            )
            
            # Items with significant decline in days to expiry
            significant_decline = merged[
                (merged['DaysToExpiry_current'] - merged['DaysToExpiry_historical']) < -30
            ]
            
            write_off_analysis['significant_decline'] = {
                'count': len(significant_decline),
                'value': significant_decline['Value'].sum(),
                'items': significant_decline[['Stock Code', 'Description', 'Value', 'DaysToExpiry_current', 'DaysToExpiry_historical']].to_dict('records')
            }
        
        return write_off_analysis
    except Exception as e:
        st.error(f"Error calculating write-off risk: {str(e)}")
        return {}

# Main app
st.markdown('<h1 class="main-header">BBD Watch Dashboard</h1>', unsafe_allow_html=True)

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Dashboard Settings")

# Threshold settings
st.sidebar.subheader("📊 Risk Thresholds")
wri_threshold_1 = st.sidebar.slider("Low Risk Threshold (Days)", 0, 365, 180, help="Items beyond this threshold are considered low risk")
wri_threshold_2 = st.sidebar.slider("Medium Risk Threshold (Days)", 0, 365, 90, help="Items between this and low risk threshold are medium risk")

# File upload
st.sidebar.subheader("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file", 
    type=["csv"],
    help="Upload a CSV file with columns: BBD, Qty On Hand, UnitCost, Stock Code, Description, Category"
)

# LLM Configuration
st.sidebar.subheader("🤖 AI Insights Configuration")
llm_provider = st.sidebar.selectbox(
    "LLM Provider",
    options=list(LLM_PROVIDERS.keys()),
    help="Choose your preferred LLM provider for generating insights"
)

# Store in session state
st.session_state['llm_provider'] = LLM_PROVIDERS[llm_provider]

# API Key input
api_key = st.sidebar.text_input(
    "API Key",
    type="password",
    help="Enter your API key for the selected LLM provider"
)
st.session_state['api_key'] = api_key

# Enable/disable AI insights
enable_ai_insights = st.sidebar.checkbox(
    "Enable AI Insights",
    value=False,
    help="Enable AI-powered chart insights generation"
)

# Historical Data Management
st.sidebar.subheader("📊 Historical Data")
enable_data_persistence = st.sidebar.checkbox(
    "Save Analysis Data",
    value=True,
    help="Automatically save analysis data for historical comparison"
)

# Load historical data
historical_data = load_historical_data() if enable_data_persistence else []

if historical_data:
    st.sidebar.info(f"📈 {len(historical_data)} historical analysis files found")
    
    # Historical data selector for comparison
    if len(historical_data) > 0:
        historical_options = [f"{h['metadata']['upload_date'][:10]} - {h['metadata']['filename']}" for h in historical_data]
        selected_historical = st.sidebar.selectbox(
            "Compare with:",
            options=["None"] + historical_options,
            help="Select a historical dataset for comparison"
        )
        
        # Store selected historical data in session state
        if selected_historical != "None":
            selected_index = historical_options.index(selected_historical)
            st.session_state['selected_historical'] = historical_data[selected_index]
        else:
            st.session_state['selected_historical'] = None

# Main content
if uploaded_file:
    # Load and validate data
    df_raw = load_and_validate_data(uploaded_file)
    
    if df_raw is not None:
        # Add Run Analysis button - properly centered
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 0.5, 1])
        with col2:
            run_analysis = st.button(
                "Run Analysis", 
                type="primary",
                help="Click to start the data analysis and generate insights",
                use_container_width=True
            )
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Use session state to persist analysis state
        if run_analysis:
            st.session_state['analysis_run'] = True
        
        # Check if analysis should be displayed (either button clicked or already run)
        if run_analysis or st.session_state.get('analysis_run', False):
            # Process data
            with st.spinner("🔄 Processing data and generating insights..."):
                df = process_data(df_raw, wri_threshold_1, wri_threshold_2)
            
            if df is not None:
                # Success message
                st.success(f"✅ Data loaded successfully! {len(df)} items processed.")
                
                # Save data if persistence is enabled
                if enable_data_persistence:
                    upload_date = datetime.now()
                    saved_file = save_analysis_data(df, uploaded_file.name, upload_date)
                    if saved_file:
                        st.info(f"💾 Analysis data saved for historical comparison")
                
                # Add reset button
                if st.button("🔄 Reset Analysis", help="Click to restart the analysis"):
                    st.session_state['analysis_run'] = False
                    st.rerun()
                
                # --- FILTERS SECTION ---
                st.sidebar.subheader("🔍 Filters")
                
                # Category filter
                categories = ['All'] + sorted(df['Category'].unique().tolist())
                selected_category = st.sidebar.selectbox("Category", categories)
                
                # Risk level filter
                risk_levels = ['All'] + sorted(df['Risk Level'].unique().tolist())
                selected_risk = st.sidebar.selectbox("Risk Level", risk_levels)
                
                # Stage filter
                stages = ['All'] + sorted(df['Stage'].unique().tolist())
                selected_stage = st.sidebar.selectbox("Stage", stages)
                
                # Search functionality
                search_term = st.sidebar.text_input("🔍 Search Stock Code/Description", "").lower()
                
                # Apply filters
                filtered_df = df.copy()
                if selected_category != 'All':
                    filtered_df = filtered_df[filtered_df['Category'] == selected_category]
                if selected_risk != 'All':
                    filtered_df = filtered_df[filtered_df['Risk Level'] == selected_risk]
                if selected_stage != 'All':
                    filtered_df = filtered_df[filtered_df['Stage'] == selected_stage]
                if search_term:
                    mask = (
                        filtered_df['Stock Code'].str.lower().str.contains(search_term, na=False) |
                        filtered_df['Description'].str.lower().str.contains(search_term, na=False)
                    )
                    filtered_df = filtered_df[mask]
                
                # Show filter summary
                if len(filtered_df) != len(df):
                    st.sidebar.info(f"Showing {len(filtered_df)} of {len(df)} items")
                
                # --- ENHANCED KPIs ---
                metrics = create_kpi_metrics(filtered_df)
                
                # Create KPI columns with better styling
                kpi_cols = st.columns(4)
                
                with kpi_cols[0]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>💰 Total Stock Value</h3>
                        <h2>{format_currency(metrics['total_value'])}</h2>
                        <p>{metrics['total_skus']} SKUs</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with kpi_cols[1]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>⚠️ Expired Stock</h3>
                        <h2>{format_currency(metrics['expired_value'])}</h2>
                        <p>{metrics['expired_skus']} SKUs expired</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with kpi_cols[2]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🔥 High Risk SKUs</h3>
                        <h2>{metrics['high_risk_count']}</h2>
                        <p>{metrics['critical_skus']} critical (≤30 days)</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with kpi_cols[3]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📅 Avg Days to Expiry</h3>
                        <h2>{metrics['avg_days']:.0f} days</h2>
                        <p>Across all items</p>
                    </div>
                    """, unsafe_allow_html=True)

                # --- ENHANCED VISUALIZATIONS ---
                st.subheader("📊 Interactive Chart Analysis")
                
                # Chart type selector
                chart_options = {
                    "Stage Distribution": "stage_distribution",
                    "Risk Analysis": "risk_analysis", 
                    "Expiry Timeline": "expiry_timeline",
                    "Value vs Days to Expiry": "value_vs_days",
                    "Category Analysis": "category_analysis",
                    "Cost Impact Analysis": "cost_impact",
                    "Trend Forecast": "trend_forecast",
                    "Heatmap": "heatmap"
                }
                
                selected_charts = st.multiselect(
                    "Select chart types to display:",
                    options=list(chart_options.keys()),
                    default=["Stage Distribution", "Risk Analysis"],
                    help="Choose the charts that best suit your analysis needs"
                )
                
                if selected_charts:
                    # Stage summary for charts - maintain fixed order
                    stage_summary = filtered_df.groupby('Stage')['Value'].sum().reset_index()
                    # Ensure all stages are present and in correct order
                    for stage in STAGE_ORDER:
                        if stage not in stage_summary['Stage'].values:
                            stage_summary = pd.concat([stage_summary, pd.DataFrame({'Stage': [stage], 'Value': [0]})], ignore_index=True)
                    
                    # Sort by the predefined stage order
                    stage_summary['Stage'] = pd.Categorical(stage_summary['Stage'], categories=STAGE_ORDER, ordered=True)
                    stage_summary = stage_summary.sort_values('Stage')
                    
                    # Create charts based on selection
                    charts_per_row = 2
                    for i in range(0, len(selected_charts), charts_per_row):
                        chart_row = selected_charts[i:i + charts_per_row]
                        cols = st.columns(len(chart_row))
                        
                        for j, chart_name in enumerate(chart_row):
                            with cols[j]:
                                if chart_options[chart_name] == "stage_distribution":
                                    # Pie and Bar charts for stage distribution
                                    chart_type = st.radio(
                                        "Chart Type:",
                                        ["Pie Chart", "Bar Chart"],
                                        key=f"stage_chart_{i}_{j}",
                                        horizontal=True
                                    )
                                    
                                    if chart_type == "Pie Chart":
                                        fig = px.pie(
                                            stage_summary, 
                                            values='Value', 
                                            names='Stage', 
                                            title='Stock Value by Stage',
                                            color='Stage',
                                            color_discrete_map=STAGE_COLORS,
                                            hole=0.4
                                        )
                                        fig.update_traces(textposition='inside', textinfo='percent+label')
                                    else:
                                        fig = px.bar(
                                            stage_summary, 
                                            x='Stage', 
                                            y='Value', 
                                            title='Total Value by Expiry Stage',
                                            color='Stage',
                                            color_discrete_map=STAGE_COLORS,
                                            text=stage_summary['Value'].apply(format_currency)
                                        )
                                        fig.update_traces(textposition='outside')
                                        fig.update_xaxes(categoryorder='array', categoryarray=STAGE_ORDER)
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                elif chart_options[chart_name] == "risk_analysis":
                                    # Risk level analysis
                                    risk_summary = filtered_df.groupby('Risk Level')['Value'].sum().reset_index()
                                    # Maintain fixed order: Low, Medium, High
                                    risk_order = ['Low', 'Medium', 'High']
                                    risk_summary['Risk Level'] = pd.Categorical(risk_summary['Risk Level'], categories=risk_order, ordered=True)
                                    risk_summary = risk_summary.sort_values('Risk Level')
                                    
                                    fig = px.bar(
                                        risk_summary,
                                        x='Risk Level',
                                        y='Value',
                                        title='Value by Risk Level',
                                        color='Risk Level',
                                        color_discrete_map={'High': '#e57373', 'Medium': '#ffb74d', 'Low': '#4caf50'},
                                        text=risk_summary['Value'].apply(format_currency)
                                    )
                                    fig.update_traces(textposition='outside')
                                    fig.update_xaxes(categoryorder='array', categoryarray=risk_order)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                elif chart_options[chart_name] == "expiry_timeline":
                                    # Expiry timeline analysis
                                    timeline_data = filtered_df[filtered_df['DaysToExpiry'] >= -90].copy()
                                    timeline_data['Expiry_Week'] = timeline_data['DaysToExpiry'] // 7
                                    
                                    timeline_summary = timeline_data.groupby('Expiry_Week')['Value'].sum().reset_index()
                                    timeline_summary['Week_Label'] = timeline_summary['Expiry_Week'].apply(
                                        lambda x: f"{x} weeks" if x >= 0 else f"Expired {abs(x)} weeks ago"
                                    )
                                    
                                    fig = px.line(
                                        timeline_summary,
                                        x='Week_Label',
                                        y='Value',
                                        title='Value Expiring by Week',
                                        markers=True
                                    )
                                    fig.update_xaxes(tickangle=45)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                elif chart_options[chart_name] == "value_vs_days":
                                    # Scatter plot: Value vs Days to Expiry
                                    scatter_data = filtered_df[filtered_df['DaysToExpiry'].between(-90, 365)].copy()
                                    # Ensure stage order is maintained in the scatter plot
                                    scatter_data['Stage'] = pd.Categorical(scatter_data['Stage'], categories=STAGE_ORDER, ordered=True)
                                    
                                    fig = px.scatter(
                                        scatter_data,
                                        x='DaysToExpiry',
                                        y='Value',
                                        color='Stage',
                                        size='Qty On Hand',
                                        hover_data=['Stock Code', 'Description', 'Category'],
                                        title='Value vs Days to Expiry',
                                        color_discrete_map=STAGE_COLORS
                                    )
                                    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Expiry Date")
                                    fig.add_vline(x=wri_threshold_2, line_dash="dash", line_color="orange", annotation_text="Medium Risk")
                                    fig.add_vline(x=wri_threshold_1, line_dash="dash", line_color="green", annotation_text="Low Risk")
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                elif chart_options[chart_name] == "category_analysis":
                                    # Category analysis
                                    category_summary = filtered_df.groupby('Category').agg({
                                        'Value': 'sum',
                                        'Qty On Hand': 'sum',
                                        'Stock Code': 'count'
                                    }).reset_index()
                                    category_summary.columns = ['Category', 'Total Value', 'Total Quantity', 'SKU Count']
                                    # Sort categories by value for better visualization
                                    category_summary = category_summary.sort_values('Total Value', ascending=False)
                                    
                                    fig = px.bar(
                                        category_summary,
                                        x='Category',
                                        y='Total Value',
                                        title='Value by Category',
                                        text=category_summary['Total Value'].apply(format_currency),
                                        hover_data=['Total Quantity', 'SKU Count']
                                    )
                                    fig.update_traces(textposition='outside')
                                    fig.update_xaxes(tickangle=45)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                elif chart_options[chart_name] == "cost_impact":
                                    # Cost impact analysis
                                    cost_data = filtered_df.copy()
                                    cost_data['Cost_Impact'] = cost_data['Value'] * (cost_data['WRI'] / 100)
                                    
                                    # Get top 10 items and ensure stage order is maintained
                                    top_items = cost_data.nlargest(10, 'Cost_Impact')
                                    top_items['Stage'] = pd.Categorical(top_items['Stage'], categories=STAGE_ORDER, ordered=True)
                                    top_items = top_items.sort_values(['Cost_Impact', 'Stage'], ascending=[False, True])
                                    
                                    fig = px.bar(
                                        top_items,
                                        x='Stock Code',
                                        y='Cost_Impact',
                                        color='Stage',
                                        title='Top 10 Items by Cost Impact (Value × WRI)',
                                        color_discrete_map=STAGE_COLORS,
                                        text=top_items['Cost_Impact'].apply(format_currency)
                                    )
                                    fig.update_traces(textposition='outside')
                                    fig.update_xaxes(tickangle=45)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                elif chart_options[chart_name] == "trend_forecast":
                                    # Trend forecast
                                    trend_data = filtered_df[filtered_df['DaysToExpiry'] >= 0].copy()
                                    trend_data['Month'] = trend_data['BBD'].dt.to_period('M').dt.to_timestamp()
                                    trend_summary = trend_data.groupby('Month')['Value'].sum().reset_index()
                                    
                                    fig = px.line(
                                        trend_summary,
                                        x='Month',
                                        y='Value',
                                        title='Monthly Expiry Forecast',
                                        markers=True
                                    )
                                    fig.update_layout(xaxis_title="Month", yaxis_title="Value (K)")
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                elif chart_options[chart_name] == "heatmap":
                                    # Heatmap: Category vs Stage
                                    heatmap_data = filtered_df.groupby(['Category', 'Stage'])['Value'].sum().reset_index()
                                    heatmap_pivot = heatmap_data.pivot(index='Category', columns='Stage', values='Value').fillna(0)
                                    
                                    # Ensure all stages are present
                                    for stage in STAGE_ORDER:
                                        if stage not in heatmap_pivot.columns:
                                            heatmap_pivot[stage] = 0
                                    
                                    # Reorder columns to match STAGE_ORDER
                                    heatmap_pivot = heatmap_pivot[STAGE_ORDER]
                                    
                                    fig = px.imshow(
                                        heatmap_pivot,
                                        title='Value Heatmap: Category vs Stage',
                                        color_continuous_scale='RdYlGn_r',
                                        aspect="auto"
                                    )
                                    fig.update_layout(
                                        xaxis_title="Stage",
                                        yaxis_title="Category"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add chart insights
                    if selected_charts:
                        st.subheader("📈 Chart Insights")
                        with st.expander("View analysis insights"):
                            
                            # Check if AI insights are enabled
                            if enable_ai_insights and st.session_state.get('api_key'):
                                st.info("🤖 Generating AI-powered insights...")
                                
                                # Generate insights for each selected chart
                                for chart_name in selected_charts:
                                    st.markdown(f"**{chart_name}:**")
                                    
                                    # Prepare data for LLM
                                    if chart_options[chart_name] == "stage_distribution":
                                        chart_data = stage_summary
                                    elif chart_options[chart_name] == "risk_analysis":
                                        chart_data = filtered_df.groupby('Risk Level')['Value'].sum().reset_index()
                                    elif chart_options[chart_name] == "category_analysis":
                                        chart_data = filtered_df.groupby('Category')['Value'].sum().reset_index()
                                    elif chart_options[chart_name] == "cost_impact":
                                        cost_data = filtered_df.copy()
                                        cost_data['Cost_Impact'] = cost_data['Value'] * (cost_data['WRI'] / 100)
                                        chart_data = cost_data.nlargest(10, 'Cost_Impact')[['Stock Code', 'Cost_Impact', 'Stage']]
                                    else:
                                        chart_data = pd.DataFrame()  # For other charts
                                    
                                    # Prepare summary data
                                    df_summary = {
                                        'total_value': metrics['total_value'],
                                        'expired_value': metrics['expired_value'],
                                        'high_risk_count': metrics['high_risk_count'],
                                        'total_skus': metrics['total_skus'],
                                        'avg_days': metrics['avg_days']
                                    }
                                    
                                    # Generate LLM insights
                                    ai_insight = generate_llm_insights(chart_data, chart_name, df_summary)
                                    st.write(ai_insight)
                                    st.markdown("---")
                            
                            else:
                                # Basic insights (fallback)
                                insights = []
                                
                                if "Stage Distribution" in selected_charts:
                                    max_stage = stage_summary.loc[stage_summary['Value'].idxmax(), 'Stage']
                                    max_value = stage_summary['Value'].max()
                                    insights.append(f"• **{max_stage}** has the highest value: {format_currency(max_value)}")
                                
                                if "Risk Analysis" in selected_charts:
                                    risk_summary = filtered_df.groupby('Risk Level')['Value'].sum()
                                    if not risk_summary.empty:
                                        highest_risk = risk_summary.idxmax()
                                        insights.append(f"• **{highest_risk} risk** items represent the highest value at risk")
                                
                                if "Category Analysis" in selected_charts:
                                    category_summary = filtered_df.groupby('Category')['Value'].sum()
                                    if not category_summary.empty:
                                        top_category = category_summary.idxmax()
                                        insights.append(f"• **{top_category}** is the highest value category")
                                
                                if insights:
                                    for insight in insights:
                                        st.write(insight)
                                else:
                                    st.write("Select charts to view insights")
                                
                                # Show AI insights option if not enabled
                                if not enable_ai_insights:
                                    st.info("💡 Enable AI Insights in the sidebar for more detailed, AI-powered analysis!")
                else:
                    st.info("Please select at least one chart type to display visualizations.")

                # --- SHELF LIFE STAGE ANALYSIS REPORT ---
                st.subheader("Shelf life stage analysis report")
                with st.expander("View shelf life stage analysis"):
                    stages = {
                        "Stage 1": "Stage 1 - Stocks beyond 6 month shelf life\n181+ days", 
                        "Stage 2": "Stage 2 - Stocks between 3 to 6 months shelf life\n91 - 180 days", 
                        "Stage 3": "Stage 3 - Stocks below 3 months shelf life\n0 - 90 days", 
                        "Expired": "Stage 4 - Stocks passed BBD Shelf life\n - 0 < days"
                    }

                    for stage_key, stage_label in stages.items():
                        st.markdown(f"### {stage_label}")
                        filtered = filtered_df[filtered_df['Stage'] == stage_key]

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

                # --- EXPIRED STOCK REPORT ---
                st.subheader("🚨 Expired Stock Report")
                with st.expander("View expired stock details"):
                    expired_items = filtered_df[filtered_df['Stage'] == 'Expired']
                    
                    if not expired_items.empty:
                        st.warning(f"⚠️ {len(expired_items)} items have expired with total value of {format_currency(expired_items['Value'].sum())}")
                        
                        # Expired items by category
                        expired_by_category = expired_items.groupby('Category').agg({
                            'Stock Code': 'count',
                            'Qty On Hand': 'sum',
                            'Value': 'sum',
                            'DaysToExpiry': 'mean'
                        }).reset_index()
                        expired_by_category.columns = ['Category', 'SKU Count', 'Total Quantity', 'Total Value', 'Avg Days Past Expiry']
                        expired_by_category['Total Value'] = expired_by_category['Total Value'].apply(format_currency)
                        expired_by_category['Avg Days Past Expiry'] = expired_by_category['Avg Days Past Expiry'].apply(lambda x: f"{abs(x):.0f} days")
                        
                        st.write("**Expired Items by Category:**")
                        st.dataframe(expired_by_category, use_container_width=True)
                        
                        # Detailed expired items
                        st.write("**Detailed Expired Items:**")
                        expired_display = expired_items[['Stock Code', 'Description', 'Category', 'BBD', 'Qty On Hand', 'UnitCost', 'Value', 'DaysToExpiry']].copy()
                        expired_display['Value'] = expired_display['Value'].apply(format_currency)
                        expired_display['UnitCost'] = expired_display['UnitCost'].apply(format_currency)
                        expired_display['Qty On Hand'] = expired_display['Qty On Hand'].apply(format_number)
                        expired_display['DaysToExpiry'] = expired_display['DaysToExpiry'].apply(lambda x: f"{abs(x)} days past expiry")
                        
                        st.dataframe(expired_display.sort_values('DaysToExpiry'), use_container_width=True)
                    else:
                        st.success("✅ No expired items found!")

                # --- NEAR-EXPIRY STOCK REPORT ---
                st.subheader("⏰ Near-Expiry Stock Report")
                with st.expander("View near-expiry stock details"):
                    # Define expiry bands
                    expiry_bands = {
                        "7 days": (0, 7),
                        "14 days": (8, 14),
                        "30 days": (15, 30),
                        "60 days": (31, 60),
                        "90 days": (61, 90)
                    }
                    
                    near_expiry_summary = []
                    for band_name, (min_days, max_days) in expiry_bands.items():
                        band_items = filtered_df[filtered_df['DaysToExpiry'].between(min_days, max_days)]
                        if not band_items.empty:
                            near_expiry_summary.append({
                                'Time Band': band_name,
                                'SKU Count': len(band_items),
                                'Total Quantity': band_items['Qty On Hand'].sum(),
                                'Total Value': format_currency(band_items['Value'].sum()),
                                'Avg Days to Expiry': f"{band_items['DaysToExpiry'].mean():.0f} days"
                            })
                    
                    if near_expiry_summary:
                        summary_df = pd.DataFrame(near_expiry_summary)
                        st.write("**Near-Expiry Summary:**")
                        st.dataframe(summary_df, use_container_width=True)
                        
                        # Detailed near-expiry items
                        near_expiry_items = filtered_df[filtered_df['DaysToExpiry'].between(0, 90)]
                        if not near_expiry_items.empty:
                            st.write("**Detailed Near-Expiry Items:**")
                            near_expiry_display = near_expiry_items[['Stock Code', 'Description', 'Category', 'BBD', 'Qty On Hand', 'UnitCost', 'Value', 'DaysToExpiry', 'Risk Level']].copy()
                            near_expiry_display['Value'] = near_expiry_display['Value'].apply(format_currency)
                            near_expiry_display['UnitCost'] = near_expiry_display['UnitCost'].apply(format_currency)
                            near_expiry_display['Qty On Hand'] = near_expiry_display['Qty On Hand'].apply(format_number)
                            near_expiry_display['DaysToExpiry'] = near_expiry_display['DaysToExpiry'].apply(lambda x: f"{x} days")
                            
                            st.dataframe(near_expiry_display.sort_values('DaysToExpiry'), use_container_width=True)
                    else:
                        st.success("✅ No items expiring within 90 days!")

                # --- AGING STOCK REPORT ---
                st.subheader("📊 Aging Stock Report")
                with st.expander("View aging stock analysis"):
                    # Define aging bands
                    aging_bands = {
                        "0-30 days": (0, 30),
                        "31-60 days": (31, 60),
                        "61-90 days": (61, 90),
                        "90+ days": (91, 365)
                    }
                    
                    aging_summary = []
                    for band_name, (min_days, max_days) in aging_bands.items():
                        band_items = filtered_df[filtered_df['DaysToExpiry'].between(min_days, max_days)]
                        if not band_items.empty:
                            aging_summary.append({
                                'Aging Band': band_name,
                                'SKU Count': len(band_items),
                                'Total Quantity': band_items['Qty On Hand'].sum(),
                                'Total Value': format_currency(band_items['Value'].sum()),
                                'Avg Days to Expiry': f"{band_items['DaysToExpiry'].mean():.0f} days"
                            })
                    
                    if aging_summary:
                        aging_df = pd.DataFrame(aging_summary)
                        st.write("**Aging Stock Summary:**")
                        st.dataframe(aging_df, use_container_width=True)
                        
                        # Aging stock visualization
                        fig = px.bar(
                            aging_df,
                            x='Aging Band',
                            y='SKU Count',
                            title='SKU Count by Aging Band',
                            text='SKU Count'
                        )
                        fig.update_traces(textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed aging items
                        aging_items = filtered_df[filtered_df['DaysToExpiry'].between(0, 365)]
                        if not aging_items.empty:
                            st.write("**Detailed Aging Stock Items:**")
                            aging_display = aging_items[['Stock Code', 'Description', 'Category', 'BBD', 'Qty On Hand', 'UnitCost', 'Value', 'DaysToExpiry', 'Stage']].copy()
                            aging_display['Value'] = aging_display['Value'].apply(format_currency)
                            aging_display['UnitCost'] = aging_display['UnitCost'].apply(format_currency)
                            aging_display['Qty On Hand'] = aging_display['Qty On Hand'].apply(format_number)
                            aging_display['DaysToExpiry'] = aging_display['DaysToExpiry'].apply(lambda x: f"{x} days")
                            
                            st.dataframe(aging_display.sort_values('DaysToExpiry'), use_container_width=True)
                    else:
                        st.info("ℹ️ No aging stock data available")

                # --- STOCK AT RISK OF OBSOLESCENCE ---
                st.subheader("⚠️ Stock at Risk of Obsolescence")
                with st.expander("View obsolescence risk analysis"):
                    # Identify slow-moving items (high quantity, low turnover potential)
                    # For now, we'll use WRI and days to expiry as proxies
                    obsolescence_risk = filtered_df[
                        (filtered_df['WRI'] > 70) & 
                        (filtered_df['DaysToExpiry'] > 30) &
                        (filtered_df['Qty On Hand'] > filtered_df['Qty On Hand'].quantile(0.75))
                    ].copy()
                    
                    if not obsolescence_risk.empty:
                        st.warning(f"⚠️ {len(obsolescence_risk)} items identified as high obsolescence risk")
                        
                        # Obsolescence risk summary
                        obsolescence_summary = obsolescence_risk.groupby('Category').agg({
                            'Stock Code': 'count',
                            'Qty On Hand': 'sum',
                            'Value': 'sum',
                            'WRI': 'mean'
                        }).reset_index()
                        obsolescence_summary.columns = ['Category', 'SKU Count', 'Total Quantity', 'Total Value', 'Avg WRI']
                        obsolescence_summary['Total Value'] = obsolescence_summary['Total Value'].apply(format_currency)
                        obsolescence_summary['Avg WRI'] = obsolescence_summary['Avg WRI'].apply(lambda x: f"{x:.1f}%")
                        
                        st.write("**Obsolescence Risk by Category:**")
                        st.dataframe(obsolescence_summary, use_container_width=True)
                        
                        # Detailed obsolescence risk items
                        st.write("**Detailed Obsolescence Risk Items:**")
                        obsolescence_display = obsolescence_risk[['Stock Code', 'Description', 'Category', 'BBD', 'Qty On Hand', 'UnitCost', 'Value', 'DaysToExpiry', 'WRI', 'Risk Level']].copy()
                        obsolescence_display['Value'] = obsolescence_display['Value'].apply(format_currency)
                        obsolescence_display['UnitCost'] = obsolescence_display['UnitCost'].apply(format_currency)
                        obsolescence_display['Qty On Hand'] = obsolescence_display['Qty On Hand'].apply(format_number)
                        obsolescence_display['WRI'] = obsolescence_display['WRI'].apply(lambda x: f"{x:.1f}%")
                        obsolescence_display['DaysToExpiry'] = obsolescence_display['DaysToExpiry'].apply(lambda x: f"{x} days")
                        
                        st.dataframe(obsolescence_display.sort_values('WRI', ascending=False), use_container_width=True)
                    else:
                        st.success("✅ No items identified as high obsolescence risk!")

                # --- BATCH TRACEABILITY REPORT ---
                st.subheader("🔍 Batch Traceability Report")
                with st.expander("View batch traceability analysis"):
                    # Check if batch information is available
                    if 'Batch' in filtered_df.columns or 'Batch No' in filtered_df.columns:
                        batch_col = 'Batch' if 'Batch' in filtered_df.columns else 'Batch No'
                        
                        # Batch analysis
                        batch_summary = filtered_df.groupby(batch_col).agg({
                            'Stock Code': 'count',
                            'Qty On Hand': 'sum',
                            'Value': 'sum',
                            'DaysToExpiry': 'mean'
                        }).reset_index()
                        batch_summary.columns = ['Batch', 'SKU Count', 'Total Quantity', 'Total Value', 'Avg Days to Expiry']
                        batch_summary['Total Value'] = batch_summary['Total Value'].apply(format_currency)
                        batch_summary['Avg Days to Expiry'] = batch_summary['Avg Days to Expiry'].apply(lambda x: f"{x:.0f} days")
                        
                        st.write("**Batch Summary:**")
                        st.dataframe(batch_summary.sort_values('Total Value', ascending=False), use_container_width=True)
                        
                        # Expiring batches
                        expiring_batches = filtered_df[filtered_df['DaysToExpiry'].between(0, 90)]
                        if not expiring_batches.empty and batch_col in expiring_batches.columns:
                            st.write("**Expiring Batches (Next 90 Days):**")
                            expiring_batch_summary = expiring_batches.groupby(batch_col).agg({
                                'Stock Code': 'count',
                                'Qty On Hand': 'sum',
                                'Value': 'sum',
                                'DaysToExpiry': 'min'
                            }).reset_index()
                            expiring_batch_summary.columns = ['Batch', 'SKU Count', 'Total Quantity', 'Total Value', 'Earliest Expiry']
                            expiring_batch_summary['Total Value'] = expiring_batch_summary['Total Value'].apply(format_currency)
                            expiring_batch_summary['Earliest Expiry'] = expiring_batch_summary['Earliest Expiry'].apply(lambda x: f"{x} days")
                            
                            st.dataframe(expiring_batch_summary.sort_values('Earliest Expiry'), use_container_width=True)
                    else:
                        st.info("ℹ️ Batch information not available in the dataset. Add 'Batch' or 'Batch No' column for batch traceability analysis.")

                # --- STOCK VALUATION BY EXPIRY ---
                st.subheader("💰 Stock Valuation by Expiry")
                with st.expander("View stock valuation analysis"):
                    # Valuation by expiry bands
                    valuation_bands = {
                        "Expired": (-365, -1),
                        "0-30 days": (0, 30),
                        "31-60 days": (31, 60),
                        "61-90 days": (61, 90),
                        "91-180 days": (91, 180),
                        "180+ days": (181, 365)
                    }
                    
                    valuation_summary = []
                    for band_name, (min_days, max_days) in valuation_bands.items():
                        band_items = filtered_df[filtered_df['DaysToExpiry'].between(min_days, max_days)]
                        if not band_items.empty:
                            valuation_summary.append({
                                'Expiry Band': band_name,
                                'SKU Count': len(band_items),
                                'Total Quantity': band_items['Qty On Hand'].sum(),
                                'Total Value': format_currency(band_items['Value'].sum()),
                                'Avg Unit Cost': format_currency(band_items['UnitCost'].mean()),
                                'Value at Risk': format_currency(band_items['Value'].sum() * (band_items['WRI'].mean() / 100))
                            })
                    
                    if valuation_summary:
                        valuation_df = pd.DataFrame(valuation_summary)
                        st.write("**Stock Valuation by Expiry Band:**")
                        st.dataframe(valuation_df, use_container_width=True)
                        
                        # Valuation visualization
                        fig = px.bar(
                            valuation_df,
                            x='Expiry Band',
                            y='SKU Count',
                            title='SKU Count by Expiry Band',
                            text='SKU Count',
                            color='Total Value'
                        )
                        fig.update_traces(textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # High value at risk items
                        high_risk_value = filtered_df[filtered_df['WRI'] > 50].copy()
                        if not high_risk_value.empty:
                            st.write("**High Value at Risk Items:**")
                            high_risk_value['Value at Risk'] = high_risk_value['Value'] * (high_risk_value['WRI'] / 100)
                            high_risk_display = high_risk_value.nlargest(10, 'Value at Risk')[['Stock Code', 'Description', 'Category', 'Value', 'WRI', 'Value at Risk', 'DaysToExpiry']].copy()
                            high_risk_display['Value'] = high_risk_display['Value'].apply(format_currency)
                            high_risk_display['Value at Risk'] = high_risk_display['Value at Risk'].apply(format_currency)
                            high_risk_display['WRI'] = high_risk_display['WRI'].apply(lambda x: f"{x:.1f}%")
                            high_risk_display['DaysToExpiry'] = high_risk_display['DaysToExpiry'].apply(lambda x: f"{x} days")
                            
                            st.dataframe(high_risk_display, use_container_width=True)
                    else:
                        st.info("ℹ️ No valuation data available")

                # --- EXPIRY TREND ANALYSIS REPORT ---
                st.subheader("📈 Expiry Trend Analysis Report")
                with st.expander("View expiry trend analysis"):
                    # Monthly expiry trend analysis
                    trend_data = filtered_df[filtered_df['DaysToExpiry'] >= -90].copy()
                    trend_data['Expiry_Month'] = trend_data['BBD'].dt.to_period('M').dt.to_timestamp()
                    
                    # Monthly expiry summary
                    monthly_trend = trend_data.groupby('Expiry_Month').agg({
                        'Stock Code': 'count',
                        'Value': 'sum',
                        'Qty On Hand': 'sum'
                    }).reset_index()
                    monthly_trend.columns = ['Month', 'SKU Count', 'Total Value', 'Total Quantity']
                    monthly_trend['Total Value'] = monthly_trend['Total Value'].apply(format_currency)
                    
                    st.write("**Monthly Expiry Trend:**")
                    st.dataframe(monthly_trend, use_container_width=True)
                    
                    # Trend visualization
                    fig = px.line(
                        monthly_trend,
                        x='Month',
                        y='SKU Count',
                        title='Monthly Expiry Trend - SKU Count',
                        markers=True
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Value trend
                    fig2 = px.line(
                        monthly_trend,
                        x='Month',
                        y='Total Value',
                        title='Monthly Expiry Trend - Value',
                        markers=True
                    )
                    fig2.update_xaxes(tickangle=45)
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Category trend analysis
                    category_trend = trend_data.groupby(['Category', 'Expiry_Month']).agg({
                        'Value': 'sum'
                    }).reset_index()
                    category_trend['Expiry_Month'] = category_trend['Expiry_Month'].dt.strftime('%Y-%m')
                    
                    if not category_trend.empty:
                        st.write("**Category-wise Expiry Trend:**")
                        category_pivot = category_trend.pivot(index='Expiry_Month', columns='Category', values='Value').fillna(0)
                        category_pivot = category_pivot.applymap(format_currency)
                        st.dataframe(category_pivot, use_container_width=True)
                    
                    # Historical comparison (if available)
                    if st.session_state.get('selected_historical'):
                        st.write("**Historical Trend Comparison:**")
                        historical_data = st.session_state['selected_historical']
                        historical_df = historical_data['data']
                        
                        # Compare current vs historical trends
                        current_expired = len(filtered_df[filtered_df['Stage'] == 'Expired'])
                        historical_expired = len(historical_df[historical_df['Stage'] == 'Expired'])
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Current Expired Items", current_expired)
                        with col2:
                            st.metric("Historical Expired Items", historical_expired, 
                                    delta=current_expired - historical_expired)

                # --- RECOMMENDED ACTIONS REPORT ---
                st.subheader("🎯 Recommended Actions Report")
                with st.expander("View recommended actions"):
                    # Generate recommended actions based on risk levels and expiry
                    def generate_action_recommendation(row):
                        if row['Stage'] == 'Expired':
                            return "Immediate disposal/write-off required"
                        elif row['DaysToExpiry'] <= 7:
                            return "Urgent: Discount or donation"
                        elif row['DaysToExpiry'] <= 30:
                            return "Promote heavily or transfer to high-demand location"
                        elif row['WRI'] > 75:
                            return "Consider bulk discount or alternative use"
                        elif row['Qty On Hand'] > filtered_df['Qty On Hand'].quantile(0.9):
                            return "High stock level - review ordering policy"
                        else:
                            return "Monitor closely"
                    
                    # Add action recommendations
                    action_df = filtered_df.copy()
                    action_df['Recommended Action'] = action_df.apply(generate_action_recommendation, axis=1)
                    
                    # Action summary
                    action_summary = action_df.groupby('Recommended Action').agg({
                        'Stock Code': 'count',
                        'Value': 'sum'
                    }).reset_index()
                    action_summary.columns = ['Recommended Action', 'SKU Count', 'Total Value']
                    action_summary['Total Value'] = action_summary['Total Value'].apply(format_currency)
                    
                    st.write("**Action Summary:**")
                    st.dataframe(action_summary.sort_values('SKU Count', ascending=False), use_container_width=True)
                    
                    # Priority actions (expired and urgent items)
                    priority_items = action_df[
                        (action_df['Stage'] == 'Expired') | 
                        (action_df['DaysToExpiry'] <= 30)
                    ].copy()
                    
                    if not priority_items.empty:
                        st.warning(f"🚨 {len(priority_items)} items require immediate attention!")
                        
                        priority_display = priority_items[['Stock Code', 'Description', 'Category', 'DaysToExpiry', 'Value', 'Recommended Action']].copy()
                        priority_display['Value'] = priority_display['Value'].apply(format_currency)
                        priority_display['DaysToExpiry'] = priority_display['DaysToExpiry'].apply(lambda x: f"{x} days" if pd.notna(x) else "Expired")
                        
                        st.write("**Priority Action Items:**")
                        st.dataframe(priority_display.sort_values('DaysToExpiry'), use_container_width=True)
                    else:
                        st.success("✅ No immediate priority actions required!")
                    
                    # Export action plan
                    action_plan = action_df[['Stock Code', 'Description', 'Category', 'DaysToExpiry', 'Value', 'Recommended Action']].copy()
                    action_plan['Value'] = action_plan['Value'].apply(format_currency)
                    action_plan['DaysToExpiry'] = action_plan['DaysToExpiry'].apply(lambda x: f"{x} days" if pd.notna(x) else "Expired")
                    
                    csv_action_plan = action_plan.to_csv(index=False)
                    st.download_button(
                        label="📥 Export Action Plan",
                        data=csv_action_plan,
                        file_name=f"action_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                # --- COMPREHENSIVE SUMMARY REPORT ---
                st.subheader("📋 Comprehensive Summary Report")
                with st.expander("View comprehensive summary"):
                    # Create comprehensive summary
                    summary_data = {
                        'Metric': [
                            'Total SKUs',
                            'Total Value',
                            'Expired Items',
                            'Expired Value',
                            'High Risk Items',
                            'Critical Items (≤30 days)',
                            'Average Days to Expiry',
                            'Items Expiring in 7 Days',
                            'Items Expiring in 30 Days',
                            'Items Expiring in 90 Days'
                        ],
                        'Value': [
                            len(filtered_df),
                            format_currency(filtered_df['Value'].sum()),
                            len(filtered_df[filtered_df['Stage'] == 'Expired']),
                            format_currency(filtered_df[filtered_df['Stage'] == 'Expired']['Value'].sum()),
                            len(filtered_df[filtered_df['Risk Level'] == 'High']),
                            len(filtered_df[filtered_df['DaysToExpiry'].between(0, 30)]),
                            f"{filtered_df['DaysToExpiry'].mean():.0f} days",
                            len(filtered_df[filtered_df['DaysToExpiry'].between(0, 7)]),
                            len(filtered_df[filtered_df['DaysToExpiry'].between(0, 30)]),
                            len(filtered_df[filtered_df['DaysToExpiry'].between(0, 90)])
                        ]
                    }
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Risk assessment
                    st.write("**Risk Assessment:**")
                    risk_assessment = {
                        'Risk Level': ['Low', 'Medium', 'High'],
                        'SKU Count': [
                            len(filtered_df[filtered_df['Risk Level'] == 'Low']),
                            len(filtered_df[filtered_df['Risk Level'] == 'Medium']),
                            len(filtered_df[filtered_df['Risk Level'] == 'High'])
                        ],
                        'Total Value': [
                            format_currency(filtered_df[filtered_df['Risk Level'] == 'Low']['Value'].sum()),
                            format_currency(filtered_df[filtered_df['Risk Level'] == 'Medium']['Value'].sum()),
                            format_currency(filtered_df[filtered_df['Risk Level'] == 'High']['Value'].sum())
                        ]
                    }
                    
                    risk_df = pd.DataFrame(risk_assessment)
                    st.dataframe(risk_df, use_container_width=True)
                    
                    # Category summary
                    st.write("**Category Summary:**")
                    category_summary = filtered_df.groupby('Category').agg({
                        'Stock Code': 'count',
                        'Value': 'sum',
                        'Qty On Hand': 'sum',
                        'DaysToExpiry': 'mean'
                    }).reset_index()
                    category_summary.columns = ['Category', 'SKU Count', 'Total Value', 'Total Quantity', 'Avg Days to Expiry']
                    category_summary['Total Value'] = category_summary['Total Value'].apply(format_currency)
                    category_summary['Avg Days to Expiry'] = category_summary['Avg Days to Expiry'].apply(lambda x: f"{x:.0f} days")
                    
                    st.dataframe(category_summary.sort_values('Total Value', ascending=False), use_container_width=True)
                    
                    # Location/Site Analysis (if available)
                    if 'Location' in filtered_df.columns or 'Site' in filtered_df.columns:
                        location_col = 'Location' if 'Location' in filtered_df.columns else 'Site'
                        st.write("**Location/Site Analysis:**")
                        location_summary = filtered_df.groupby(location_col).agg({
                            'Stock Code': 'count',
                            'Value': 'sum',
                            'Qty On Hand': 'sum',
                            'DaysToExpiry': 'mean'
                        }).reset_index()
                        location_summary.columns = ['Location', 'SKU Count', 'Total Value', 'Total Quantity', 'Avg Days to Expiry']
                        location_summary['Total Value'] = location_summary['Total Value'].apply(format_currency)
                        location_summary['Avg Days to Expiry'] = location_summary['Avg Days to Expiry'].apply(lambda x: f"{x:.0f} days")
                        
                        st.dataframe(location_summary.sort_values('Total Value', ascending=False), use_container_width=True)
                        
                        # Location risk analysis
                        location_risk = filtered_df.groupby(location_col).agg({
                            'Stock Code': 'count',
                            'Value': 'sum',
                            'Risk Level': lambda x: (x == 'High').sum()
                        }).reset_index()
                        location_risk.columns = ['Location', 'Total SKUs', 'Total Value', 'High Risk SKUs']
                        location_risk['Total Value'] = location_risk['Total Value'].apply(format_currency)
                        location_risk['Risk Percentage'] = (location_risk['High Risk SKUs'] / location_risk['Total SKUs'] * 100).apply(lambda x: f"{x:.1f}%")
                        
                        st.write("**Location Risk Analysis:**")
                        st.dataframe(location_risk.sort_values('High Risk SKUs', ascending=False), use_container_width=True)
                    else:
                        st.info("ℹ️ Location/Site information not available. Add 'Location' or 'Site' column for location analysis.")
                    
                    # Export comprehensive summary
                    comprehensive_summary = {
                        'Summary': summary_df,
                        'Risk Assessment': risk_df,
                        'Category Summary': category_summary
                    }
                    
                    # Create a comprehensive Excel export
                    import io
                    from openpyxl import Workbook
                    from openpyxl.utils.dataframe import dataframe_to_rows
                    
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                        risk_df.to_excel(writer, sheet_name='Risk Assessment', index=False)
                        category_summary.to_excel(writer, sheet_name='Category Summary', index=False)
                        action_plan.to_excel(writer, sheet_name='Action Plan', index=False)
                        
                        # Add location analysis if available
                        if 'Location' in filtered_df.columns or 'Site' in filtered_df.columns:
                            location_col = 'Location' if 'Location' in filtered_df.columns else 'Site'
                            location_summary = filtered_df.groupby(location_col).agg({
                                'Stock Code': 'count',
                                'Value': 'sum',
                                'Qty On Hand': 'sum',
                                'DaysToExpiry': 'mean'
                            }).reset_index()
                            location_summary.to_excel(writer, sheet_name='Location Analysis', index=False)
                    
                    output.seek(0)
                    st.download_button(
                        label="📥 Export Comprehensive Report (Excel)",
                        data=output.getvalue(),
                        file_name=f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        else:
            st.info("📊 Click 'Run Analysis' to start processing your data and generate insights.")

else:
    # Welcome screen with instructions
    st.markdown("""
    ## Welcome to BBD Watch Dashboard! 📦
    
    This dashboard helps you monitor and analyze raw material expiry dates to minimize waste and optimize inventory management.
    
    ### 📋 Required CSV Format:
    Your CSV file should contain the following columns:
    - **BBD**: Best Before Date
    - **Qty On Hand**: Quantity available
    - **UnitCost**: Cost per unit
    - **Stock Code**: Unique identifier
    - **Description**: Item description
    - **Category**: Item category
    
    ### 📋 Optional Columns (for enhanced analysis):
    - **Batch** or **Batch No**: Batch number for traceability
    - **Location** or **Site**: Storage location
    - **MFG Date** or **Manufacture Date**: Manufacturing date
    
    ### 🚀 Features:
    - 📊 Real-time expiry analysis
    - 🔍 Advanced filtering and search
    - 📈 Interactive visualizations
    - 🚨 Critical alerts for expiring items
    - 📥 Export functionality
    
    **Please upload your CSV file using the sidebar to get started!**
    """)
    
    # Add sample data structure
    st.subheader("📄 Sample Data Structure")
    sample_data = pd.DataFrame({
        'Stock Code': ['SKU001', 'SKU002', 'SKU003'],
        'Description': ['Sample Item 1', 'Sample Item 2', 'Sample Item 3'],
        'Category': ['Category A', 'Category B', 'Category A'],
        'BBD': ['2024-06-15', '2024-08-20', '2024-12-01'],
        'Qty On Hand': [100, 50, 200],
        'UnitCost': [10.50, 25.00, 5.75]
    })
    st.dataframe(sample_data, use_container_width=True)
