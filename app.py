import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="DataForge AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "dataset" not in st.session_state:
    st.session_state.dataset = None

if "processed" not in st.session_state:
    st.session_state.processed = None

if "filename" not in st.session_state:
    st.session_state.filename = ""

if "target" not in st.session_state:
    st.session_state.target = None

if "removed_duplicates" not in st.session_state:
    st.session_state.removed_duplicates = 0

if "missing_target_rows" not in st.session_state:
    st.session_state.missing_target_rows = 0

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] {
font-family: 'Inter', Arial, Helvetica, sans-serif;
}

* {
scrollbar-width: thin;
scrollbar-color: #6c5cff #0a0c14;
}

::-webkit-scrollbar {
width: 9px;
height: 9px;
}

::-webkit-scrollbar-track {
background: #0a0c14;
}

::-webkit-scrollbar-thumb {
background: linear-gradient(180deg, #7d6bff, #4636c9);
border-radius: 20px;
}

@keyframes auraMove {
0% { transform: translate(0px, 0px) scale(1); }
50% { transform: translate(30px, -20px) scale(1.08); }
100% { transform: translate(0px, 0px) scale(1); }
}

@keyframes shimmer {
0% { background-position: -400px 0; }
100% { background-position: 400px 0; }
}

@keyframes floatY {
0%, 100% { transform: translateY(0px); }
50% { transform: translateY(-6px); }
}

@keyframes glowPulse {
0%, 100% { opacity: 0.55; }
50% { opacity: 1; }
}

@keyframes fadeUp {
from { opacity: 0; transform: translateY(14px); }
to { opacity: 1; transform: translateY(0px); }
}

.stApp {
background:
radial-gradient(circle at 5% 5%, rgba(92, 67, 255, 0.25), transparent 25%),
radial-gradient(circle at 95% 10%, rgba(0, 210, 255, 0.18), transparent 25%),
radial-gradient(circle at 50% 100%, rgba(125, 55, 255, 0.14), transparent 30%),
radial-gradient(circle at 30% 60%, rgba(255, 45, 170, 0.06), transparent 35%),
#05060b;
color: #ffffff;
}

.main .block-container * {
animation: fadeUp 0.5s ease both;
}

#MainMenu {
visibility: hidden;
}

footer {
visibility: hidden;
}

header[data-testid="stHeader"] {
background: rgba(5, 6, 11, 0.0) !important;
visibility: visible !important;
height: 3.2rem;
}

[data-testid="stToolbar"] {
visibility: hidden;
}

[data-testid="stDecoration"] {
visibility: hidden;
}

[data-testid="collapsedControl"] {
visibility: visible !important;
color: #ffffff !important;
background: rgba(120, 100, 255, 0.15);
border: 1px solid rgba(150, 135, 255, 0.35);
border-radius: 10px;
padding: 2px;
top: 0.6rem !important;
}

[data-testid="collapsedControl"] svg {
fill: #ffffff !important;
}

[data-testid="baseButton-headerNoPadding"] svg {
fill: #ffffff !important;
}

.block-container {
max-width: 1500px;
padding-top: 20px;
padding-bottom: 80px;
}

section[data-testid="stSidebar"] {
background:
linear-gradient(180deg, #0c0e18 0%, #070910 50%, #05060b 100%);
border-right: 1px solid rgba(255,255,255,0.08);
min-width: 290px !important;
width: 290px !important;
box-shadow: 12px 0 40px rgba(0,0,0,0.45);
}

section[data-testid="stSidebar"] > div {
padding-top: 1.2rem;
}

.sidebar-logo {
font-family: 'Space Grotesk', sans-serif;
font-size: 28px;
font-weight: 900;
letter-spacing: -1px;
background: linear-gradient(90deg, #ffffff, #b9adff);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
margin-bottom: 2px;
animation: floatY 4s ease-in-out infinite;
display: inline-block;
}

.sidebar-sub {
font-size: 10px;
letter-spacing: 1.5px;
color: #737b94;
margin-bottom: 25px;
}

.sidebar-line {
height: 1px;
background: linear-gradient(90deg, transparent, rgba(255,255,255,0.14), transparent);
margin: 18px 0;
}

.sidebar-title {
color: #ffffff;
font-size: 12px;
font-weight: 800;
letter-spacing: 1px;
margin-bottom: 12px;
text-transform: uppercase;
}

.sidebar-step {
position: relative;
padding: 11px 12px 11px 16px;
margin-bottom: 7px;
border-radius: 11px;
background: linear-gradient(135deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
border: 1px solid rgba(255,255,255,0.06);
color: #9da5bc;
font-size: 12px;
font-weight: 600;
transition: all 0.25s cubic-bezier(.2,.9,.3,1.3);
}

.sidebar-step:hover {
transform: translateX(4px);
border-color: rgba(140,125,255,0.45);
color: #e8e6ff;
box-shadow: 0 6px 18px rgba(90,70,255,0.18);
}

.sidebar-step::before {
content: "";
position: absolute;
left: 0;
top: 8px;
bottom: 8px;
width: 3px;
border-radius: 4px;
background: linear-gradient(180deg, #8f7dff, #00d9ff);
opacity: 0.7;
}

.sidebar-number {
color: #8174ff;
font-weight: 800;
margin-right: 7px;
}

.hero {
position: relative;
overflow: hidden;
min-height: 300px;
padding: 55px;
border-radius: 30px;
background: linear-gradient(135deg, rgba(32,27,82,0.98), rgba(10,14,30,0.98));
border: 1px solid rgba(255,255,255,0.09);
box-shadow:
0 35px 100px rgba(0,0,0,0.5),
inset 0 1px 0 rgba(255,255,255,0.08),
0 0 80px rgba(100, 80, 255, 0.08);
transform-style: preserve-3d;
perspective: 1000px;
}

.hero::before {
content: "";
position: absolute;
inset: 0;
background:
repeating-linear-gradient(
115deg,
rgba(255,255,255,0.03) 0px,
rgba(255,255,255,0.03) 1px,
transparent 1px,
transparent 60px
);
pointer-events: none;
}

.hero-glow-one {
position: absolute;
width: 400px;
height: 400px;
border-radius: 50%;
right: -100px;
top: -180px;
background: #6355ff;
filter: blur(120px);
opacity: 0.28;
animation: auraMove 9s ease-in-out infinite;
}

.hero-glow-two {
position: absolute;
width: 280px;
height: 280px;
border-radius: 50%;
left: -130px;
bottom: -180px;
background: #00d9ff;
filter: blur(120px);
opacity: 0.16;
animation: auraMove 12s ease-in-out infinite reverse;
}

.hero-content {
position: relative;
z-index: 5;
}

.hero-badge {
display: inline-block;
padding: 8px 16px;
border-radius: 999px;
background: linear-gradient(90deg, rgba(112,95,255,0.18), rgba(0,217,255,0.10));
border: 1px solid rgba(130,115,255,0.35);
color: #baaeff;
font-size: 11px;
font-weight: 800;
letter-spacing: 0.9px;
margin-bottom: 18px;
box-shadow: 0 0 22px rgba(120,100,255,0.20);
animation: glowPulse 3s ease-in-out infinite;
}

.hero-title {
font-family: 'Space Grotesk', sans-serif;
font-size: 66px;
font-weight: 900;
letter-spacing: -4px;
line-height: 1;
margin-bottom: 18px;
color: white;
text-shadow: 0 8px 40px rgba(110, 90, 255, 0.35);
}

.hero-title span {
background: linear-gradient(90deg, #ffffff, #a69bff, #5de0ff, #a69bff);
background-size: 300% 100%;
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
animation: shimmer 6s linear infinite;
}

.hero-description {
max-width: 750px;
color: #9da6be;
font-size: 16px;
line-height: 1.8;
}

.section {
margin-top: 34px;
margin-bottom: 15px;
display: flex;
align-items: baseline;
gap: 10px;
}

.section h2 {
font-family: 'Space Grotesk', sans-serif;
font-size: 21px;
margin: 0;
color: white;
position: relative;
}

.section h2::after {
content: "";
display: block;
width: 42px;
height: 3px;
margin-top: 8px;
border-radius: 3px;
background: linear-gradient(90deg, #8f7dff, #00d9ff);
}

.section p {
color: #727b92;
font-size: 12px;
margin-top: 5px;
}

.metric {
position: relative;
overflow: hidden;
min-height: 125px;
padding: 22px;
border-radius: 20px;
background: linear-gradient(145deg, rgba(255,255,255,0.07), rgba(255,255,255,0.018));
border: 1px solid rgba(255,255,255,0.08);
box-shadow:
0 20px 50px rgba(0,0,0,0.32),
inset 0 1px 0 rgba(255,255,255,0.05);
transition:
transform 0.35s cubic-bezier(.2,.9,.3,1.3),
box-shadow 0.35s ease,
border-color 0.35s ease;
transform-style: preserve-3d;
will-change: transform;
}

.metric:hover {
transform: translateY(-7px) rotateX(4deg) rotateY(-3deg) scale(1.015);
border-color: rgba(125,110,255,0.45);
box-shadow:
0 30px 70px rgba(0,0,0,0.45),
0 0 40px rgba(100,80,255,0.18);
}

.metric::after {
content: "";
position: absolute;
width: 110px;
height: 110px;
right: -50px;
bottom: -50px;
border-radius: 50%;
background: #6c5cff;
filter: blur(45px);
opacity: 0.12;
}

.metric::before {
content: "";
position: absolute;
top: 0;
left: 0;
right: 0;
height: 1px;
background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent);
}

.metric-label {
color: #8890a8;
font-size: 10px;
font-weight: 800;
letter-spacing: 1.3px;
text-transform: uppercase;
}

.metric-value {
font-family: 'Space Grotesk', sans-serif;
color: #ffffff;
font-size: 32px;
font-weight: 900;
margin-top: 8px;
}

.metric-info {
color: #626b82;
font-size: 11px;
margin-top: 4px;
}

[data-testid="stFileUploader"] {
padding: 15px;
border-radius: 20px;
background: linear-gradient(145deg, rgba(35,35,65,0.80), rgba(10,12,22,0.85));
border: 1.5px dashed rgba(130,115,255,0.55);
transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

[data-testid="stFileUploader"]:hover {
border-color: rgba(160,150,255,0.85);
box-shadow: 0 0 30px rgba(110,90,255,0.15);
}

.stButton > button {
position: relative;
overflow: hidden;
width: 100%;
min-height: 48px;
border-radius: 14px;
border: 1px solid rgba(140,125,255,0.45);
background: linear-gradient(135deg, #7565ff, #4637ca);
color: white;
font-weight: 800;
box-shadow:
0 12px 35px rgba(80,65,255,0.30),
inset 0 1px 0 rgba(255,255,255,0.15);
transition:
transform 0.15s ease,
box-shadow 0.25s ease,
border-color 0.25s ease;
}

.stButton > button:hover {
transform: translateY(-2px);
border-color: rgba(190,180,255,0.7);
box-shadow: 0 18px 45px rgba(80,65,255,0.45);
}

.stButton > button:active {
transform: translateY(0px) scale(0.98);
}

.stDownloadButton > button {
width: 100%;
min-height: 48px;
border-radius: 14px;
background: linear-gradient(135deg, #6758ff, #4536c9);
color: white;
font-weight: 800;
box-shadow: 0 12px 35px rgba(80,65,255,0.30);
transition: transform 0.15s ease, box-shadow 0.25s ease;
}

.stDownloadButton > button:hover {
transform: translateY(-2px);
box-shadow: 0 18px 45px rgba(80,65,255,0.45);
}

div[data-baseweb="select"] > div {
background: rgba(255,255,255,0.04);
border: 1px solid rgba(255,255,255,0.10);
border-radius: 12px;
transition: border-color 0.25s ease;
}

div[data-baseweb="select"] > div:hover {
border-color: rgba(140,125,255,0.5);
}

[data-testid="stDataFrame"] {
border-radius: 17px;
overflow: hidden;
border: 1px solid rgba(255,255,255,0.08);
box-shadow: 0 18px 45px rgba(0,0,0,0.30);
}

.stTabs [data-baseweb="tab-list"] {
gap: 6px;
}

.stTabs [data-baseweb="tab"] {
background: rgba(255,255,255,0.03);
border-radius: 10px 10px 0 0;
border: 1px solid rgba(255,255,255,0.06);
color: #9da5bc;
font-weight: 700;
}

.stTabs [aria-selected="true"] {
background: linear-gradient(135deg, rgba(117,101,255,0.25), rgba(0,217,255,0.10)) !important;
color: #ffffff !important;
border-color: rgba(140,125,255,0.45) !important;
}

.success-box {
padding: 14px 18px;
border-radius: 13px;
background: rgba(40,220,150,0.08);
border: 1px solid rgba(40,220,150,0.25);
color: #79efb8;
font-size: 13px;
font-weight: 700;
box-shadow: 0 10px 30px rgba(40,220,150,0.08);
}

.footer {
margin-top: 70px;
padding-top: 20px;
border-top: 1px solid rgba(255,255,255,0.06);
text-align: center;
color: #4e566b;
font-size: 11px;
letter-spacing: 1.5px;
}

</style>
""",
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown(
        """
<div class="sidebar-logo">
⚡ DataForge
</div>

<div class="sidebar-sub">
AUTOMATED ML DATA ENGINE
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-title">WORKFLOW</div>',
        unsafe_allow_html=True
    )

    steps = [
        ("01", "Upload Dataset"),
        ("02", "Inspect Dataset"),
        ("03", "Select Target"),
        ("04", "Preprocess"),
        ("05", "Validate"),
        ("06", "Export")
    ]

    for number, text in steps:
        st.markdown(
            f"""
<div class="sidebar-step">
<span class="sidebar-number">
{number}
</span>
{text}
</div>
""",
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="sidebar-line"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-title">SYSTEM</div>',
        unsafe_allow_html=True
    )

    if st.session_state.dataset is not None:
        st.success("● Dataset Active")
        st.caption(st.session_state.filename)
    else:
        st.info("○ Waiting for Dataset")

    st.markdown(
        '<div class="sidebar-line"></div>',
        unsafe_allow_html=True
    )

    st.caption("DATAFORGE AI")
    st.caption("Raw Data → ML Ready")

st.markdown(
    """
<div class="hero">

<div class="hero-glow-one"></div>
<div class="hero-glow-two"></div>

<div class="hero-content">

<div class="hero-badge">
⚡ AUTOMATED ML DATA PREPARATION
</div>

<div class="hero-title">
<span>DataForge</span>
</div>

<div class="hero-description">

Turn messy raw datasets into clean,
structured and machine-learning-ready data.

DataForge automatically analyzes your dataset,
detects feature types, handles missing values,
encodes categorical data and prepares the
final dataset for ML workflows.

</div>

</div>

</div>
""",
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="section">
<h2>Import Dataset</h2>
<p>Start by uploading your raw CSV dataset.</p>
</div>
""",
    unsafe_allow_html=True
)

uploaded = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded is not None:

    if uploaded.name != st.session_state.filename:

        try:
            df_new = pd.read_csv(uploaded)

            st.session_state.dataset = df_new
            st.session_state.filename = uploaded.name
            st.session_state.processed = None
            st.session_state.target = None
            st.session_state.removed_duplicates = 0
            st.session_state.missing_target_rows = 0

        except Exception as e:
            st.error(f"Could not read the file: {e}")
            st.stop()

df = st.session_state.dataset

if df is None:

    st.markdown(
        """
<div class="metric">

<div class="metric-label">
DATAFORGE ENGINE
</div>

<div style="
font-size:20px;
font-weight:800;
margin-top:10px;
">
Your ML data pipeline starts here.
</div>

<div style="
color:#737c93;
font-size:13px;
margin-top:8px;
">
Upload a CSV dataset above and DataForge
will analyze and prepare it automatically.
</div>

</div>
""",
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="footer">
DATAFORGE AI • RAW DATA → CLEAN DATA → ML READY
</div>
""",
        unsafe_allow_html=True
    )

    st.stop()

if df.empty:
    st.error("This dataset is empty.")
    st.stop()

if df.shape[1] < 2:
    st.error("Dataset needs at least 2 columns.")
    st.stop()

st.markdown(
    """
<div class="success-box">
✓ Dataset loaded successfully — DataForge engine is ready.
</div>
""",
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="section">
<h2>Dataset Intelligence</h2>
<p>Quick health overview of your raw dataset.</p>
</div>
""",
    unsafe_allow_html=True
)

rows = len(df)
cols = len(df.columns)
missing = int(df.isnull().sum().sum())
duplicates = int(df.duplicated().sum())

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
<div class="metric">
<div class="metric-label">ROWS</div>
<div class="metric-value">{rows:,}</div>
<div class="metric-info">Dataset records</div>
</div>
""",
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f"""
<div class="metric">
<div class="metric-label">COLUMNS</div>
<div class="metric-value">{cols}</div>
<div class="metric-info">Dataset fields</div>
</div>
""",
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f"""
<div class="metric">
<div class="metric-label">MISSING</div>
<div class="metric-value">{missing:,}</div>
<div class="metric-info">Null values</div>
</div>
""",
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        f"""
<div class="metric">
<div class="metric-label">DUPLICATES</div>
<div class="metric-value">{duplicates:,}</div>
<div class="metric-info">Duplicate records</div>
</div>
""",
        unsafe_allow_html=True
    )

st.markdown(
    """
<div class="section">
<h2>Dataset Preview</h2>
<p>Inspect your raw data before preprocessing.</p>
</div>
""",
    unsafe_allow_html=True
)

max_preview = max(1, min(50, len(df)))
min_preview = min(5, max_preview)

preview = st.slider(
    "Rows to preview",
    min_value=min_preview,
    max_value=max_preview,
    value=min(10, max_preview)
)

st.dataframe(
    df.head(preview),
    use_container_width=True,
    hide_index=True
)

st.markdown(
    """
<div class="section">
<h2>Machine Learning Configuration</h2>
<p>Select the column your future ML model should predict.</p>
</div>
""",
    unsafe_allow_html=True
)

target = st.selectbox(
    "Target Column",
    df.columns.tolist()
)

if st.session_state.target is not None and st.session_state.target != target:
    st.session_state.processed = None
    st.session_state.removed_duplicates = 0
    st.session_state.missing_target_rows = 0

st.session_state.target = target

target_series = df[target]

target_unique = target_series.nunique(dropna=True)
target_missing = int(target_series.isnull().sum())

tc1, tc2, tc3 = st.columns(3)

with tc1:
    st.markdown(
        f"""
<div class="metric">
<div class="metric-label">TARGET</div>
<div class="metric-value" style="font-size:20px;">{target}</div>
</div>
""",
        unsafe_allow_html=True
    )

with tc2:
    st.markdown(
        f"""
<div class="metric">
<div class="metric-label">UNIQUE VALUES</div>
<div class="metric-value">{target_unique}</div>
</div>
""",
        unsafe_allow_html=True
    )

with tc3:
    st.markdown(
        f"""
<div class="metric">
<div class="metric-label">TARGET MISSING</div>
<div class="metric-value">{target_missing}</div>
</div>
""",
        unsafe_allow_html=True
    )

X = df.drop(columns=[target]).copy()
y = df[target].copy()

empty_cols = [
    col
    for col in X.columns
    if X[col].isnull().all()
]

if empty_cols:
    X = X.drop(columns=empty_cols)
    st.warning(
        "Completely empty columns detected and excluded: "
        + ", ".join(empty_cols)
    )

numeric_cols = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_cols = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

supported = numeric_cols + categorical_cols

unsupported = [
    col
    for col in X.columns
    if col not in supported
]

if unsupported:

    X = X.drop(columns=unsupported)

    st.warning(
        "Unsupported columns excluded: "
        + ", ".join(unsupported)
    )

    numeric_cols = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

st.markdown(
    """
<div class="section">
<h2>Feature Intelligence</h2>
<p>Automatically detected feature types.</p>
</div>
""",
    unsafe_allow_html=True
)

fc1, fc2, fc3 = st.columns(3)

with fc1:
    st.markdown(
        f"""
<div class="metric">
<div class="metric-label">NUMERICAL</div>
<div class="metric-value">{len(numeric_cols)}</div>
</div>
""",
        unsafe_allow_html=True
    )

with fc2:
    st.markdown(
        f"""
<div class="metric">
<div class="metric-label">CATEGORICAL</div>
<div class="metric-value">{len(categorical_cols)}</div>
</div>
""",
        unsafe_allow_html=True
    )

with fc3:
    st.markdown(
        f"""
<div class="metric">
<div class="metric-label">FEATURES</div>
<div class="metric-value">{len(X.columns)}</div>
</div>
""",
        unsafe_allow_html=True
    )

with st.expander("View detected columns"):

    left, right = st.columns(2)

    with left:
        st.markdown("#### Numerical")
        st.write(
            numeric_cols
            if numeric_cols
            else "None"
        )

    with right:
        st.markdown("#### Categorical")
        st.write(
            categorical_cols
            if categorical_cols
            else "None"
        )

st.markdown(
    """
<div class="section">
<h2>Data Quality Report</h2>
<p>Column-level quality analysis.</p>
</div>
""",
    unsafe_allow_html=True
)

quality = pd.DataFrame({
    "Column": df.columns,
    "Type": [
        str(df[col].dtype)
        for col in df.columns
    ],
    "Missing": [
        int(df[col].isnull().sum())
        for col in df.columns
    ],
    "Missing %": [
        round(
            df[col].isnull().mean() * 100,
            2
        )
        for col in df.columns
    ],
    "Unique": [
        int(df[col].nunique())
        for col in df.columns
    ]
})

st.dataframe(
    quality,
    use_container_width=True,
    hide_index=True
)

st.markdown(
    """
<div class="section">
<h2>Preprocessing Engine</h2>
<p>Configure how DataForge prepares your ML features.</p>
</div>
""",
    unsafe_allow_html=True
)

p1, p2 = st.columns(2)

with p1:

    numeric_imputation = st.selectbox(
        "Numerical Imputation",
        [
            "Median",
            "Mean"
        ]
    )

    scaling = st.checkbox(
        "Standardize numerical features",
        value=True
    )

with p2:

    categorical_imputation = st.selectbox(
        "Categorical Imputation",
        [
            "Most Frequent",
            "Constant: Unknown"
        ]
    )

    duplicate_removal = st.checkbox(
        "Remove duplicate rows",
        value=True
    )

transformers = []

if numeric_cols:

    if numeric_imputation == "Median":
        strategy = "median"
    else:
        strategy = "mean"

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy=strategy
            )
        )
    ]

    if scaling:
        numeric_steps.append(
            (
                "scaler",
                StandardScaler()
            )
        )

    numeric_pipeline = Pipeline(
        numeric_steps
    )

    transformers.append(
        (
            "numeric",
            numeric_pipeline,
            numeric_cols
        )
    )

if categorical_cols:

    if categorical_imputation == "Most Frequent":

        cat_imputer = SimpleImputer(
            strategy="most_frequent"
        )

    else:

        cat_imputer = SimpleImputer(
            strategy="constant",
            fill_value="Unknown"
        )

    categorical_pipeline = Pipeline([
        (
            "imputer",
            cat_imputer
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    transformers.append(
        (
            "categorical",
            categorical_pipeline,
            categorical_cols
        )
    )

if len(X.columns) == 0:
    st.error("No usable features were detected.")
    st.stop()

if not transformers:
    st.error("Could not build preprocessing pipeline.")
    st.stop()

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)

st.markdown(
    """
<div class="section">
<h2>Forge ML Dataset</h2>
<p>Run the complete preprocessing pipeline.</p>
</div>
""",
    unsafe_allow_html=True
)

forge = st.button(
    "⚡ FORGE ML-READY DATASET"
)

if forge:

    try:

        with st.spinner(
            "DataForge is engineering your dataset..."
        ):

            valid_rows = y.notna()

            X_valid = X.loc[
                valid_rows
            ].copy()

            y_valid = y.loc[
                valid_rows
            ].copy()

            missing_target_rows = int(
                (~valid_rows).sum()
            )

            transformed = preprocessor.fit_transform(
                X_valid
            )

            feature_names = (
                preprocessor
                .get_feature_names_out()
            )

            result = pd.DataFrame(
                transformed,
                columns=feature_names
            )

            result[target] = (
                y_valid
                .reset_index(drop=True)
            )

            removed_duplicates = 0

            if duplicate_removal:

                before = len(result)

                result = (
                    result
                    .drop_duplicates()
                    .reset_index(drop=True)
                )

                removed_duplicates = (
                    before - len(result)
                )

            st.session_state.processed = result
            st.session_state.removed_duplicates = removed_duplicates
            st.session_state.missing_target_rows = missing_target_rows

            st.success(
                "✓ Dataset successfully forged into ML-ready format!"
            )

    except Exception as e:

        st.error(
            "DataForge encountered an error."
        )

        st.exception(e)

result = st.session_state.processed

if result is not None:

    st.markdown(
        """
<div class="section">
<h2>Forge Results</h2>
<p>Your transformed machine-learning-ready dataset.</p>
</div>
""",
        unsafe_allow_html=True
    )

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.markdown(
            f"""
<div class="metric">
<div class="metric-label">ORIGINAL FEATURES</div>
<div class="metric-value">{len(X.columns)}</div>
</div>
""",
            unsafe_allow_html=True
        )

    with r2:
        st.markdown(
            f"""
<div class="metric">
<div class="metric-label">ML FEATURES</div>
<div class="metric-value">{result.shape[1] - 1}</div>
</div>
""",
            unsafe_allow_html=True
        )

    with r3:
        st.markdown(
            f"""
<div class="metric">
<div class="metric-label">FINAL ROWS</div>
<div class="metric-value">{result.shape[0]:,}</div>
</div>
""",
            unsafe_allow_html=True
        )

    with r4:

        expansion = (
            result.shape[1] - 1
            - len(X.columns)
        )

        st.markdown(
            f"""
<div class="metric">
<div class="metric-label">FEATURE EXPANSION</div>
<div class="metric-value">{expansion:+}</div>
</div>
""",
            unsafe_allow_html=True
        )

    tab1, tab2, tab3 = st.tabs(
        [
            "📊 ML Ready Data",
            "🔍 Features",
            "📋 Summary"
        ]
    )

    with tab1:

        st.dataframe(
            result.head(25),
            use_container_width=True,
            hide_index=True
        )

    with tab2:

        feature_list = pd.DataFrame({
            "Feature": result.columns,
            "Position": range(
                1,
                len(result.columns) + 1
            )
        })

        st.dataframe(
            feature_list,
            use_container_width=True,
            hide_index=True
        )

    with tab3:

        summary = pd.DataFrame({

            "Processing": [

                "Original Rows",
                "Final Rows",
                "Original Features",
                "Final ML Features",
                "Numerical Features",
                "Categorical Features",
                "Missing Target Rows",
                "Duplicate Rows Removed"

            ],

            "Result": [

                len(df),
                len(result),
                len(X.columns),
                result.shape[1] - 1,
                len(numeric_cols),
                len(categorical_cols),
                st.session_state.missing_target_rows,
                st.session_state.removed_duplicates

            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

    st.markdown(
        """
<div class="section">
<h2>Export</h2>
<p>Download your final ML-ready dataset.</p>
</div>
""",
        unsafe_allow_html=True
    )

    csv = (
        result
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        label="⬇ DOWNLOAD ML-READY CSV",
        data=csv,
        file_name="dataforge_ml_ready.csv",
        mime="text/csv"
    )

st.markdown(
    """
<div class="footer">

DATAFORGE AI
&nbsp; • &nbsp;
RAW DATA
→
ANALYZE
→
CLEAN
→
TRANSFORM
→
ML READY

</div>
""",
    unsafe_allow_html=True
)