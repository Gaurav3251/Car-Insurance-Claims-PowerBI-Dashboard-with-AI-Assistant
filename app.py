import html
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError
from google.genai import types


DATA_FILE = "Car_Insurance_Claim.csv"
CLAIM_COLUMN = "Outcome"


st.set_page_config(
    page_title="Car Insurance Claims Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()


def get_secret(name: str) -> str:
    env_value = (os.getenv(name, "") or "").strip()
    if env_value:
        return env_value

    try:
        return (st.secrets.get(name, "") or "").strip()
    except Exception:
        return ""


GEMINI_API_KEY = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
GEMINI_MODEL = get_secret("GEMINI_MODEL")
DEFAULT_GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
]


def get_model_candidates() -> list[str]:
    configured_models = [
        model.strip()
        for model in GEMINI_MODEL.split(",")
        if model.strip()
    ] if GEMINI_MODEL else []
    return list(dict.fromkeys(configured_models + DEFAULT_GEMINI_MODELS))


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #0b1020; color: #e8eefc; }
    [data-testid="stSidebar"] { background: #111827 !important; border-right: 1px solid #263244; }
    [data-testid="stSidebar"] * { color: #d6deee !important; }

    .hero {
        background: radial-gradient(circle at top left, rgba(59,130,246,.35), transparent 35%),
                    linear-gradient(135deg, #111827, #172033);
        border: 1px solid #273449;
        border-radius: 22px;
        padding: 26px 30px;
        margin-bottom: 20px;
    }
    .eyebrow { color: #93c5fd; font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
    .hero h1 { margin: 6px 0 8px 0; font-size: 2rem; line-height: 1.15; color: #f8fbff; }
    .hero p { margin: 0; color: #b8c4d8; max-width: 780px; line-height: 1.55; }

    .status-card, .guide-card {
        background: #111827;
        border: 1px solid #273449;
        border-radius: 16px;
        padding: 16px 18px;
    }
    .status-card strong { color: #f8fbff; }
    .muted { color: #94a3b8; font-size: .88rem; }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(130px, 1fr));
        gap: 12px;
        margin: 12px 0 20px 0;
    }
    .metric-card {
        background: #111827;
        border: 1px solid #273449;
        border-radius: 16px;
        padding: 15px;
    }
    .metric-card .value { display: block; color: #60a5fa; font-size: 1.45rem; font-weight: 800; }
    .metric-card .label { display: block; color: #9caec8; font-size: .78rem; margin-top: 4px; }

    .chat-note {
        background: rgba(37, 99, 235, .12);
        border: 1px solid rgba(96, 165, 250, .35);
        border-radius: 14px;
        padding: 13px 15px;
        color: #dbeafe;
        margin-bottom: 14px;
    }
    .user-bubble { display: flex; justify-content: flex-end; margin: 13px 0; }
    .assistant-bubble { display: flex; justify-content: flex-start; margin: 13px 0; }
    .bubble {
        max-width: 78%;
        border-radius: 18px;
        padding: 13px 16px;
        line-height: 1.6;
        font-size: .93rem;
        white-space: pre-wrap;
    }
    .user-bubble .bubble {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border-bottom-right-radius: 5px;
    }
    .assistant-bubble .bubble {
        background: #111827;
        color: #e5edf9;
        border: 1px solid #273449;
        border-bottom-left-radius: 5px;
    }
    @media (max-width: 1000px) {
        .metric-grid { grid-template-columns: repeat(2, minmax(130px, 1fr)); }
        .bubble { max-width: 92%; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()
    return df


def claim_rate(series: pd.Series) -> float:
    return float(series.mean() * 100) if len(series) else 0.0


def format_pct(value: float) -> str:
    return f"{value:.1f}%"


def detect_risk_band(row: pd.Series) -> str:
    violations = row.get("Speeding Violations", 0) or 0
    duis = row.get("DUIS", 0) or 0
    accidents = row.get("Past Accidents", 0) or 0

    if duis > 0 or accidents >= 2 or violations >= 3:
        return "High"
    if accidents == 1 or violations in (1, 2):
        return "Medium"
    return "Low"


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared[CLAIM_COLUMN] = pd.to_numeric(prepared[CLAIM_COLUMN], errors="coerce").fillna(0).astype(int)
    prepared["Risk Band"] = prepared.apply(detect_risk_band, axis=1)
    return prepared


def grouped_rate(df: pd.DataFrame, column: str, limit: int | None = None) -> pd.DataFrame:
    result = (
        df.groupby(column, dropna=False)
        .agg(
            policies=(CLAIM_COLUMN, "size"),
            claims=(CLAIM_COLUMN, "sum"),
            claim_rate=(CLAIM_COLUMN, "mean"),
        )
        .reset_index()
    )
    result["claim_rate"] = (result["claim_rate"] * 100).round(1)
    result = result.sort_values(["claim_rate", "policies"], ascending=[False, False])
    return result.head(limit) if limit else result


def build_context(df: pd.DataFrame) -> str:
    rows = len(df)
    total_claims = int(df[CLAIM_COLUMN].sum())
    overall_rate = claim_rate(df[CLAIM_COLUMN])
    avg_mileage = df["Annual Mileage"].mean() if "Annual Mileage" in df.columns else None

    sections = [
        "DATASET: Car Insurance Claims",
        f"ROWS: {rows:,}",
        f"COLUMNS: {', '.join(df.columns)}",
        f"OVERALL: {total_claims:,} claims from {rows:,} policies; claim rate {overall_rate:.1f}%.",
    ]

    if avg_mileage is not None:
        sections.append(f"ANNUAL MILEAGE: average {avg_mileage:,.0f}; median {df['Annual Mileage'].median():,.0f}.")

    for column in [
        "Age",
        "Gender",
        "Income",
        "Education",
        "Driving Experience",
        "Vehicle Year",
        "Vehicle Type",
        "Risk Band",
        "Postal Code",
    ]:
        if column in df.columns:
            grouped = grouped_rate(df, column, limit=12)
            sections.append(f"CLAIM RATE BY {column.upper()}:\n{grouped.to_string(index=False)}")

    for column in ["Speeding Violations", "DUIS", "Past Accidents"]:
        if column in df.columns:
            corr = df[[column, CLAIM_COLUMN]].corr(numeric_only=True).iloc[0, 1]
            sections.append(f"CORRELATION: {column} vs claim outcome = {corr:.3f}.")

    if {"Age", "Income"}.issubset(df.columns):
        cross = (
            df.groupby(["Age", "Income"])[CLAIM_COLUMN]
            .mean()
            .mul(100)
            .round(1)
            .sort_values(ascending=False)
            .head(12)
        )
        sections.append(f"TOP AGE x INCOME CLAIM-RATE SEGMENTS:\n{cross.to_string()}")

    return "\n\n".join(sections)


SYSTEM_PROMPT = """You are a Smart Financial Data Assistant for a car insurance claims analytics project.

Use only the provided dataset context. Do not invent numbers. If the context cannot answer the question, say what is missing.

Answer like a business analyst:
1. Direct answer first.
2. Explain what the result means for claim risk, pricing, operations, or customer strategy.
3. State the data used at the end.
4. Mention limitations when relevant, especially sample size, missing columns, correlation not proving causation, and possible bias.

Keep answers concise, structured, and presentation-ready. Never ask for API keys, uploads, Python code, or implementation details from the user."""


def ask_gemini(question: str, data_context: str, history: list[dict[str, str]]) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)
    recent_history = "\n".join(
        f"{message['role'].upper()}: {message['content']}" for message in history[-8:]
    )
    prompt = f"""RECENT CONVERSATION:
{recent_history or "No previous conversation."}

DATA CONTEXT COMPUTED FROM THE CSV:
{data_context}

USER QUESTION:
{question}"""
    api_errors = []

    for model_name in get_model_candidates():
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
            )
            return response.text
        except ClientError as error:
            error_text = str(error)
            error_status = getattr(error, "code", None) or getattr(error, "status", None)
            is_quota_error = error_status == 429 or "429" in error_text or "RESOURCE_EXHAUSTED" in error_text
            is_model_error = error_status == 404 or "404" in error_text or "NOT_FOUND" in error_text

            api_errors.append(f"{model_name}: {error_status or 'API error'}")
            if not is_quota_error and not is_model_error:
                return (
                    "Gemini returned an API error, so I could not generate the AI answer.\n\n"
                    f"Error details: `{error_text}`. Check your API key, model name, quota, and billing/project settings."
                )
        except Exception as error:
            return (
                "The assistant hit an unexpected Gemini/connection error, but the app is still running.\n\n"
                f"Error details: `{str(error)}`."
            )

    return (
        "I could not generate an AI answer because none of the configured Gemini models are available for this API key/project.\n\n"
        "Most likely causes:\n"
        "- The model name in `.env` is unavailable for your key.\n"
        "- Your free-tier quota is exhausted.\n"
        "- The API project/billing settings do not allow these models yet.\n\n"
        f"Tried models: `{', '.join(get_model_candidates())}`.\n"
        f"API results: `{'; '.join(api_errors)}`.\n\n"
        "Fix: open Google AI Studio, choose a model that works with your key, then add it to `.env` as "
        "`GEMINI_MODEL=that-model-name`. You can also remove `GEMINI_MODEL` and let the app try its fallback list."
    )


def render_message(role: str, content: str) -> None:
    safe_content = html.escape(content).replace("\n", "<br>")
    wrapper = "user-bubble" if role == "user" else "assistant-bubble"
    st.markdown(f'<div class="{wrapper}"><div class="bubble">{safe_content}</div></div>', unsafe_allow_html=True)


def render_metrics(df: pd.DataFrame) -> None:
    total = len(df)
    claims = int(df[CLAIM_COLUMN].sum())
    rate = claim_rate(df[CLAIM_COLUMN])
    avg_mileage = df["Annual Mileage"].mean() if "Annual Mileage" in df.columns else 0
    high_risk = int((df["Risk Band"] == "High").sum()) if "Risk Band" in df.columns else 0

    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card"><span class="value">{total:,}</span><span class="label">Policies Analyzed</span></div>
            <div class="metric-card"><span class="value">{claims:,}</span><span class="label">Claims Filed</span></div>
            <div class="metric-card"><span class="value">{rate:.1f}%</span><span class="label">Overall Claim Rate</span></div>
            <div class="metric-card"><span class="value">{avg_mileage:,.0f}</span><span class="label">Avg Annual Mileage</span></div>
            <div class="metric-card"><span class="value">{high_risk:,}</span><span class="label">High-Risk Drivers</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(df: pd.DataFrame) -> None:
    with st.sidebar:
        st.markdown("## Project Assistant")
        st.markdown(
            f"""
            <div class="status-card">
                <strong>Dataset connected</strong><br>
                <span class="muted">{DATA_FILE}</span><br><br>
                <strong>{len(df):,}</strong> rows · <strong>{len(df.columns)}</strong> columns
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Backend status")
        if GEMINI_API_KEY:
            st.success("Gemini key loaded securely from backend config.")
            st.caption(f"Model fallback order: `{', '.join(get_model_candidates())}`")
        else:
            st.error("Missing backend Gemini key.")
            st.caption("Add GEMINI_API_KEY to `.env` or Streamlit secrets. It is never requested in the UI.")

        st.markdown("### Useful questions")
        for question in EXAMPLE_QUESTIONS:
            if st.button(question, use_container_width=True):
                st.session_state.pending_question = question
                st.rerun()

        st.markdown("---")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        with st.expander("Columns used"):
            st.write(", ".join(df.columns))


EXAMPLE_QUESTIONS = [
    "What is the overall claim rate and why does it matter?",
    "Which age group has the highest claim rate?",
    "How do past accidents affect claim risk?",
    "Which income segment looks riskiest?",
    "Compare claim risk by vehicle year.",
    "Give me 3 business recommendations from this data.",
]


try:
    raw_df = load_data()
except FileNotFoundError:
    st.error(f"Dataset not found: `{DATA_FILE}`. Keep the CSV in the same folder as `app.py`.")
    st.stop()

df = prepare_data(raw_df)
render_sidebar(df)

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Smart Financial Data Assistant</div>
        <h1>Car Insurance Claims Analytics</h1>
        <p>
            Ask natural-language questions about claim risk, customer segments, driving behavior,
            and business recommendations. The assistant uses the connected CSV only; no repeated uploads,
            and no API key exposure in the frontend.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

render_metrics(df)

left, right = st.columns([1.6, 1])
with left:
    st.markdown("### Ask the assistant")
    st.markdown(
        """
        <div class="chat-note">
            Try questions about summaries, rankings, filters, segment comparisons, insights, and recommendations.
            If a question needs data that is not in the CSV, the assistant should say that clearly.
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown("### Assistant scope")
    st.markdown(
        """
        <div class="guide-card">
            <strong>Works well for</strong><br>
            <span class="muted">Claim rate, rankings, segment comparison, risk drivers, recommendations.</span><br><br>
            <strong>Will not answer reliably</strong><br>
            <span class="muted">Profitability, premium amounts, future forecasts, or geography beyond postal code unless those fields exist.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.info("Start with one of the sidebar examples, or ask your own question below.")

for message in st.session_state.messages:
    render_message(message["role"], message["content"])

question = st.chat_input("Ask about claim risk, segments, trends, or recommendations...")

if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    if not GEMINI_API_KEY:
        answer = (
            "I cannot run the AI response yet because `GEMINI_API_KEY` is missing from backend configuration. "
            "Add it to `.env` or Streamlit secrets, then restart the app. The key should not be pasted into the UI."
        )
    else:
        with st.spinner("Analyzing the connected claims dataset..."):
            context = build_context(df)
            answer = ask_gemini(question, context, st.session_state.messages[:-1])

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
