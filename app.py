"""
LLM-Powered Business Intelligence Assistant — Cloud Deploy Edition
Run with: streamlit run app.py
This is the cloud-hosted variant (Groq free-tier API) for live demo purposes.
See the GitHub repo for the fully local, zero-API-key Ollama version, which
keeps all data on-device.
"""
import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
from openai import OpenAI

from query_engine import execute_spec, QuerySpecError
from nl_to_query import SCHEMA_DESCRIPTION, GROQ_MODEL, correct_agg_heuristic, correct_operation_heuristic, correct_metric_heuristic, correct_trend_heuristic

st.set_page_config(page_title="BI Assistant — Hamad Raouf", layout="wide", page_icon="💬")

# ───────────────────────── DATA ─────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("sales_data.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"] = pd.to_datetime(df["Ship Date"])
    return df

df = load_data()

# ───────────────────────── SIDEBAR ─────────────────────────
st.sidebar.title("⚙️ Setup")
st.sidebar.info("This is the **cloud demo** (Groq free-tier API). For a 100% local, "
                 "zero-API-key version where your data never leaves your machine, "
                 "see the [GitHub repo](https://github.com/hamadraouf-3/llm-bi-assistant).")

# Prefer a securely-stored key (Streamlit Secrets) so visitors can try the demo
# without needing their own key. Falls back to manual entry if none is configured.
preset_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""

if preset_key:
    api_key = preset_key
    st.sidebar.success("✅ Ready to chat — no setup needed for this demo.")
else:
    api_key = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Get a free key at console.groq.com/keys — no credit card required."
    )
    st.sidebar.caption("Your key is used only for this session and is never stored.")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Dataset")
st.sidebar.write(f"**{len(df):,}** orders | **{df['Order Date'].min().year}–{df['Order Date'].max().year}**")
st.sidebar.write(f"Regions: {', '.join(df['Region'].unique())}")
st.sidebar.write(f"Categories: {', '.join(df['Category'].unique())}")

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Try asking:")
example_questions = [
    "What were our total sales last year?",
    "Which region is the most profitable?",
    "What are our top 5 best-selling sub-categories?",
    "How much profit did Furniture generate?",
    "Show me the sales trend by year",
]
for q in example_questions:
    st.sidebar.markdown(f"- _{q}_")

st.sidebar.markdown("---")
st.sidebar.caption("Built by Hamad Raouf · AI & Data Science Specialist")

# ───────────────────────── HEADER ─────────────────────────
st.title("💬 LLM-Powered Business Intelligence Assistant")
st.caption("Ask business questions about sales data in plain English. Powered by a cloud LLM (Llama 3.3 via Groq) + a safe, structured query layer (not free-form code execution).")

if not api_key:
    st.warning("👈 Enter a free Groq API key in the sidebar to start chatting. Get one at console.groq.com/keys (no credit card needed).")
    st.info("This assistant translates your question into a validated query spec (never raw code), "
            "runs it safely against the dataset, then asks the LLM to explain the result in plain business language — "
            "and fact-checks the answer before showing it to you.")
    st.stop()

client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

# ───────────────────────── SESSION STATE ─────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "spec_history" not in st.session_state:
    st.session_state.spec_history = []

# ───────────────────────── CORE PIPELINE ─────────────────────────
def question_to_spec(question, history):
    context_block = ""
    if history:
        recent = history[-3:]
        lines = [f'- Q: "{h["question"]}" -> spec: {json.dumps(h["spec"])}' for h in recent]
        context_block = "\n\nRecent conversation for context (resolve follow-ups against this):\n" + "\n".join(lines)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=400,
        messages=[
            {"role": "system", "content": SCHEMA_DESCRIPTION + context_block},
            {"role": "user", "content": question},
        ],
    )
    raw = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
    spec = json.loads(raw)
    spec = correct_trend_heuristic(question, spec)
    spec = correct_operation_heuristic(question, spec)
    spec = correct_agg_heuristic(question, spec)
    spec = correct_metric_heuristic(question, spec)
    return spec


FREQ_DATE_FORMAT = {"Y": "%Y", "M": "%Y-%m", "W": "%Y-%m-%d", "D": "%Y-%m-%d"}


def format_dim_value(value, result):
    """Formats a dimension value for display — dates get a clean format based on
    the trend frequency (e.g. just '2017' for yearly) instead of a full timestamp."""
    if result["result_type"] == "timeseries" and hasattr(value, "strftime"):
        freq = result.get("spec_used", {}).get("freq", "M")
        if freq == "Q":
            return f"{value.year}-Q{value.quarter}"
        fmt = FREQ_DATE_FORMAT.get(freq, "%Y-%m-%d")
        return value.strftime(fmt)
    return str(value)


def _get_top_row(result):
    """Shared helper: correctly finds the highest-value row regardless of result type."""
    data = result["data"]
    dim_col = data.columns[0]
    metric_col = data.columns[1]
    if result["result_type"] == "timeseries":
        top_row = data.loc[data[metric_col].idxmax()]
    else:
        top_row = data.iloc[0]  # groupby/top_n already sorted descending
    return dim_col, metric_col, top_row


def format_result_for_llm(result):
    """
    Builds a clear, pre-digested text summary of the result for the LLM.
    Small local models are unreliable at reading raw tables, so we extract
    the key facts (top/bottom rows) explicitly instead of making the model
    parse a table itself.

    NOTE: for "timeseries" results the rows are sorted chronologically, NOT
    by value — so the top/bottom must be found by value, not row position.
    """
    if result["result_type"] == "scalar":
        return f"Value: {result['data']:,.2f}"

    data = result["data"]
    dim_col = data.columns[0]
    metric_col = data.columns[1]

    lines = [f"Full breakdown ({dim_col} -> {metric_col}):"]
    for _, row in data.iterrows():
        lines.append(f"  - {format_dim_value(row[dim_col], result)}: {row[metric_col]:,.2f}")

    if result["result_type"] == "timeseries":
        top_row = data.loc[data[metric_col].idxmax()]
        bottom_row = data.loc[data[metric_col].idxmin()]
    else:
        top_row = data.iloc[0]
        bottom_row = data.iloc[-1]

    lines.append(f"\nHighest: {format_dim_value(top_row[dim_col], result)} with {top_row[metric_col]:,.2f}")
    if len(data) > 1:
        lines.append(f"Lowest: {format_dim_value(bottom_row[dim_col], result)} with {bottom_row[metric_col]:,.2f}")

    return "\n".join(lines)


def build_fallback_answer(result):
    """Fully deterministic, code-generated answer — used when the LLM hallucinates."""
    if result["result_type"] == "scalar":
        return f"The result is {result['data']:,.2f}."
    dim_col, metric_col, top_row = _get_top_row(result)
    return f"The highest is {format_dim_value(top_row[dim_col], result)} with {metric_col} of {top_row[metric_col]:,.2f}."


def validate_answer(result, answer_text):
    """Checks the LLM's answer actually contains the correct top figure/name."""
    if result["result_type"] == "scalar":
        expected = f"{result['data']:,.2f}"
        return expected in answer_text or expected.replace(",", "") in answer_text.replace(",", "")
    dim_col, metric_col, top_row = _get_top_row(result)
    expected_name = format_dim_value(top_row[dim_col], result)
    expected_value = f"{top_row[metric_col]:,.2f}"
    name_ok = expected_name.lower() in answer_text.lower()
    value_ok = expected_value in answer_text or expected_value.replace(",", "") in answer_text.replace(",", "")
    return name_ok and value_ok


def result_to_answer(question, result):
    data_str = format_result_for_llm(result)

    prompt = f"""The user asked: "{question}"

The data query returned this result (in US dollars unless the metric is Quantity):
{data_str}

Write a concise, business-friendly answer (1-3 sentences) using ONLY the numbers and names shown above.

Strict rules:
- If there is a "Highest" line above, your answer MUST name that exact item and exact number — do not pick a different one.
- Do NOT invent percentages, comparisons, trends, or figures that are not in the data above.
- Do NOT add generic business filler like "this shows strong growth" unless the data above actually contains that comparison.
- If the metric is Sales, Profit, or Discount, format it as a dollar amount.
- Do not mention JSON, code, or query mechanics — just state the finding plainly, like a careful analyst reporting only what was measured."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    llm_answer = response.choices[0].message.content.strip()

    if validate_answer(result, llm_answer):
        return llm_answer
    else:
        return build_fallback_answer(result) + \
            "\n\n_(Note: the local model's phrased answer didn't match the verified data, so a direct fact-checked answer is shown instead.)_"


def maybe_chart(result):
    """Auto-generate a simple chart for table/timeseries results."""
    if result["result_type"] == "table":
        data = result["data"]
        dim_col = data.columns[0]
        metric_col = data.columns[1]
        fig = px.bar(data, x=dim_col, y=metric_col, text_auto='.2s',
                     title=f"{metric_col} by {dim_col}")
        return fig
    if result["result_type"] == "timeseries":
        data = result["data"]
        fig = px.line(data, x=data.columns[0], y=data.columns[1], markers=True,
                       title=f"{data.columns[1]} over time")
        return fig
    return None


def escape_dollars(text):
    """Streamlit's markdown renders $...$ as LaTeX math — escape $ so dollar
    amounts display as plain text instead of being misinterpreted/mangled."""
    return text.replace("$", "\\$")


# ───────────────────────── CHAT DISPLAY ─────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(escape_dollars(msg["content"]))
        if msg.get("chart") is not None:
            st.plotly_chart(msg["chart"], use_container_width=True)
        if msg.get("table") is not None:
            with st.expander("📋 View underlying data"):
                st.dataframe(msg["table"], use_container_width=True)

# ───────────────────────── CHAT INPUT ─────────────────────────
question = st.chat_input("Ask a question about the sales data...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                spec = question_to_spec(question, st.session_state.spec_history)

                if spec.get("operation") == "unsupported":
                    answer = f"I can't answer that with the available data. {spec.get('reason', '')}"
                    st.markdown(escape_dollars(answer))
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    result = execute_spec(df, spec)
                    answer = result_to_answer(question, result)
                    st.markdown(escape_dollars(answer))

                    chart = maybe_chart(result)
                    table = result["data"] if result["result_type"] != "scalar" else None

                    if chart is not None:
                        st.plotly_chart(chart, use_container_width=True)
                    if table is not None:
                        with st.expander("📋 View underlying data"):
                            st.dataframe(table, use_container_width=True)

                    st.session_state.messages.append({
                        "role": "assistant", "content": answer,
                        "chart": chart, "table": table
                    })
                    st.session_state.spec_history.append({"question": question, "spec": spec})

            except QuerySpecError as e:
                err = f"I tried to query something not supported by this dataset: {e}"
                st.markdown(escape_dollars(err))
                st.session_state.messages.append({"role": "assistant", "content": err})
            except json.JSONDecodeError:
                err = "I had trouble understanding how to query that — could you rephrase the question?"
                st.markdown(escape_dollars(err))
                st.session_state.messages.append({"role": "assistant", "content": err})
            except Exception as e:
                err = f"Something went wrong: {e}"
                st.markdown(escape_dollars(err))
                st.session_state.messages.append({"role": "assistant", "content": err})

st.markdown("---")
st.caption("LLM-Powered BI Assistant · Streamlit + Groq (Llama 3.3, cloud demo) · Hamad Raouf — AI & Data Science Specialist")
