"""
NL -> Query Spec translator, using Groq (free-tier cloud inference, OpenAI-compatible API).
This is the cloud-deploy variant of the project (see the main repo for the
fully local Ollama version). Converts a natural-language business question
into the structured JSON spec consumed by query_engine.execute_spec().
"""
import json
import os
from openai import OpenAI

GROQ_MODEL = "llama-3.3-70b-versatile"   # larger cloud model — far fewer of the heuristic
                                          # corrections below actually trigger with this model,
                                          # but they're kept as a safety net regardless.

SCHEMA_DESCRIPTION = """
You translate a business question into a structured JSON query spec. You never write code.

IMPORTANT — dataset date range: the data covers Order Date from 2014-01-01 to 2017-12-30.
There is no data after 2017. Resolve relative time phrases against this range, not the real
current date:
- "last year" / "this year" -> filter Order Date to the year 2017 (the most recent full year
  in the data) using filters [{"field": "Order Date", "op": ">=", "value": "2017-01-01"},
  {"field": "Order Date", "op": "<=", "value": "2017-12-31"}], operation "filter_aggregate".
- "previous year" / "year before that" -> 2016, same pattern.
- If the question does not mention a time period, use "overall_aggregate" across all years
  — do NOT silently filter to one year.

Available dimensions (group_by / filter fields): Region, Category, Sub-Category, Segment,
State, City, Customer Name, Ship Mode, Product Name, Order Date.

Available metrics: Sales, Profit, Quantity, Discount.

IMPORTANT — choosing "metric":
- "best-selling", "top-selling", "sells the most" default to "Sales" (revenue in dollars),
  NOT "Quantity" — unless the question explicitly says "units", "items", or "quantity".
- "most profitable" / "makes the most money" -> "Profit".
- "most ordered" / "highest quantity" / "most units" -> "Quantity".

IMPORTANT — choosing "agg":
- Default to "sum" for almost all business questions: "most profitable", "best-selling",
  "total", "how much did we sell/make" all mean SUM, not max/mean.
- Only use "max" if the question explicitly asks about a single biggest transaction/order
  (e.g. "what was our largest single sale?").
- Only use "mean"/"average" if the question explicitly says "average" or "per order".
- Only use "count" if the question asks "how many orders/transactions".
- When in doubt, use "sum".

Available operations:
- "overall_aggregate": a single number across the whole dataset (optionally filtered)
- "filter_aggregate": a single number after applying filters
- "groupby": breakdown of a metric by one dimension
- "top_n": top N values of a dimension ranked by a metric
- "trend": metric over time (freq: D, W, M, Q, Y)

Respond with ONLY a JSON object, no prose, no markdown fences. Schema:
{
  "operation": "overall_aggregate" | "filter_aggregate" | "groupby" | "top_n" | "trend",
  "metric": "Sales" | "Profit" | "Quantity" | "Discount",
  "agg": "sum" | "mean" | "count" | "max" | "min",
  "group_by": "<dimension>"        (required for groupby / top_n),
  "n": <int>                        (only for top_n, default 5),
  "freq": "M"|"Q"|"Y"|"D"|"W"        (only for trend),
  "sort": "asc" | "desc"            (optional, default desc),
  "filters": [{"field": "<dimension or Order Date>", "op": "=="|"!="|">="|"<=", "value": "<value>"}]
}

If the question cannot be answered with this vocabulary, respond with:
{"operation": "unsupported", "reason": "<short explanation>"}
"""


def correct_metric_heuristic(question: str, spec: dict) -> dict:
    """
    "Best-selling" is ambiguous in plain English but has a standard business meaning:
    revenue (Sales), not units sold. Small local models often default to Quantity instead.
    We override deterministically unless the question explicitly says units/items/quantity.
    """
    q_lower = question.lower()
    explicit_units = any(w in q_lower for w in ["units", "items", "quantity", "how many"])
    if explicit_units:
        return spec

    if any(p in q_lower for p in ["best-selling", "best selling", "top-selling", "top selling", "sells the most"]):
        if spec.get("metric") != "Sales":
            spec = dict(spec)
            spec["metric"] = "Sales"

    return spec


TREND_KEYWORDS = ["trend", "over time", "by year", "by month", "by quarter", "by week",
                   "monthly", "yearly", "quarterly", "weekly", "year over year", "growth"]

FREQ_KEYWORDS = {
    "year": "Y", "yearly": "Y", "annual": "Y",
    "quarter": "Q", "quarterly": "Q",
    "month": "M", "monthly": "M",
    "week": "W", "weekly": "W",
    "day": "D", "daily": "D",
}


def correct_trend_heuristic(question: str, spec: dict) -> dict:
    """
    "Show the trend / X by year/month/etc." requires the 'trend' operation with
    a resampling freq — it is not a groupby. Small local models sometimes invent
    a malformed group_by (e.g. "Year[Order Date]") instead of using trend+freq.
    We deterministically redirect when the question matches a time-trend pattern.
    """
    q_lower = question.lower()
    if not any(kw in q_lower for kw in TREND_KEYWORDS):
        return spec

    if spec.get("operation") == "trend":
        return spec  # already correct

    # Figure out the right freq from the question wording, default to Year
    freq = "Y"
    for word, code in FREQ_KEYWORDS.items():
        if word in q_lower:
            freq = code
            break

    spec = dict(spec)
    spec["operation"] = "trend"
    spec["freq"] = freq
    spec.pop("group_by", None)  # not used by trend
    spec.setdefault("metric", "Sales")
    spec.setdefault("agg", "sum")
    return spec


DIMENSION_KEYWORDS = {
    "region": "Region", "category": "Category", "sub-category": "Sub-Category",
    "subcategory": "Sub-Category", "segment": "Segment", "state": "State",
    "city": "City", "customer": "Customer Name", "ship mode": "Ship Mode",
    "product": "Product Name",
}


def correct_operation_heuristic(question: str, spec: dict) -> dict:
    """
    "Which region/category/etc. is most/least X" requires a breakdown (groupby)
    to identify which one — it cannot be answered with a single overall number.
    Small local models sometimes pick 'overall_aggregate' for these questions,
    which forces them to hallucinate a name since the result contains none.
    We deterministically redirect to 'groupby' when the question pattern matches.
    """
    q_lower = question.lower()
    asks_which = any(w in q_lower for w in ["which ", "what region", "what category", "who is", "what is the top", "what's the top"])
    if not asks_which:
        return spec

    if spec.get("operation") in {"groupby", "top_n"}:
        return spec  # already correct shape

    # Find which dimension the question is actually asking about
    matched_dim = None
    for keyword, dim in DIMENSION_KEYWORDS.items():
        if keyword in q_lower:
            matched_dim = dim
            break

    if matched_dim:
        spec = dict(spec)
        spec["operation"] = "groupby"
        spec["group_by"] = matched_dim
        spec.setdefault("agg", "sum")
        spec.setdefault("sort", "desc")

    return spec


SUM_TRIGGER_PHRASES = [
    "most profitable", "least profitable", "best-selling", "best selling",
    "total", "how much did we", "how much was", "overall", "altogether",
    "combined", "sum of", "revenue", "made in", "sold in"
]


def correct_agg_heuristic(question: str, spec: dict) -> dict:
    """
    Small local models frequently pick agg='max' or 'mean' for phrasings that
    actually mean 'sum' (e.g. "most profitable region" = highest TOTAL profit,
    not highest single order). Rather than trust the model's instruction-following,
    we deterministically override agg to 'sum' when the question matches common
    business phrasings, unless the question explicitly says "average" or "largest
    single"/"biggest single" (which legitimately mean mean/max).
    """
    if spec.get("operation") not in {"groupby", "top_n", "overall_aggregate", "filter_aggregate"}:
        return spec

    q_lower = question.lower()
    explicit_average = any(w in q_lower for w in ["average", "per order", "mean", "typical"])
    explicit_single_max = any(w in q_lower for w in ["largest single", "biggest single", "single largest", "single biggest", "largest order", "biggest order"])
    explicit_count = any(w in q_lower for w in ["how many orders", "how many transactions", "number of orders", "count of"])

    if explicit_average or explicit_single_max or explicit_count:
        return spec  # trust the model here — question is unambiguous

    if any(phrase in q_lower for phrase in SUM_TRIGGER_PHRASES) and spec.get("agg") != "sum":
        spec = dict(spec)
        spec["agg"] = "sum"

    return spec


def get_client(api_key=None):
    return OpenAI(
        api_key=api_key or os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )


def question_to_spec(question: str, client=None, history=None) -> dict:
    """
    history: list of {"question": str, "spec": dict} from previous turns,
    used so follow-up questions like "what about last year?" can resolve
    context (e.g. same metric/dimension, different filter).
    """
    client = client or get_client()

    context_block = ""
    if history:
        recent = history[-3:]  # last 3 turns is enough context, keeps token cost low
        lines = [f'- Q: "{h["question"]}" -> spec: {json.dumps(h["spec"])}' for h in recent]
        context_block = "\n\nRecent conversation for context (resolve pronouns/follow-ups against this):\n" + "\n".join(lines)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=400,
        messages=[
            {"role": "system", "content": SCHEMA_DESCRIPTION + context_block},
            {"role": "user", "content": question},
        ],
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    spec = json.loads(raw)
    spec = correct_trend_heuristic(question, spec)
    spec = correct_operation_heuristic(question, spec)
    spec = correct_agg_heuristic(question, spec)
    spec = correct_metric_heuristic(question, spec)
    return spec


def format_result_for_llm(result):
    """
    Builds a clear, pre-digested text summary of the result for the LLM.
    Small local models are unreliable at reading raw tables, so we extract
    the key facts (top/bottom rows) explicitly instead of making the model
    parse a table itself.

    IMPORTANT: for "table" results (groupby/top_n) the data already comes
    sorted descending by the metric, so row 0 is the highest. For "timeseries"
    results the data is sorted chronologically, NOT by value — so we must
    explicitly find the max/min row by value rather than assume row order.
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
        # Chronological order — find true max/min by value, not by position.
        top_row = data.loc[data[metric_col].idxmax()]
        bottom_row = data.loc[data[metric_col].idxmin()]
    else:
        # groupby/top_n — already sorted descending by execute_spec, row 0 is highest.
        top_row = data.iloc[0]
        bottom_row = data.iloc[-1]

    lines.append(f"\nHighest: {format_dim_value(top_row[dim_col], result)} with {top_row[metric_col]:,.2f}")
    if len(data) > 1:
        lines.append(f"Lowest: {format_dim_value(bottom_row[dim_col], result)} with {bottom_row[metric_col]:,.2f}")

    return "\n".join(lines)


FREQ_DATE_FORMAT = {"Y": "%Y", "Q": "%Y-Q%q", "M": "%Y-%m", "W": "%Y-%m-%d", "D": "%Y-%m-%d"}


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


def build_fallback_answer(question: str, result: dict) -> str:
    """
    A fully deterministic, code-generated answer (no LLM involved).
    Used when the LLM's phrased answer fails the factual validation check —
    guarantees the user never sees a wrong number, even if the local model hallucinates.
    """
    if result["result_type"] == "scalar":
        return f"The result is {result['data']:,.2f}."

    dim_col, metric_col, top_row = _get_top_row(result)
    return f"The highest is {format_dim_value(top_row[dim_col], result)} with {metric_col} of {top_row[metric_col]:,.2f}."


def validate_answer(result: dict, answer_text: str) -> bool:
    """
    Checks that the LLM's answer actually contains the correct top figure/name.
    Small local models can hallucinate even when given the correct data verbatim —
    this is a safety net, not a style check.
    """
    if result["result_type"] == "scalar":
        expected = f"{result['data']:,.2f}"
        return expected in answer_text or expected.replace(",", "") in answer_text.replace(",", "")

    dim_col, metric_col, top_row = _get_top_row(result)
    expected_name = format_dim_value(top_row[dim_col], result)
    expected_value = f"{top_row[metric_col]:,.2f}"
    name_ok = expected_name.lower() in answer_text.lower()
    value_ok = expected_value in answer_text or expected_value.replace(",", "") in answer_text.replace(",", "")
    return name_ok and value_ok


def result_to_answer(question: str, result: dict, client=None) -> str:
    """
    Second LLM call: turn the raw numeric result into a business-friendly answer.
    If the LLM's answer fails factual validation (wrong name/number), we fall back
    to a deterministic, code-generated sentence instead of risking a hallucinated figure.
    """
    client = client or get_client()
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
        # Local model hallucinated — use the guaranteed-correct fallback instead.
        return build_fallback_answer(question, result) + \
            "\n\n_(Note: the local model's phrased answer didn't match the verified data, so a direct fact-checked answer is shown instead.)_"


# ── Self-test (requires GROQ_API_KEY set) ──
if __name__ == "__main__":
    import pandas as pd
    from query_engine import execute_spec, QuerySpecError

    df = pd.read_pickle("sales_clean.pkl")

    test_questions = [
        "What were our total sales last year?",
        "Which region is the most profitable?",
        "What are our top 5 best-selling sub-categories?",
        "How much did we sell in the Furniture category?",
    ]

    for q in test_questions:
        print(f"\n{'='*60}\nQ: {q}")
        try:
            spec = question_to_spec(q)
            print(f"Spec: {spec}")
            if spec.get("operation") == "unsupported":
                print(f"Unsupported: {spec.get('reason')}")
                continue
            result = execute_spec(df, spec)
            answer = result_to_answer(q, result)
            print(f"A: {answer}")
        except QuerySpecError as e:
            print(f"Rejected spec: {e}")
        except Exception as e:
            print(f"Error: {e}")

