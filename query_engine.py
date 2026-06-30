"""
Query Executor — the safe, deterministic execution layer.
Takes a structured JSON "query spec" (never raw code) and runs it
against the sales dataframe using a fixed, audited set of operations.
"""
import pandas as pd

VALID_METRICS = {"Sales", "Profit", "Quantity", "Discount"}
VALID_DIMENSIONS = {"Region", "Category", "Sub-Category", "Segment", "State",
                     "City", "Customer Name", "Ship Mode", "Product Name"}
VALID_AGGS = {"sum", "mean", "count", "max", "min"}


class QuerySpecError(ValueError):
    pass


def validate_spec(spec: dict):
    """Reject anything outside the allowed vocabulary before it touches the data."""
    op = spec.get("operation")
    if op not in {"groupby", "filter_aggregate", "trend", "top_n", "overall_aggregate"}:
        raise QuerySpecError(f"Unsupported operation: {op}")

    metric = spec.get("metric")
    if metric not in VALID_METRICS:
        raise QuerySpecError(f"Unsupported metric: {metric}")

    agg = spec.get("agg", "sum")
    if agg not in VALID_AGGS:
        raise QuerySpecError(f"Unsupported aggregation: {agg}")

    group_by = spec.get("group_by")
    if group_by and group_by not in VALID_DIMENSIONS:
        raise QuerySpecError(f"Unsupported group_by dimension: {group_by}")

    filters = spec.get("filters") or []
    for f in filters:
        if f.get("field") not in (VALID_DIMENSIONS | {"Order Date", "Region", "Category"}):
            raise QuerySpecError(f"Unsupported filter field: {f.get('field')}")


def execute_spec(df: pd.DataFrame, spec: dict) -> dict:
    """
    Executes a validated query spec against the dataframe.
    Returns a dict with: result_type, data (DataFrame or scalar), spec_used.
    """
    validate_spec(spec)
    working = df.copy()

    # Apply filters first (e.g. date range, category = "Furniture")
    for f in spec.get("filters", []):
        field, op, value = f["field"], f.get("op", "=="), f["value"]
        if field == "Order Date":
            working = working[working["Order Date"] >= pd.to_datetime(value)] if op == ">=" else \
                      working[working["Order Date"] <= pd.to_datetime(value)] if op == "<=" else working
        else:
            if op == "==":
                working = working[working[field] == value]
            elif op == "!=":
                working = working[working[field] != value]

    operation = spec["operation"]
    metric = spec["metric"]
    agg = spec.get("agg", "sum")

    if operation == "overall_aggregate":
        value = getattr(working[metric], agg)()
        return {"result_type": "scalar", "data": value, "spec_used": spec}

    if operation == "groupby":
        group_by = spec["group_by"]
        grouped = working.groupby(group_by)[metric].agg(agg).reset_index()
        grouped = grouped.sort_values(metric, ascending=spec.get("sort", "desc") == "asc")
        return {"result_type": "table", "data": grouped, "spec_used": spec}

    if operation == "top_n":
        group_by = spec["group_by"]
        n = spec.get("n", 5)
        grouped = working.groupby(group_by)[metric].agg(agg).reset_index()
        grouped = grouped.sort_values(metric, ascending=False).head(n)
        return {"result_type": "table", "data": grouped, "spec_used": spec}

    if operation == "trend":
        freq_map = {"M": "ME", "Q": "QE", "Y": "YE", "D": "D", "W": "W"}
        freq = freq_map.get(spec.get("freq", "M"), "ME")
        ts = working.set_index("Order Date").resample(freq)[metric].agg(agg).reset_index()
        return {"result_type": "timeseries", "data": ts, "spec_used": spec}

    if operation == "filter_aggregate":
        value = getattr(working[metric], agg)()
        return {"result_type": "scalar", "data": value, "spec_used": spec}

    raise QuerySpecError(f"Unhandled operation: {operation}")


# ── Self-test with hand-crafted specs (no LLM yet) ──
if __name__ == "__main__":
    df = pd.read_pickle("sales_clean.pkl")

    tests = [
        {"operation": "overall_aggregate", "metric": "Sales", "agg": "sum"},
        {"operation": "groupby", "group_by": "Region", "metric": "Profit", "agg": "sum", "sort": "desc"},
        {"operation": "top_n", "group_by": "Sub-Category", "metric": "Sales", "agg": "sum", "n": 5},
        {"operation": "filter_aggregate", "metric": "Sales", "agg": "sum",
         "filters": [{"field": "Category", "op": "==", "value": "Furniture"}]},
        {"operation": "trend", "metric": "Sales", "agg": "sum", "freq": "Y"},
    ]

    for t in tests:
        result = execute_spec(df, t)
        print(f"\n--- {t['operation']} ---")
        print(result["data"] if result["result_type"] != "scalar" else f"Value: {result['data']:,.2f}")

    # Test that an invalid spec is rejected
    try:
        execute_spec(df, {"operation": "groupby", "group_by": "Customer ID", "metric": "Sales", "agg": "sum"})
        print("ERROR: should have raised")
    except QuerySpecError as e:
        print(f"\nCorrectly rejected invalid spec: {e}")
