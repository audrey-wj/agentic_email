import pandas as pd
from openai import AzureOpenAI

def dedupe_properties(df, exclude_cols, dedupe_keys):
    unique_properties = (df
                        .drop(columns=[c for c in exclude_cols if c in df.columns])
                        .drop_duplicates(subset=dedupe_keys)
    )
    return unique_properties



def format_property_for_prompt(row, exclude_fields=None) -> str:
    """Auto-format a single property row (pd.Series or dict), excluding specified fields."""
    exclude_fields = exclude_fields or []
    lines = []
    for key, value in row.items():
        if key.lower().replace("_", "").replace("-", "") in {f.replace("_", "") for f in exclude_fields}:
            continue
        try:
            if value is None or value == "" or pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        label = key.replace("_", " ").title()
        lines.append(f"  {label}: {value}")
    return "- Property:\n" + "\n".join(lines)


def format_customer_for_prompt(row, exclude_fields=None) -> str:
    """Auto-format a single customer row (pd.Series or dict), excluding specified fields."""
    exclude_fields = exclude_fields or []
    lines = []
    for key, value in row.items():
        if key.lower().replace("_", "").replace("-", "") in {f.replace("_", "") for f in exclude_fields}:
            continue
        try:
            if value is None or value == "" or pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        label = key.replace("_", " ").title()
        lines.append(f"  {label}: {value}")
    return "- Customer:\n" + "\n".join(lines)



def build_prompt_simple(property_row, instruction: str) -> str:
    property_block = format_property_for_prompt(property_row)

    return f"""You are a real estate copywriter.

PROPERTY:
{property_block}

Write highlight for this property {instruction}
If any property attributes are 0 or not available, skip that attribute completely, 
do not use the 0 or null values in the writing.
"""


def build_prompt(property_row, instruction: str, customer_row=None
                 , property_exclude_fields=None
                 , customer_exclude_fields=None) -> str:
    # Handle single row (pd.Series or dict) or list of rows

    if isinstance(property_row, (list, pd.DataFrame)):
        if isinstance(property_row, pd.DataFrame):
            property_block = "\n".join(format_property_for_prompt(row, property_exclude_fields) for _, row in property_row.iterrows())
        else:  # list of dicts or pd.Series
            property_block = "\n".join(format_property_for_prompt(r, property_exclude_fields) for r in property_row)
    else:
        property_block = format_property_for_prompt(property_row, property_exclude_fields)


    # Optional customer block
    customer_section = ""
    if customer_row is not None:
        customer_block = format_customer_for_prompt(customer_row, customer_exclude_fields)
        customer_section = f"\nCUSTOMER:\n{customer_block}\n"

    return f"""You are a real estate copywriter.
{customer_section}
PROPERTY:
{property_block}

Write highlight for the property (or properties) {instruction}
If any property attributes are 0 or not available, skip that attribute completely, 
do not use the 0 or null values in the writing.
"""


def write_property_highlight(client, model, prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a real estate copywriter. Write concise, compelling property highlights."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,
        temperature=0.7,
    )
    
    return response.choices[0].message.content.strip()
