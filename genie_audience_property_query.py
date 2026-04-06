import requests
from dotenv import load_dotenv
import os
import pandas as pd
import time

def build_query_prompt(question: str) -> str:
    return f"""
    {question}
    Always include these property detail columns: ListingPrice, StreetNumber,
    StreetName, City, State, County, TransactionType, SQFT. Drop properties that
    do not have all the details.
    Include other columns relevant to the question.
    Drop customers that do not meet the property count threshold.
    """


def genie_audience_property_query(prompt, conversation_id=None):

    load_dotenv()

    DATABRICKS_INSTANCE = os.getenv("DATABRICKS_INSTANCE")
    USER_AUTHENTICATION_TOKEN = os.getenv("USER_AUTHENTICATION_TOKEN")
    GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID")

    headers = {"Authorization": f"Bearer {USER_AUTHENTICATION_TOKEN}"}
    payload = {"content": prompt}   

    # Start or continue a conversation
    if conversation_id:
        url = f"https://{DATABRICKS_INSTANCE}/api/2.0/genie/spaces/{GENIE_SPACE_ID}/conversations/{conversation_id}/messages"
    else:
        url = f"https://{DATABRICKS_INSTANCE}/api/2.0/genie/spaces/{GENIE_SPACE_ID}/start-conversation"


    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    conversation_id = data["conversation_id"]
    message_id = data["message_id"]

    # Poll until COMPLETED (timeout: 300s)
    url = f"https://{DATABRICKS_INSTANCE}/api/2.0/genie/spaces/{GENIE_SPACE_ID}/conversations/{conversation_id}/messages/{message_id}"
    timeout = 300
    elapsed = 0
    while True:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        status = data.get("status")
        print(f"Status: {status} ({elapsed}s elapsed)")
        if status == "COMPLETED":
            break
        if elapsed >= timeout:
            raise TimeoutError(f"Genie query did not complete within {timeout}s. Last status: {status}")
        elapsed += 5
        time.sleep(5)
        


    # Extract IDs
    query = data["attachments"][0]["query"]["query"]
    attachment_id = data["attachments"][0]["attachment_id"]
    statement_id = data["query_result"]["statement_id"]
    # summary = data["attachments"][3]["text"]["content"]

    # Fetch first chunk of query result
    base_url = f"https://{DATABRICKS_INSTANCE}/api/2.0/genie/spaces/{GENIE_SPACE_ID}/conversations/{conversation_id}/messages/{message_id}/query-result/{attachment_id}"
    response = requests.get(base_url, headers=headers)
    response.raise_for_status()
    first = response.json()

    statement = first["statement_response"]
    columns = [col["name"] for col in statement["manifest"]["schema"]["columns"]]
    total_chunks = statement["manifest"].get("total_chunk_count", 1)

    if total_chunks == 0:
        print("Warning: no data returned from query.")
        return {"query": query,
                "conversation_id": conversation_id,
                "attachment": first,
                "result_table": pd.DataFrame(),
                "result_json": {}
                }

    all_rows = list(statement["result"]["data_array"])

    # Fetch additional chunks if any
    for chunk_index in range(1, total_chunks):
        chunk_url = f"https://{DATABRICKS_INSTANCE}/api/2.0/sql/statements/{statement_id}/result/chunks/{chunk_index}"
        chunk_response = requests.get(chunk_url, headers=headers)
        chunk_response.raise_for_status()
        all_rows.extend(chunk_response.json()["data_array"])

    # Build outputs
    result_json = [dict(zip(columns, row)) for row in all_rows]
    df = pd.DataFrame(all_rows, columns=columns)

    return {"query": query, 
            "result_table": df, 
            "result_json": result_json, 
            "conversation_id": conversation_id,
            "attachment": first
            }

