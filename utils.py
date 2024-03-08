import json
from langchain.chains import RetrievalQAWithSourcesChain

def output_response(response) -> None:
    if not response:
        print("There's no response.")
    else:
        print(response)
    print("-----")

def is_answer_formatted_in_json(answer):
    try:
        json.loads(answer, strict=False)
        return True
    except ValueError:
        return False

def format_escape_characters(s):
    return s.replace('"', '\\"').replace("\n", "\\n")


def transform_to_json(result):
    formatted_result_string = format_escape_characters(result["answer"]+result["sources"])
    return f"""
        {{
        "result": "{formatted_result_string}"
        }}"""

def _parse_source_docs(q_and_a_tool: RetrievalQAWithSourcesChain, query: str):
    result = q_and_a_tool({"question": query})
    return transform_to_json(result)