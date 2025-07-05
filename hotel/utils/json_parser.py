import re
import json
from typing import Dict

def parse_json_from_response(text: str) -> Dict:
    try:
        json_match = re.search(r"```json\n(\{.*?\})\n```", text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
        else:
            json_text_match = re.search(r"(\{.*?\})", text, re.DOTALL)
            if json_text_match:
                json_text = json_text_match.group(1)
            else:
                return {"error": "No JSON object found in the response"}
        return json.loads(json_text)
    except Exception as e:
        return {"error": f"Parsing error: {str(e)}", "raw_response": text}
