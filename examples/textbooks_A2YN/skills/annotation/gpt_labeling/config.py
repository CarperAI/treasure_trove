{
  "schema": 1,
  "description": "Uses an LLM to evaluate the quality of a code excerpt.",
  "type": "completion",
  "completion": {
    "max_tokens": 500,
    "temperature": 0.3,
    "top_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    },
  "input": {
    "parameters": [
      {
        "name": "code",
        "description": "The code to evaluate",
        "defaultValue": ""
      }
    ]
  }
}