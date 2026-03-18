from pathlib import Path
import datetime
import json

def record_tokens_usage(query, context, output):
    ""
    # tokens_counter = tiktoken.encoding_for_model("gpt-4o-mini")            
    tokens_in= 200
    tokens_out= 400
    record={
        "query": query,
        "date": datetime.datetime.now().isoformat(),
        "input_tokens": tokens_in,
        "output_tokens": tokens_out
    }
    print(Path("/"))
    curr_path=Path("./logs/tokens_usage.json")
    print(curr_path)
    curr_path.parent.mkdir(parents=True, exist_ok=True)
    
    if curr_path.exists():
        with open(curr_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    else:
        records = []
    
    records.append(record)
    
    with open(curr_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    
record_tokens_usage("toto", None, None)