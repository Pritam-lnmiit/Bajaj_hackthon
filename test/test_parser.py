import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.query_parser_agent import parse_user_query

query = "46M, knee surgery, Pune, 3-month-old policy"
parsed = parse_user_query(query)

print("✅ Parsed Output:\n")
print(parsed)
