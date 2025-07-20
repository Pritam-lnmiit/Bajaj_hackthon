import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.web_search_agent import search_policy_location

query = "waiting period for knee surgery"
location = "Pune"

results = search_policy_location(query, location)

for i, r in enumerate(results):
    print(f"\n🔗 Result {i+1}: {r['title']}")
    print(r["url"])
    print(r["snippet"])
