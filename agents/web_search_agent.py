
import serpapi
from dotenv import load_dotenv
import os

load_dotenv()

def search_policy_location(query: str, location: str) -> list:
    try:
        api_key = os.getenv("SERPAPI_KEY")
        if not api_key:
            raise ValueError("SERPAPI_KEY not found in .env")
        
        search_query = f"{query} insurance policy coverage {location}"
        params = {
            "engine": "google",
            "q": search_query,
            "api_key": api_key,
            "num": 5
        }
        results = serpapi.search(params)
        web_results = [
            {
                "title": result.get("title", ""),
                "snippet": result.get("snippet", ""),
                "link": result.get("link", ""),
                "source": "web"
            }
            for result in results.get("organic_results", [])
        ]
        return web_results
    except Exception as e:
        print(f"Web search error: {str(e)}")
        return []
