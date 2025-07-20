import os
from dotenv import load_dotenv
from upstash_vector import Index
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Initialize Upstash Vector DB Index
index = Index(
    url=os.getenv("UPSTASH_VECTOR_URL"),
    token=os.getenv("UPSTASH_VECTOR_TOKEN")
)

# Load embedding model
embedding_model = OpenAIEmbeddings()

# Main retrieval function
def retrieve_chunks(query: str, k: int = 5) -> list:
    try:
        query_vector = embedding_model.embed_query(query)

        # 🧠 Perform semantic search
        results = index.query(
            vector=query_vector,
            top_k=k,
            include_metadata=True
        )

        if not results:
            print(f"❌ No chunks found for: {query}")
            return []

        # 🪛 Debug logging
        print(f"\n🔍 Query: {query}")
        print(f"🧪 Top match score: {results[0].score:.4f}")
        print(f"📦 Matches found: {len(results)}")

        # ✅ Extract relevant data
        relevant_chunks = []
        for r in results:
            text = r.metadata.get("text", "")
            score = r.score
            relevant_chunks.append({
                "text": text,
                "score": score
            })

        return relevant_chunks

    except Exception as e:
        print(f"❌ Error in retrieve_chunks: {e}")
        return []
