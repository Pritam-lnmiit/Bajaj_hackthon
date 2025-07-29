import pdfplumber
import uuid
import os
from dotenv import load_dotenv
from openai import OpenAI
from upstash_vector import Index

# ✅ Load environment variables from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UPSTASH_URL = os.getenv("UPSTASH_VECTOR_URL")
UPSTASH_TOKEN = os.getenv("UPSTASH_VECTOR_TOKEN")

# ✅ Initialize OpenAI and Upstash clients
client = OpenAI(api_key=OPENAI_API_KEY)
index = Index(url=UPSTASH_URL, token=UPSTASH_TOKEN)

# ✅ PDF path (modify as needed)
pdf_path = r"Data\Bajaj Allianz Health Insurance Complete Guide & FAQ.pdf"

# ✅ Extract all text from PDF
full_text = ""
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        full_text += page.extract_text() + "\n"

# ✅ Normalize colon types (handles unicode)
text = full_text.replace("Q：", "Q:").replace("A：", "A:")

# ✅ Split into Q&A blocks
qa_blocks = [b.strip() for b in text.split("Q:") if b.strip()]
faq_list = []

for block in qa_blocks:
    if "A:" in block:
        question, answer = block.split("A:", 1)
        faq_list.append({
            "question": question.strip().replace("\n", " "),
            "answer": answer.strip().replace("\n", " ")
        })

# ✅ Function to get OpenAI embeddings
def generate_embedding(text: str) -> list:
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return []

# ✅ Upload to Upstash Vector
success = 0
for i, faq in enumerate(faq_list, start=1):
    combined = f"Q: {faq['question']}\nA: {faq['answer']}"
    emb = generate_embedding(combined)
    if emb:
        vector_data = {
            "id": str(uuid.uuid4()),
            "vector": emb,
            "metadata": {
                "question": faq["question"],
                "answer": faq["answer"],
                "type": "bajaj_pdf_faq"
            }
        }
        index.upsert(vector_data)
        print(f"✅ [{i}/{len(faq_list)}] Uploaded: {faq['question'][:60]}...")
        success += 1
    else:
        print(f"❌ Skipped: {faq['question'][:60]}...")

print(f"\n✅ Upload complete: {success}/{len(faq_list)} Q&A pairs sent to Upstash.")
