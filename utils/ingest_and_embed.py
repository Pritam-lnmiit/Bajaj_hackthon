import os
import uuid
from dotenv import load_dotenv
from upstash_vector import Index
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# Initialize Upstash Index
index = Index(
    url=os.getenv("UPSTASH_VECTOR_URL"),
    token=os.getenv("UPSTASH_VECTOR_TOKEN")
)

def load_documents(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(full_path)
            docs.extend(loader.load())
        elif file.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(full_path)
            docs.extend(loader.load())
    return docs

def embed_and_upload(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embedder = OpenAIEmbeddings()

    for i, chunk in enumerate(chunks):
        try:
            content = chunk.page_content.strip()
            if not content:
                continue

            vector = embedder.embed_query(content)
            doc_id = str(uuid.uuid4())

            # ✅ FIXED: batch-style upload with 'vectors' key
            index.upsert(
                vectors=[{
                    "id": doc_id,
                    "vector": vector,
                    "metadata": {"text": content[:1000]}
                }]
            )

            print(f"✅ Uploaded chunk {i} (ID: {doc_id})")

        except Exception as e:
            print(f"❌ Error on chunk {i}: {str(e)}")

    print(f"✅ Uploaded {len(chunks)} chunks to Upstash Vector!")

if __name__ == "__main__":
    docs = load_documents("data")
    if not docs:
        print("❌ No documents found.")
    else:
        print(f"📄 Loaded {len(docs)} documents. Uploading...")
        embed_and_upload(docs)