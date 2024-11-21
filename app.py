from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
from typing import List
import chromadb
from transformers import AutoTokenizer, AutoModel
from PyPDF2 import PdfReader
from docx import Document
import openpyxl
import os

app = FastAPI()

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient()
collections = chroma_client.list_collections()
c = chroma_client.get_or_create_collection("66f9c55c4377cd2b098aa8b7")
print("Collections:", c.get(include=['embeddings', 'documents', 'metadatas']))

# Load LaBSE model
LOCAL_MODEL_PATH = "./models/LaBSE"
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModel.from_pretrained(LOCAL_MODEL_PATH)

# Function to compute embeddings
def generate_embedding(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Helper function to extract text from different file types
def extract_text(file_path: str):
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif file_extension == ".pdf":
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    elif file_extension == ".docx":
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    elif file_extension in [".xls", ".xlsx"]:
        workbook = openpyxl.load_workbook(file_path)
        sheet = workbook.active
        rows = list(sheet.iter_rows(values_only=True))
        return "\n".join(["\t".join([str(cell) for cell in row if cell is not None]) for row in rows])
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

# Endpoint to handle file embedding generation
class EmbeddingRequest(BaseModel):
    tenantId: str
    filePath: str
    hash: str

@app.post("/generate-embeddings")
async def generate_embeddings(request: EmbeddingRequest):
    print("Tenant ID:", request.tenantId)
    FILE_UPLOAD_PATH = "/Users/businessfundamentals/business_fundamentals/nitrogen-backend/"
    try:
        # Extract text from the file
        text = extract_text(FILE_UPLOAD_PATH + request.filePath)
        print("Extracted Text:", text)

        # Generate embeddings
        embedding = generate_embedding(text)
        print("Generated Embedding:", embedding)

        # Store embeddings in ChromaDB
        collection = chroma_client.get_or_create_collection(request.tenantId)
        document_id = request.hash  # Use hash as a unique ID

        try:
            collection.add(
                ids=[document_id],
                documents=[text],
                metadatas=[{"filePath": request.filePath, "hash": request.hash}],
                embeddings=embedding.tolist(),  # Ensure embeddings is 2D
            )
        except Exception as e:
            print("Error adding to ChromaDB:", e)
            raise

        # Verify stored data
        stored_data = collection.get()
        print("Stored Data After Addition:", stored_data)

        return {"message": "Embeddings generated successfully"}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to handle queries
class QueryRequest(BaseModel):
    tenantId: str
    query: str
    top_k: int = 5  # Number of results to return

@app.post("/query")
async def query_database(request: QueryRequest):
    try:
        # Generate embedding for the query
        query = request.query.lower().strip()
        query_embedding = generate_embedding(query)
        print("Query Embedding:", query_embedding)

        # Retrieve the collection for the tenant
        collection = chroma_client.get_or_create_collection(request.tenantId)

        # Perform similarity search in ChromaDB
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=request.top_k  # Use requested top_k
        )

        print("Similarity Results (Before Filtering):", results)

        # Filter results for tenantId, if needed (ChromaDB should handle tenant scoping natively)
        filtered_results = {
            "ids": results["ids"],
            "documents": results["documents"],
            "metadatas": results["metadatas"],
            "distances": results["distances"],
        }

        print("Similarity Results (Filtered):", filtered_results)

        if not filtered_results["documents"] or len(filtered_results["documents"][0]) == 0:
            return {
                "query": request.query,
                "results": {
                    "ids": [],
                    "documents": [],
                    "metadatas": [],
                    "distances": []
                },
                "message": "No relevant results found"
            }

        return {"query": request.query, "results": filtered_results}
    except Exception as e:
        print("Error querying database:", e)
        raise HTTPException(status_code=500, detail=str(e))
    
class DeleteEmbeddingRequest(BaseModel):
    tenantId: str
    hash: str
    
@app.post("/delete-embeddings")
async def delete_embeddings(request: DeleteEmbeddingRequest):
    try:
        # Retrieve the collection
        collection = chroma_client.get_or_create_collection(request.tenantId)

        # Remove the embeddings using the hash as the ID
        collection.delete(ids=[request.hash])

        return {"message": "Embeddings deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

