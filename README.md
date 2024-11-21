
# Python Backend for RAG Application

This Python backend processes documents, generates embeddings using the LaBSE model, and integrates with ChromaDB for vector similarity search. The application is built using FastAPI.

---

## **Prerequisites**
Before you start, ensure you have the following installed on your system:

- Python 3.9 or higher
- pip (Python package manager)
- Virtual environment (optional but recommended)
- Git (for cloning the repository)
- Access to the internet for downloading Python packages and LaBSE model

---

## **Setup Instructions**

### 1. **Clone the Repository**
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. **Create and Activate a Virtual Environment**
1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```
2. Activate the virtual environment:
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```

### 3. **Install Dependencies**
Use the provided `requirements.txt` file to install all necessary dependencies:
```bash
pip install -r requirements.txt
```

### 4. **Verify Installation**
Check that the required libraries are installed:
```bash
pip list
```
Ensure the following are included:
- `fastapi`
- `uvicorn`
- `transformers`
- `torch`
- `chromadb`
- Other dependencies listed in `requirements.txt`

---

## **Run the Application**

### 1. **Start the FastAPI Server**
Run the following command to start the server:
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 2. **Check the API Documentation**
Open your browser and visit:
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Redoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 3. **Test the Application**
Use a tool like `curl`, Postman, or Swagger UI to interact with the API. For example, to test the `/generate-embeddings` endpoint:
```bash
curl -X POST "http://127.0.0.1:8000/generate-embeddings" -H "Content-Type: application/json" -d '{"tenantId": "123", "filePath": "/path/to/file.txt", "hash": "abc123"}'
```

---

## **Troubleshooting**

### Common Issues and Solutions

1. **`ModuleNotFoundError` for `transformers` or other libraries**:
   - Ensure you installed dependencies in the correct Python environment (e.g., `venv`):
     ```bash
     pip install -r requirements.txt
     ```

2. **Uvicorn Using the Wrong Python Interpreter**:
   - Ensure the virtual environment is activated before running `uvicorn`:
     ```bash
     source venv/bin/activate
     python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
     ```

3. **CUDA or GPU Compatibility Issues**:
   - If using GPU, ensure you’ve installed the correct version of PyTorch for your CUDA setup. Otherwise, install the CPU-only version:
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
     ```

---

### Notes

- Replace `<repository-url>` with the actual URL of the repository when cloning.
- Make sure `app.py` is in the root directory of the project.

---

Let me know if you encounter any issues or need further assistance!
