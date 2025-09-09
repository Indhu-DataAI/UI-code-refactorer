# import os
# import shutil
# from pathlib import Path
# from zipfile import ZipFile
# from io import BytesIO
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.responses import FileResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List
# import re

# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import DirectoryLoader, TextLoader
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from dotenv import load_dotenv

# load_dotenv()

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000","http://127.0.0.1:3000","http://localhost:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# UPLOAD_DIR = Path("uploaded_code")
# UPLOAD_DIR.mkdir(exist_ok=True)
# INDEX_PATH = Path("faiss_index")
# TEMP_DIR = Path("temp_updated")
# TEMP_DIR.mkdir(exist_ok=True)

# class UserStory(BaseModel):
#     story: str

# # --- FAISS vectorstore creation ---
# def create_vectorstore(code_dir=UPLOAD_DIR, index_path=INDEX_PATH):
#     exts = [".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rb", ".php", ".html", ".css", ".json"]
#     docs = []

#     for file_path in code_dir.rglob("*"):
#         if file_path.suffix.lower() in exts and file_path.is_file():
#             try:
#                 text = file_path.read_text(encoding="utf-8", errors="ignore")
#                 docs.append(Document(
#                     page_content=f"[FILEPATH: {file_path}]\n{text}",
#                     metadata={"source": str(file_path)}
#                 ))
#             except Exception as e:
#                 print(f"⚠️ Could not read {file_path}: {e}")

#     if not docs:
#         raise ValueError("No valid source files found in uploaded codebase.")

#     splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
#     split_docs = splitter.split_documents(docs)

#     embeddings = OpenAIEmbeddings(
#         model="text-embedding-3-large",
#         api_key=os.getenv("OPENAI_API_KEY")
#     )
#     vectordb = FAISS.from_documents(split_docs, embeddings)
#     vectordb.save_local(index_path)
#     return vectordb


# # --- Upload ZIP and extract ---
# @app.post("/upload-zip/")
# async def upload_zip(file: UploadFile = File(...)):
#     if not file.filename.endswith(".zip"):
#         raise HTTPException(status_code=400, detail="Only ZIP files are allowed")

#     # Clear previous uploads
#     if UPLOAD_DIR.exists():
#         shutil.rmtree(UPLOAD_DIR)
#     UPLOAD_DIR.mkdir(exist_ok=True)

#     content = await file.read()
#     with ZipFile(BytesIO(content)) as zip_file:
#         zip_file.extractall(UPLOAD_DIR)

#     try:
#         create_vectorstore(code_dir=UPLOAD_DIR, index_path=INDEX_PATH)
#         return {"message": "ZIP uploaded, extracted, and indexed successfully."}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

# # --- Generate updated code and return ZIP ---
# @app.post("/generate-zip/")
# async def generate_zip(story: UserStory):
#     if not INDEX_PATH.exists():
#         raise HTTPException(status_code=400, detail="No code uploaded/indexed yet.")

#     try:
#         embeddings = OpenAIEmbeddings(
#             model="text-embedding-3-large",
#             api_key=os.getenv("OPENAI_API_KEY")
#         )
#         vectordb = FAISS.load_local(str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True)

#         # Retrieve relevant code files
#         docs = vectordb.similarity_search(story.story, k=5)
#         context = ""
#         for doc in docs:
#             source = doc.metadata.get('source', 'unknown')
#             content = doc.page_content
#             if '[FILEPATH:' in content:
#                 content = content.split('\n', 1)[1] if '\n' in content else content
#             context += f"=== FILE: {source} ===\n{content}\n\n"

#         # Prompt LLM
#         llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
#         prompt = f"""You are a senior software engineer. Analyze the codebase and implement the requested changes.

# USER STORY: {story.story}

# CODEBASE CONTEXT:
# {context}

# INSTRUCTIONS:
# 1. Identify files to modify
# 2. Apply requested changes
# 3. Return COMPLETE updated files in this format:

# === FILE: path/to/filename.ext ===
# [COMPLETE FILE CONTENT WITH CHANGES]
# === END FILE ===

# Include all imports and existing code. No explanations, only updated code."""

#         response = llm.invoke(prompt)
#         updated_code = response.content

#         # Save updated files to TEMP_DIR
#         if TEMP_DIR.exists():
#             shutil.rmtree(TEMP_DIR)
#         TEMP_DIR.mkdir(exist_ok=True)

#         matches = re.findall(r"=== FILE: (.+?) ===\n(.*?)\n=== END FILE ===", updated_code, re.DOTALL)
#         for filepath, content in matches:
#             save_path = TEMP_DIR / filepath
#             save_path.parent.mkdir(parents=True, exist_ok=True)
#             save_path.write_text(content)

#         # Create ZIP
#         zip_path = TEMP_DIR / "updated_code.zip"
#         shutil.make_archive(str(zip_path.with_suffix('')), 'zip', TEMP_DIR)

#         return FileResponse(path=zip_path, filename="updated_code.zip", media_type="application/zip")

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to generate updated ZIP: {str(e)}")

# # --- Health check ---
# @app.get("/health")
# async def health_check():
#     return {
#         "status": "healthy",
#         "upload_dir_exists": UPLOAD_DIR.exists(),
#         "index_exists": INDEX_PATH.exists(),
#         "openai_key_set": bool(os.getenv("OPENAI_API_KEY"))
#     }
