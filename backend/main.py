import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import shutil
import zipfile
from zipfile import ZipFile
from io import BytesIO
from langchain_community.chat_models import ChatOllama

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool, AgentType
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file configuration - FIXED: Don't mount at root path
dist_path = Path("../dist")
if not dist_path.exists():
    dist_path = Path("./dist")
if not dist_path.exists():
    dist_path = Path("../../dist")

# IMPORTANT: Only mount static files if dist exists AND at a specific path, not root
if dist_path.exists():
    # Mount assets and other static files at specific paths
    try:
        app.mount("/static", StaticFiles(directory=str(dist_path)), name="static")
        print(f"[SUCCESS] Mounted static files from: {dist_path.absolute()}")
    except Exception as e:
        print(f"[WARNING] Failed to mount static files: {e}")
else:
    print(f"[WARNING] Frontend dist folder not found")

UPLOAD_DIR = Path("uploaded_code")
UPLOAD_DIR.mkdir(exist_ok=True)
INDEX_PATH = Path("faiss_index")
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)
UPDATED_DIR = "updated_code"
FINAL_ZIP = "updated_codebase.zip"

class UserStory(BaseModel):
    story: str

def load_code_files(code_dir=UPLOAD_DIR):
    """Load and process code files with improved chunking strategy"""
    code_extensions = [
        "*.py", "*.js", "*.jsx", "*.ts", "*.tsx", "*.java", "*.go", "*.rb", "*.php",
        "*.html", "*.css", "*.scss", "*.sass", "*.json", "*.xml", "*.yaml", "*.yml",
        "*.cpp", "*.c", "*.h", "*.cs", "*.swift", "*.kt", "*.rs", "*.vue", "*.svelte",
        "*.md", "*.txt", "*.env", "*.config", "*.conf", "*.ini", "*.toml"
    ]
    
    docs = []
    files_processed = 0
    files_failed = 0
    
    print(f"🔍 Scanning directory: {code_dir}")
    
    for root, dirs, files in os.walk(code_dir):
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.next', 'dist', 'build']]
        
        for file in files:
            file_path = Path(root) / file
            relative_path = file_path.relative_to(code_dir)
            
            if any(file_path.match(ext) for ext in code_extensions):
                try:
                    content = ""
                    encodings = ['utf-8', 'latin-1', 'cp1252']
                    
                    for encoding in encodings:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                content = f.read()
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    if content:
                        doc = Document(
                            page_content=content,
                            metadata={
                                'source': str(relative_path),
                                'full_path': str(file_path),
                                'file_type': file_path.suffix,
                                'file_size': len(content)
                            }
                        )
                        docs.append(doc)
                        files_processed += 1
                        print(f"✅ Processed: {relative_path}")
                    else:
                        print(f"⚠️  Empty or unreadable: {relative_path}")
                        files_failed += 1
                        
                except Exception as e:
                    print(f"❌ Failed to process {relative_path}: {str(e)}")
                    files_failed += 1
    
    print(f"📊 Processing complete: {files_processed} files processed, {files_failed} failed")
    return docs

def create_vectorstore(code_dir=UPLOAD_DIR, index_path=INDEX_PATH):
    """Create FAISS vectorstore with comprehensive error handling and logging"""
    
    print(f"🚀 Starting vectorstore creation...")
    print(f"   📂 Source directory: {code_dir}")
    print(f"   📂 Index path: {index_path}")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")
    
    print(f"   🔑 OpenAI API key: {'✅ Set' if api_key else '❌ Missing'} (length: {len(api_key) if api_key else 0})")
    
    print("📚 Loading code files...")
    docs = load_code_files(code_dir)
    
    if not docs:
        raise ValueError(f"No valid source files found in {code_dir}. Check file permissions and formats.")
    
    print(f"📊 Loaded {len(docs)} documents successfully")
    
    total_content = sum(len(doc.page_content) for doc in docs)
    file_types = {}
    for doc in docs:
        ext = doc.metadata.get('file_type', 'unknown')
        file_types[ext] = file_types.get(ext, 0) + 1
    
    print(f"   📄 Total content size: {total_content:,} characters")
    print(f"   📂 File types: {file_types}")
    
    print("✂️  Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=[
            "\n\nclass ",
            "\n\nfunction ",
            "\n\ndef ",
            "\n\n",
            "\n",
            " ",
            ""
        ]
    )
    
    split_docs = []
    for i, doc in enumerate(docs):
        try:
            chunks = splitter.split_documents([doc])
            print(f"   📄 {doc.metadata['source']}: {len(chunks)} chunks")
            
            for chunk in chunks:
                enhanced_content = f"FILE_PATH: {chunk.metadata['source']}\n\n{chunk.page_content}"
                chunk.page_content = enhanced_content
                split_docs.append(chunk)
                
        except Exception as e:
            print(f"   ❌ Failed to split {doc.metadata.get('source', f'doc_{i}')}: {e}")
            continue
    
    if not split_docs:
        raise ValueError("No chunks created from documents. Check document content and splitting configuration.")
    
    print(f"✅ Created {len(split_docs)} chunks from {len(docs)} files")
    
    print("🧠 Creating embeddings...")
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=api_key
        )
        
        print("   🧪 Testing embeddings with sample text...")
        test_embedding = embeddings.embed_query("test")
        print(f"   ✅ Embedding test successful (dimension: {len(test_embedding)})")
        
    except Exception as e:
        print(f"   ❌ Embedding initialization failed: {e}")
        raise ValueError(f"Failed to initialize OpenAI embeddings: {str(e)}")
    
    print("🔄 Building FAISS index...")
    try:
        batch_size = 50
        vectordb = None
        
        for i in range(0, len(split_docs), batch_size):
            batch = split_docs[i:i+batch_size]
            print(f"   📦 Processing batch {i//batch_size + 1}/{(len(split_docs)-1)//batch_size + 1} ({len(batch)} chunks)")
            
            if vectordb is None:
                vectordb = FAISS.from_documents(batch, embeddings)
            else:
                batch_db = FAISS.from_documents(batch, embeddings)
                vectordb.merge_from(batch_db)
        
        print("💾 Saving FAISS index...")
        
        index_path.mkdir(exist_ok=True, parents=True)
        vectordb.save_local(str(index_path))
        
        saved_files = list(index_path.glob("*"))
        print(f"   📁 Saved index files: {[f.name for f in saved_files]}")
        
        if not saved_files:
            raise ValueError("No index files were saved")
        
        print(f"✅ FAISS index created successfully at {index_path}")
        return vectordb
        
    except Exception as e:
        print(f"❌ FAISS vectorstore creation failed: {str(e)}")
        import traceback
        print(f"   📋 Full traceback: {traceback.format_exc()}")
        raise ValueError(f"Failed to create FAISS vectorstore: {str(e)}")

# API ENDPOINTS - FIXED: Properly organized and accessible

@app.post("/upload-zip/")
async def upload_zip(file: UploadFile = File(...)):
    """Upload and index a ZIP file with detailed error reporting"""
    print(f"📦 API CALL: /upload-zip/ - Processing ZIP file: {file.filename}")
    
    if not file.filename or not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files are allowed")
    
    try:
        print("🧹 Clearing previous data...")
        if UPLOAD_DIR.exists():
            shutil.rmtree(UPLOAD_DIR)
        if INDEX_PATH.exists():
            shutil.rmtree(INDEX_PATH)
        
        UPLOAD_DIR.mkdir(exist_ok=True)
        
        content = await file.read()
        print(f"📁 Extracting ZIP ({len(content)} bytes)...")
        
        extracted_count = 0
        with ZipFile(BytesIO(content)) as zip_file:
            file_list = zip_file.namelist()
            print(f"📋 ZIP contains {len(file_list)} entries")
            
            for item in file_list:
                try:
                    zip_file.extract(item, UPLOAD_DIR)
                    if not item.endswith('/'):
                        extracted_count += 1
                except Exception as e:
                    print(f"   ⚠️  Failed to extract {item}: {e}")
        
        extracted_files = []
        if UPLOAD_DIR.exists():
            for root, dirs, files in os.walk(UPLOAD_DIR):
                for file_name in files:
                    file_path = Path(root) / file_name
                    rel_path = file_path.relative_to(UPLOAD_DIR)
                    try:
                        size = file_path.stat().st_size
                        extracted_files.append({
                            "path": str(rel_path),
                            "size": size,
                            "extension": file_path.suffix
                        })
                    except Exception as e:
                        print(f"   ⚠️  Could not stat {rel_path}: {e}")
        
        print(f"✅ Successfully extracted {len(extracted_files)} files")
        
        extensions = {}
        for f in extracted_files:
            ext = f.get("extension", "no_extension")
            extensions[ext] = extensions.get(ext, 0) + 1
        print(f"📊 File types found: {extensions}")
        
        print("🔄 Creating FAISS index...")
        try:
            vectordb = create_vectorstore(code_dir=UPLOAD_DIR, index_path=INDEX_PATH)
            
            index_created = INDEX_PATH.exists()
            index_files = list(INDEX_PATH.glob("*")) if index_created else []
            
            print(f"✅ FAISS index creation completed")
            
            return {
                "message": "ZIP uploaded, extracted, and indexed successfully",
                "files_extracted": len(extracted_files),
                "file_types": extensions,
                "index_created": index_created,
                "index_files": [f.name for f in index_files],
                "upload_dir": str(UPLOAD_DIR.absolute()),
                "index_dir": str(INDEX_PATH.absolute())
            }
            
        except Exception as index_error:
            print(f"❌ Index creation failed: {str(index_error)}")
            
            return {
                "message": "ZIP extracted but indexing failed",
                "files_extracted": len(extracted_files),
                "file_types": extensions,
                "index_created": False,
                "index_error": str(index_error),
                "upload_dir": str(UPLOAD_DIR.absolute())
            }
        
    except Exception as e:
        print(f"❌ Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")

@app.post("/generate-updated-zip/")
async def generate_updated_zip(user_story: UserStory):
    """Generate updated code using RAG and return the result"""
    print(f"🚀 API CALL: /generate-updated-zip/ - Story: {user_story.story[:100]}...")
    
    if not INDEX_PATH.exists():
        raise HTTPException(status_code=400, detail="No codebase indexed yet. Please upload a ZIP file first.")
    
    try:
        print(f"🚀 Starting code generation...")
        
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        vectordb = FAISS.load_local(str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True)
        
        print("🔍 Performing semantic search...")
        initial_docs = vectordb.similarity_search(user_story.story, k=15)
        
        relevant_files = set()
        for doc in initial_docs:
            source = doc.metadata.get('source', '')
            if source:
                relevant_files.add(source)
        
        print(f"📁 Found {len(relevant_files)} relevant files: {list(relevant_files)}")
        
        context_parts = []
        file_contents = {}
        
        for file_path in relevant_files:
            file_chunks = vectordb.similarity_search(f"FILE_PATH: {file_path}", k=10)
            
            file_content_parts = []
            for chunk in file_chunks:
                if chunk.metadata.get('source') == file_path:
                    content = chunk.page_content
                    if content.startswith("FILE_PATH:"):
                        content = content.split("\n\n", 1)[1] if "\n\n" in content else content
                    file_content_parts.append(content)
            
            if file_content_parts:
                combined_content = "\n".join(file_content_parts)
                file_contents[file_path] = combined_content
                context_parts.append(f"=== FILE: {file_path} ===\n{combined_content}\n")
        
        full_context = "\n".join(context_parts)
        print(f"📝 Built context with {len(file_contents)} files, {len(full_context)} characters")
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        
        enhanced_prompt = f"""You are an expert software engineer tasked with modifying an existing codebase. 

MODIFICATION REQUEST:
{user_story.story}

CURRENT CODEBASE CONTEXT:
{full_context}

INSTRUCTIONS:
1. Carefully analyze the user's request and understand what changes are needed
2. Identify all files that need to be modified to fulfill the request
3. Make ONLY the necessary changes while preserving existing functionality
4. Return complete modified files using the EXACT format below
5. Include ALL imports, functions, classes, and existing code in modified files
6. Ensure all syntax is correct and follows the original code style

OUTPUT FORMAT (use exactly this format):
=== FILE: path/to/filename.ext ===
[COMPLETE FILE CONTENT WITH MODIFICATIONS]
=== END FILE ===

=== FILE: another/file.ext ===
[COMPLETE FILE CONTENT WITH MODIFICATIONS]
=== END FILE ===

Requirements:
- Only include files that need modifications
- Each file must be complete and functional
- Maintain existing imports and dependencies
- Follow the original code structure and style
- Make minimal changes to achieve the requested functionality

Begin generating the modified files now:"""

        print("🤖 Generating code with LLM...")
        response = llm.invoke(enhanced_prompt)
        updated_code = response.content
        
        print(f"✅ Code generation complete: {len(updated_code)} characters generated")
        
        return {"updated_code": updated_code}
        
    except Exception as e:
        print(f"❌ Code generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")

def parse_updated_code(updated_code: str):
    files = {}
    current_file = None
    content = []
    
    for line in updated_code.splitlines():
        if line.startswith("=== FILE:"):
            if current_file and content:
                files[current_file] = "\n".join(content).strip()
            current_file = line.replace("=== FILE:", "").replace("===", "").strip()
            content = []
        elif line.startswith("=== END FILE ==="):
            if current_file and content:
                files[current_file] = "\n".join(content).strip()
            current_file, content = None, []
        else:
            if current_file:
                content.append(line)
    return files

@app.post("/save-updated-zip/")
async def save_updated_zip(updated_code: str = Form(...)):
    """Save AI-updated code into files and repackage as ZIP"""
    print(f"🚀 API CALL: /save-updated-zip/ - Code length: {len(updated_code)}")
    
    try:
        if not UPLOAD_DIR.exists():
            raise HTTPException(status_code=400, detail="No uploaded codebase found")

        files = parse_updated_code(updated_code)
        if not files:
            raise HTTPException(status_code=400, detail="No files parsed from updated code")

        print(f"📁 Parsed {len(files)} files: {list(files.keys())}")

        updated_dir = Path(UPDATED_DIR)
        if updated_dir.exists():
            shutil.rmtree(updated_dir)
        shutil.copytree(UPLOAD_DIR, updated_dir)

        for filepath, content in files.items():
            file_path = updated_dir / filepath
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            print(f"✅ Updated: {filepath}")

        final_zip_path = Path(FINAL_ZIP)
        if final_zip_path.exists():
            final_zip_path.unlink()

        with ZipFile(final_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files_in_dir in os.walk(updated_dir):
                for file in files_in_dir:
                    abs_path = Path(root) / file
                    rel_path = abs_path.relative_to(updated_dir)
                    zipf.write(abs_path, rel_path)

        print(f"✅ Created ZIP with {len(files)} updated files")
        return {"message": "Updated ZIP created successfully", "download_url": "/download-zip"}

    except Exception as e:
        print(f"❌ Save failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save updated ZIP: {e}")

@app.get("/download-zip")
async def download_zip():
    """Download the latest updated ZIP"""
    print("🚀 API CALL: /download-zip")
    
    final_zip_path = Path(FINAL_ZIP)
    if not final_zip_path.exists():
        raise HTTPException(status_code=404, detail="No updated ZIP found. Generate it first.")
    
    print(f"✅ Serving ZIP file: {final_zip_path} ({final_zip_path.stat().st_size} bytes)")
    return FileResponse(final_zip_path, filename="updated_codebase.zip", media_type="application/zip")

@app.get("/list-indexed-files/")
async def list_indexed_files():
    """Debug endpoint with detailed diagnostics"""
    print("🚀 API CALL: /list-indexed-files/")
    
    debug_info = {
        "directories": {
            "upload_dir_exists": UPLOAD_DIR.exists(),
            "upload_dir_path": str(UPLOAD_DIR.absolute()),
            "index_dir_exists": INDEX_PATH.exists(),
            "index_dir_path": str(INDEX_PATH.absolute())
        },
        "environment": {
            "openai_key_set": bool(os.getenv("OPENAI_API_KEY")),
            "openai_key_length": len(os.getenv("OPENAI_API_KEY", "")) if os.getenv("OPENAI_API_KEY") else 0
        }
    }
    
    actual_files = []
    if UPLOAD_DIR.exists():
        for root, dirs, files in os.walk(UPLOAD_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, UPLOAD_DIR)
                try:
                    file_size = os.path.getsize(file_path)
                    actual_files.append({
                        "path": rel_path,
                        "size": file_size,
                        "extension": Path(file).suffix
                    })
                except Exception as e:
                    actual_files.append({
                        "path": rel_path,
                        "error": str(e)
                    })
        
        debug_info["actual_files"] = actual_files
        debug_info["total_actual_files"] = len(actual_files)
    else:
        debug_info["actual_files"] = []
        debug_info["upload_dir_error"] = "Upload directory does not exist"
    
    if not INDEX_PATH.exists():
        debug_info["index_status"] = "Index directory does not exist"
        return debug_info
    
    index_files = []
    if INDEX_PATH.exists():
        for item in INDEX_PATH.iterdir():
            index_files.append({
                "name": item.name,
                "is_file": item.is_file(),
                "size": item.stat().st_size if item.is_file() else None
            })
    
    debug_info["index_files"] = index_files
    
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        vectordb = FAISS.load_local(str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True)
        sample_docs = vectordb.similarity_search("", k=50)
        
        indexed_files = {}
        for doc in sample_docs:
            source = doc.metadata.get("source", "unknown")
            if source not in indexed_files:
                indexed_files[source] = {
                    "chunks": 0,
                    "total_content_length": 0,
                    "preview": ""
                }
            
            indexed_files[source]["chunks"] += 1
            indexed_files[source]["total_content_length"] += len(doc.page_content)
            
            if not indexed_files[source]["preview"]:
                content = doc.page_content
                if content.startswith("FILE_PATH:"):
                    content = content.split("\n\n", 1)[1] if "\n\n" in content else content
                preview = content[:200] + "..." if len(content) > 200 else content
                indexed_files[source]["preview"] = preview
        
        debug_info.update({
            "status": "success",
            "indexed_files": indexed_files,
            "total_indexed_files": len(indexed_files),
            "total_chunks": sum(info["chunks"] for info in indexed_files.values()),
            "sample_chunk_count": len(sample_docs)
        })
        
        return debug_info
        
    except Exception as e:
        debug_info["index_load_error"] = str(e)
        debug_info["index_status"] = "Index exists but failed to load"
        return debug_info

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    print("🚀 API CALL: /health")
    return {
        "status": "healthy",
        "upload_dir_exists": UPLOAD_DIR.exists(),
        "index_exists": INDEX_PATH.exists(),
        "openai_key_set": bool(os.getenv("OPENAI_API_KEY"))
    }

@app.post("/clear-all-data/")
async def clear_all_data():
    """Clear all uploaded files and indexes for a fresh start"""
    print("🚀 API CALL: /clear-all-data/")
    try:
        if UPLOAD_DIR.exists():
            shutil.rmtree(UPLOAD_DIR)
        if INDEX_PATH.exists():
            shutil.rmtree(INDEX_PATH)
        if os.path.exists(UPDATED_DIR):
            shutil.rmtree(UPDATED_DIR)
        if os.path.exists(FINAL_ZIP):
            os.remove(FINAL_ZIP)
        
        UPLOAD_DIR.mkdir(exist_ok=True)
        
        print("✅ All data cleared successfully")
        return {"message": "All data cleared successfully. Ready for fresh upload."}
    except Exception as e:
        print(f"❌ Clear failed: {str(e)}")
        return {"error": f"Failed to clear data: {str(e)}"}

# FRONTEND SERVING - FIXED: Only serve frontend for non-API routes
@app.get("/")
async def serve_frontend():
    """Serve the React frontend"""
    possible_paths = [
        Path("../dist/index.html"),
        Path("./dist/index.html"), 
        Path("../../dist/index.html")
    ]
    
    for index_path in possible_paths:
        if index_path.exists():
            print(f"[SUCCESS] Serving frontend from: {index_path.absolute()}")
            return FileResponse(str(index_path))
    
    print("[WARNING] Frontend index.html not found")
    return {
        "message": "API is running but frontend not found", 
        "checked_paths": [str(p) for p in possible_paths],
        "current_dir": str(Path.cwd()),
        "api_endpoints": [
            "/upload-zip/",
            "/generate-updated-zip/", 
            "/save-updated-zip/",
            "/download-zip",
            "/list-indexed-files/",
            "/health",
            "/clear-all-data/"
        ]
    }

# FIXED: Catch-all route that properly handles API vs frontend routing
@app.get("/{full_path:path}")
async def serve_frontend_routes(full_path: str):
    """Handle React Router routes and static assets - FIXED"""
    
    # CRITICAL: Don't interfere with API routes - check for API prefixes
    api_routes = [
        "upload-zip", "generate-updated-zip", "save-updated-zip", 
        "download-zip", "list-indexed-files", "health", "clear-all-data",
        "docs", "openapi.json", "redoc"
    ]
    
    # If it's an API route, return 404 (let FastAPI handle it properly)
    if any(full_path.startswith(route) for route in api_routes):
        print(f"[API] Attempted to access API route via catch-all: /{full_path}")
        raise HTTPException(status_code=404, detail="API endpoint not found")
    
    # Check for static assets first
    possible_dist_paths = [
        Path("../dist"),
        Path("./dist"), 
        Path("../../dist")
    ]
    
    for dist_path in possible_dist_paths:
        if dist_path.exists():
            asset_path = dist_path / full_path
            if asset_path.exists() and asset_path.is_file():
                print(f"[STATIC] Serving static asset: {asset_path}")
                return FileResponse(str(asset_path))
            break
    
    # Fall back to serving index.html for React Router
    for dist_path in possible_dist_paths:
        index_path = dist_path / "index.html"
        if index_path.exists():
            print(f"[FRONTEND] Serving React app for route: /{full_path}")
            return FileResponse(str(index_path))
    
    print(f"[WARNING] No frontend found for route: /{full_path}")
    raise HTTPException(status_code=404, detail="Frontend not found")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting server on port {port}")
    print("📍 API endpoints available at:")
    print("   POST /upload-zip/")
    print("   POST /generate-updated-zip/") 
    print("   POST /save-updated-zip/")
    print("   GET  /download-zip")
    print("   GET  /list-indexed-files/")
    print("   GET  /health")
    print("   POST /clear-all-data/")
    uvicorn.run(app, host="0.0.0.0", port=port)