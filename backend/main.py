import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import shutil
import zipfile
from zipfile import ZipFile
from io import BytesIO
from fastapi.responses import FileResponse
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

UPLOAD_DIR = Path("uploaded_code")
UPLOAD_DIR.mkdir(exist_ok=True)
INDEX_PATH = Path("faiss_index")
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)
UPDATED_DIR = "updated_code"
FINAL_ZIP = "updated_codebase.zip"

class UserStory(BaseModel):
    story: str

# 🔧 IMPROVED: Enhanced code file loader with better error handling
def load_code_files(code_dir=UPLOAD_DIR):
    """Load and process code files with improved chunking strategy"""
    # Extended list of code file extensions
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
    
    # Walk through directory manually for better control
    for root, dirs, files in os.walk(code_dir):
        # Skip common non-code directories
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.next', 'dist', 'build']]
        
        for file in files:
            file_path = Path(root) / file
            relative_path = file_path.relative_to(code_dir)
            
            # Check if file matches any code extension
            if any(file_path.match(ext) for ext in code_extensions):
                try:
                    # Read file with proper encoding detection
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
                        # Create document with enhanced metadata
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

# 🔧 ENHANCED: More robust vectorstore creation with detailed logging
def create_vectorstore(code_dir=UPLOAD_DIR, index_path=INDEX_PATH):
    """Create FAISS vectorstore with comprehensive error handling and logging"""
    
    print(f"🚀 Starting vectorstore creation...")
    print(f"   📂 Source directory: {code_dir}")
    print(f"   📂 Index path: {index_path}")
    
    # Check OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")
    
    print(f"   🔑 OpenAI API key: {'✅ Set' if api_key else '❌ Missing'} (length: {len(api_key) if api_key else 0})")
    
    # Load all code files
    print("📚 Loading code files...")
    docs = load_code_files(code_dir)
    
    if not docs:
        raise ValueError(f"No valid source files found in {code_dir}. Check file permissions and formats.")
    
    print(f"📊 Loaded {len(docs)} documents successfully")
    
    # Log document details
    total_content = sum(len(doc.page_content) for doc in docs)
    file_types = {}
    for doc in docs:
        ext = doc.metadata.get('file_type', 'unknown')
        file_types[ext] = file_types.get(ext, 0) + 1
    
    print(f"   📄 Total content size: {total_content:,} characters")
    print(f"   📂 File types: {file_types}")
    
    # Create text splitter with detailed logging
    print("✂️  Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=[
            "\n\nclass ",  # Class definitions
            "\n\nfunction ",  # Function definitions  
            "\n\ndef ",  # Python functions
            "\n\n",  # Double newlines
            "\n",  # Single newlines
            " ",  # Spaces
            ""  # Characters
        ]
    )
    
    # Split documents with file path context
    split_docs = []
    for i, doc in enumerate(docs):
        try:
            chunks = splitter.split_documents([doc])
            print(f"   📄 {doc.metadata['source']}: {len(chunks)} chunks")
            
            for chunk in chunks:
                # Add file context to each chunk
                enhanced_content = f"FILE_PATH: {chunk.metadata['source']}\n\n{chunk.page_content}"
                chunk.page_content = enhanced_content
                split_docs.append(chunk)
                
        except Exception as e:
            print(f"   ❌ Failed to split {doc.metadata.get('source', f'doc_{i}')}: {e}")
            continue
    
    if not split_docs:
        raise ValueError("No chunks created from documents. Check document content and splitting configuration.")
    
    print(f"✅ Created {len(split_docs)} chunks from {len(docs)} files")
    
    # Create embeddings with error handling
    print("🧠 Creating embeddings...")
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=api_key
        )
        
        # Test embeddings with a small sample first
        print("   🧪 Testing embeddings with sample text...")
        test_embedding = embeddings.embed_query("test")
        print(f"   ✅ Embedding test successful (dimension: {len(test_embedding)})")
        
    except Exception as e:
        print(f"   ❌ Embedding initialization failed: {e}")
        raise ValueError(f"Failed to initialize OpenAI embeddings: {str(e)}")
    
    # Create FAISS vectorstore
    print("🔄 Building FAISS index...")
    try:
        # Process in smaller batches to avoid memory issues
        batch_size = 50
        vectordb = None
        
        for i in range(0, len(split_docs), batch_size):
            batch = split_docs[i:i+batch_size]
            print(f"   📦 Processing batch {i//batch_size + 1}/{(len(split_docs)-1)//batch_size + 1} ({len(batch)} chunks)")
            
            if vectordb is None:
                # Create initial vectorstore
                vectordb = FAISS.from_documents(batch, embeddings)
            else:
                # Add to existing vectorstore
                batch_db = FAISS.from_documents(batch, embeddings)
                vectordb.merge_from(batch_db)
        
        print("💾 Saving FAISS index...")
        
        # Ensure index directory exists
        index_path.mkdir(exist_ok=True, parents=True)
        
        # Save the index
        vectordb.save_local(str(index_path))
        
        # Verify saved files
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

# 🔧 UNIFIED: Single endpoint for code generation with RAG
@app.post("/generate-updated-zip/")
async def generate_updated_zip(user_story: UserStory):
    """Generate updated code using RAG and return the result"""
    
    if not INDEX_PATH.exists():
        raise HTTPException(status_code=400, detail="No codebase indexed yet. Please upload a ZIP file first.")
    
    try:
        print(f"🚀 Starting code generation for story: {user_story.story}")
        
        # Load the vectorstore
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        vectordb = FAISS.load_local(str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True)
        
        # 🔧 IMPROVED: Multi-stage retrieval for better context
        print("🔍 Performing semantic search...")
        
        # First, get initial relevant documents
        initial_docs = vectordb.similarity_search(user_story.story, k=15)
        
        # Extract file paths and get complete file contexts
        relevant_files = set()
        for doc in initial_docs:
            source = doc.metadata.get('source', '')
            if source:
                relevant_files.add(source)
        
        print(f"📁 Found {len(relevant_files)} relevant files: {list(relevant_files)}")
        
        # Build comprehensive context
        context_parts = []
        file_contents = {}
        
        # Get all chunks for each relevant file
        for file_path in relevant_files:
            file_chunks = vectordb.similarity_search(f"FILE_PATH: {file_path}", k=10)
            
            # Combine chunks for complete file context
            file_content_parts = []
            for chunk in file_chunks:
                if chunk.metadata.get('source') == file_path:
                    # Remove the FILE_PATH prefix for cleaner content
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
        
        # Create LLM for code generation
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
        
        # 🔧 ENHANCED: More detailed prompt for better code generation
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

# 🔧 ENHANCED: ZIP upload with comprehensive error handling and debugging
# Helper: parse LLM response into file updates
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
    try:
        if not UPLOAD_DIR.exists():
            raise HTTPException(status_code=400, detail="No uploaded codebase found")

        files = parse_updated_code(updated_code)
        if not files:
            raise HTTPException(status_code=400, detail="No files parsed from updated code")

        updated_dir = Path(UPDATED_DIR)
        if updated_dir.exists():
            shutil.rmtree(updated_dir)
        shutil.copytree(UPLOAD_DIR, updated_dir)

        # Write updates into updated_dir
        for filepath, content in files.items():
            file_path = updated_dir / filepath
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

        # Re-zip everything
        final_zip_path = Path(FINAL_ZIP)
        if final_zip_path.exists():
            final_zip_path.unlink()

        with ZipFile(final_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files_in_dir in os.walk(updated_dir):
                for file in files_in_dir:
                    abs_path = Path(root) / file
                    rel_path = abs_path.relative_to(updated_dir)
                    zipf.write(abs_path, rel_path)

        return {"message": "Updated ZIP created successfully", "download_url": "/download-zip"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save updated ZIP: {e}")

@app.get("/download-zip")
async def download_zip():
    """Download the latest updated ZIP"""
    final_zip_path = Path(FINAL_ZIP)
    if not final_zip_path.exists():
        raise HTTPException(status_code=404, detail="No updated ZIP found. Generate it first.")
    return FileResponse(final_zip_path, filename="updated_codebase.zip", media_type="application/zip")

@app.post("/upload-zip/")
async def upload_zip(file: UploadFile = File(...)):
    """Upload and index a ZIP file with detailed error reporting"""
    
    if not file.filename or not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files are allowed")
    
    print(f"📦 Processing ZIP file: {file.filename}")
    
    try:
        # Clear previous data
        print("🧹 Clearing previous data...")
        if UPLOAD_DIR.exists():
            shutil.rmtree(UPLOAD_DIR)
            print(f"   ✅ Removed old upload directory")
        if INDEX_PATH.exists():
            shutil.rmtree(INDEX_PATH)
            print(f"   ✅ Removed old index directory")
        
        UPLOAD_DIR.mkdir(exist_ok=True)
        print(f"   ✅ Created fresh upload directory: {UPLOAD_DIR}")
        
        # Extract ZIP contents
        content = await file.read()
        print(f"📁 Extracting ZIP ({len(content)} bytes)...")
        
        extracted_count = 0
        with ZipFile(BytesIO(content)) as zip_file:
            # List all files in ZIP
            file_list = zip_file.namelist()
            print(f"📋 ZIP contains {len(file_list)} entries")
            
            # Extract all files with progress
            for item in file_list:
                try:
                    zip_file.extract(item, UPLOAD_DIR)
                    if not item.endswith('/'):  # Don't count directories
                        extracted_count += 1
                except Exception as e:
                    print(f"   ⚠️  Failed to extract {item}: {e}")
        
        # Verify extraction with detailed file analysis
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
        
        print(f"✅ Successfully extracted {len(extracted_files)} files to {UPLOAD_DIR}")
        
        # Log file types found
        extensions = {}
        for f in extracted_files:
            ext = f.get("extension", "no_extension")
            extensions[ext] = extensions.get(ext, 0) + 1
        print(f"📊 File types found: {extensions}")
        
        # Create vector index with detailed error handling
        print("🔄 Creating FAISS index...")
        try:
            vectordb = create_vectorstore(code_dir=UPLOAD_DIR, index_path=INDEX_PATH)
            
            # Verify index was created
            index_created = INDEX_PATH.exists()
            index_files = list(INDEX_PATH.glob("*")) if index_created else []
            
            print(f"✅ FAISS index creation completed")
            print(f"   📂 Index directory exists: {index_created}")
            print(f"   📄 Index files created: {[f.name for f in index_files]}")
            
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
            import traceback
            print(f"   📋 Full traceback: {traceback.format_exc()}")
            
            # Return partial success - files extracted but not indexed
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
        import traceback
        print(f"   📋 Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")

# 🔧 ENHANCED: Debug endpoint with comprehensive diagnostics
@app.get("/list-indexed-files/")
async def list_indexed_files():
    """Debug endpoint with detailed diagnostics"""
    
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
    
    # Check actual uploaded files
    actual_files = []
    if UPLOAD_DIR.exists():
        print(f"🔍 Scanning upload directory: {UPLOAD_DIR}")
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
    
    # Check if index exists and try to load it
    if not INDEX_PATH.exists():
        debug_info["index_status"] = "Index directory does not exist"
        return debug_info
    
    # Check index directory contents
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
        # Try to load the FAISS index
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        print(f"🔄 Attempting to load FAISS index from {INDEX_PATH}")
        vectordb = FAISS.load_local(str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True)
        
        # Sample the vector database
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
            
            # Get a preview if we don't have one yet
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
        print(f"❌ Failed to load FAISS index: {str(e)}")
        return debug_info

# 🔧 NEW: Test endpoint for diagnosing indexing issues
@app.post("/test-indexing/")
async def test_indexing():
    """Test the indexing process with a simple file"""
    
    try:
        # Create a test directory and file
        test_dir = Path("test_indexing")
        if test_dir.exists():
            shutil.rmtree(test_dir)
        test_dir.mkdir()
        
        # Create a simple test file
        test_file = test_dir / "test.js"
        test_content = """
// Test JavaScript file
function hello(name) {
    return `Hello, ${name}!`;
}

class TestClass {
    constructor(value) {
        this.value = value;
    }
    
    getValue() {
        return this.value;
    }
}

export { hello, TestClass };
"""
        test_file.write_text(test_content)
        
        print(f"🧪 Created test file: {test_file}")
        
        # Test the indexing process
        test_index_path = Path("test_faiss_index")
        if test_index_path.exists():
            shutil.rmtree(test_index_path)
            
        vectordb = create_vectorstore(code_dir=test_dir, index_path=test_index_path)
        
        # Test retrieval
        test_query = "javascript function"
        results = vectordb.similarity_search(test_query, k=3)
        
        # Clean up
        shutil.rmtree(test_dir)
        shutil.rmtree(test_index_path)
        
        return {
            "status": "success",
            "message": "Indexing test completed successfully",
            "test_results": {
                "chunks_created": len(results),
                "sample_content": results[0].page_content[:200] if results else "No content found"
            }
        }
        
    except Exception as e:
        import traceback
        return {
            "status": "failed", 
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "upload_dir_exists": UPLOAD_DIR.exists(),
        "index_exists": INDEX_PATH.exists(),
        "openai_key_set": bool(os.getenv("OPENAI_API_KEY"))
    }

# Clear all data endpoint for debugging
@app.post("/clear-all-data/")
async def clear_all_data():
    """Clear all uploaded files and indexes for a fresh start"""
    try:
        if UPLOAD_DIR.exists():
            shutil.rmtree(UPLOAD_DIR)
        if INDEX_PATH.exists():
            shutil.rmtree(INDEX_PATH)
        if os.path.exists(UPDATED_DIR):
            shutil.rmtree(UPDATED_DIR)
        if os.path.exists(FINAL_ZIP):
            os.remove(FINAL_ZIP)
        
        # Recreate directories
        UPLOAD_DIR.mkdir(exist_ok=True)
        
        return {"message": "All data cleared successfully. Ready for fresh upload."}
    except Exception as e:
        return {"error": f"Failed to clear data: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)