import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import json

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.schema import Document
from langchain.callbacks import StreamingStdOutCallbackHandler

class CodebaseRAG:
    """Simple RAG framework for codebase using LangChain"""
    
    def __init__(self, openai_api_key: str, faiss_index_path: str = "./faiss_index"):
        """Initialize the RAG system"""
        self.openai_api_key = openai_api_key
        self.faiss_index_path = faiss_index_path
        
        # Initialize OpenAI components
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-4o-mini",
            temperature=0.1,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        # Text splitters for different languages
        self.text_splitters = {
            'python': RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=2000,
                chunk_overlap=200
            ),
            'javascript': RecursiveCharacterTextSplitter.from_language(
                language=Language.JS,
                chunk_size=2000,
                chunk_overlap=200
            ),
            'html': RecursiveCharacterTextSplitter.from_language(
                language=Language.HTML,
                chunk_size=2000,
                chunk_overlap=200
            ),
            'default': RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
        }
        
        self.vectorstore = None
        self.retrieval_chain = None
        
    def load_and_chunk_codebase(self, codebase_path: str):
        """Load and chunk the entire codebase"""
        print(f"Loading codebase from {codebase_path}...")
        
        # File extensions and their corresponding languages
        file_extensions = {
            '*.py': 'python',
            '*.js': 'javascript', 
            '*.jsx': 'javascript',
            '*.ts': 'javascript',
            '*.tsx': 'javascript',
            '*.html': 'html',
            '*.css': 'default',
            '*.scss': 'default',
            '*.vue': 'javascript',
            '*.md': 'default'
        }
        
        all_documents = []
        
        # Load files by extension
        for pattern, language in file_extensions.items():
            try:
                loader = DirectoryLoader(
                    codebase_path,
                    glob=f"**/{pattern}",
                    loader_cls=TextLoader,
                    loader_kwargs={'encoding': 'utf-8'}
                )
                documents = loader.load()
                
                if documents:
                    print(f"Found {len(documents)} {pattern} files")
                    
                    # Add language metadata
                    for doc in documents:
                        doc.metadata['language'] = language
                        doc.metadata['file_type'] = pattern.replace('*', '').replace('.', '')
                    
                    # Split documents
                    splitter = self.text_splitters.get(language, self.text_splitters['default'])
                    chunks = splitter.split_documents(documents)
                    
                    all_documents.extend(chunks)
                    
            except Exception as e:
                print(f"Error loading {pattern} files: {e}")
                continue
        
        print(f"Total chunks created: {len(all_documents)}")
        
        # Create FAISS vector store
        print("Creating FAISS vector store...")
        self.vectorstore = FAISS.from_documents(
            documents=all_documents,
            embedding=self.embeddings
        )
        
        # Save FAISS index to disk
        print(f"Saving FAISS index to {self.faiss_index_path}")
        self.vectorstore.save_local(self.faiss_index_path)
        
        # Create retrieval chain
        self.retrieval_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 8}
            )
        )
        
        print("RAG system ready!")
    
    def load_existing_vectorstore(self):
        """Load existing FAISS vector store from disk"""
        if Path(self.faiss_index_path).exists():
            print("Loading existing FAISS vector store...")
            try:
                self.vectorstore = FAISS.load_local(
                    self.faiss_index_path,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
                self.retrieval_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 8}
                    )
                )
                print("FAISS vector store loaded!")
                return True
            except Exception as e:
                print(f"Error loading FAISS index: {e}")
                return False
        return False
    
    def add_documents_to_existing_store(self, documents: List[Document]):
        """Add new documents to existing FAISS store"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        print(f"Adding {len(documents)} new documents to FAISS store...")
        self.vectorstore.add_documents(documents)
        
        # Save updated index
        self.vectorstore.save_local(self.faiss_index_path)
        print("Documents added and index updated!")
    
    def get_vectorstore_info(self):
        """Get information about the FAISS vector store"""
        if self.vectorstore is None:
            return "Vector store not initialized"
        
        # FAISS doesn't have a direct count method, but we can estimate
        try:
            # Try to get the dimension and approximate count
            index_info = {
                "type": "FAISS",
                "index_path": self.faiss_index_path,
                "status": "loaded" if self.vectorstore else "not loaded"
            }
            return index_info
        except Exception as e:
            return f"Error getting store info: {e}"
    
    def search_codebase(self, query: str, k: int = 5) -> List[Document]:
        """Search for relevant code chunks using FAISS similarity search"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Run load_and_chunk_codebase first.")
        
        docs = self.vectorstore.similarity_search(query, k=k)
        return docs
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Search with similarity scores"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized.")
        
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        return docs_with_scores
    
    def max_marginal_relevance_search(self, query: str, k: int = 5, fetch_k: int = 20) -> List[Document]:
        """Use MMR search for more diverse results"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized.")
        
        docs = self.vectorstore.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k
        )
        return docs
    
    def get_code_context(self, user_story: str) -> str:
        """Get relevant code context for user story"""
        if not self.retrieval_chain:
            raise ValueError("Retrieval chain not initialized.")
        
        context_query = f"""
        Find code examples and patterns related to: {user_story}
        Look for similar UI components, styling patterns, and implementation approaches.
        """
        
        result = self.retrieval_chain.run(context_query)
        return result

class CodeGenerationAgent:
    """Agentic code generator using LangChain agents"""
    
    def __init__(self, rag_system: CodebaseRAG):
        self.rag = rag_system
        self.llm = rag_system.llm
        
        # Create tools for the agent
        self.tools = self._create_tools()
        
        # Create agent prompt
        self.agent_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert software developer and UI/UX designer. 
            You have access to a codebase through RAG tools and can generate high-quality code.
            
            Your capabilities:
            1. Search through existing codebase for patterns and examples
            2. Generate new UI code based on user stories
            3. Ensure consistency with existing code style and patterns
            4. Create complete, functional components
            
            Always:
            - Follow the existing code patterns and conventions
            - Write clean, maintainable code
            - Include proper documentation and comments
            - Consider responsive design and accessibility
            - Use appropriate styling frameworks if detected in codebase"""),
            
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent
        self.agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.agent_prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent"""
        
        def search_codebase_tool(query: str) -> str:
            """Search the codebase for relevant code examples using FAISS"""
            try:
                # Use MMR search for more diverse results
                docs = self.rag.max_marginal_relevance_search(query, k=5)
                results = []
                
                for i, doc in enumerate(docs):
                    results.append(f"=== Result {i+1} ===")
                    results.append(f"File: {doc.metadata.get('source', 'Unknown')}")
                    results.append(f"Language: {doc.metadata.get('language', 'Unknown')}")
                    results.append(f"File Type: {doc.metadata.get('file_type', 'Unknown')}")
                    results.append("Content:")
                    
                    # Truncate long content but keep it readable
                    content = doc.page_content
                    if len(content) > 1500:
                        content = content[:1500] + "\n...[truncated]"
                    
                    results.append(content)
                    results.append("-" * 50)
                
                return "\n".join(results) if results else "No relevant code found."
                
            except Exception as e:
                return f"Error searching codebase: {e}"
        
        def search_with_scores_tool(query: str) -> str:
            """Search codebase with similarity scores"""
            try:
                docs_with_scores = self.rag.similarity_search_with_score(query, k=3)
                results = []
                
                for doc, score in docs_with_scores:
                    results.append(f"Score: {score:.3f}")
                    results.append(f"File: {doc.metadata.get('source', 'Unknown')}")
                    results.append(f"Content: {doc.page_content[:500]}...")
                    results.append("-" * 30)
                
                return "\n".join(results) if results else "No relevant code found."
                
            except Exception as e:
                return f"Error searching with scores: {e}"
        
        def get_context_tool(user_story: str) -> str:
            """Get comprehensive context for a user story"""
            try:
                return self.rag.get_code_context(user_story)
            except Exception as e:
                return f"Error getting context: {e}"
        
        def analyze_patterns_tool(component_type: str) -> str:
            """Analyze patterns for specific component types using FAISS"""
            query = f"{component_type} component pattern structure style implementation"
            try:
                # Use regular similarity search for pattern analysis
                docs = self.rag.search_codebase(query, k=8)
                
                patterns = {
                    'styling': set(),
                    'structure': [],
                    'imports': set(),
                    'frameworks': set(),
                    'file_types': set()
                }
                
                for doc in docs:
                    content = doc.page_content.lower()
                    metadata = doc.metadata
                    
                    # Track file types
                    if 'file_type' in metadata:
                        patterns['file_types'].add(metadata['file_type'])
                    
                    # Extract styling patterns
                    if 'classname' in content or 'class=' in content:
                        patterns['styling'].add('CSS classes')
                    if 'styled-components' in content or 'styled.' in content:
                        patterns['styling'].add('Styled Components')
                    if 'tailwind' in content or 'tw-' in content or 'bg-' in content:
                        patterns['styling'].add('Tailwind CSS')
                    if 'css-modules' in content or 'module.css' in content:
                        patterns['styling'].add('CSS Modules')
                    if '@emotion' in content or 'emotion' in content:
                        patterns['styling'].add('Emotion')
                    
                    # Extract framework patterns
                    if 'react' in content or 'usestate' in content or 'useeffect' in content:
                        patterns['frameworks'].add('React')
                    if 'vue' in content or 'v-if' in content or 'v-for' in content:
                        patterns['frameworks'].add('Vue')
                    if 'angular' in content or '@component' in content:
                        patterns['frameworks'].add('Angular')
                    
                    # Extract common imports (first few lines usually)
                    lines = doc.page_content.split('\n')[:10]  # Check first 10 lines
                    for line in lines:
                        line_clean = line.strip()
                        if line_clean.startswith('import') and len(patterns['imports']) < 5:
                            patterns['imports'].add(line_clean)
                
                # Build comprehensive analysis
                result = f"🔍 Pattern Analysis for '{component_type}':\n\n"
                
                result += f"📁 File Types Found: {', '.join(patterns['file_types']) if patterns['file_types'] else 'Mixed'}\n"
                result += f"🎨 Styling Approaches: {', '.join(patterns['styling']) if patterns['styling'] else 'Standard CSS'}\n"
                result += f"⚛️ Frameworks Detected: {', '.join(patterns['frameworks']) if patterns['frameworks'] else 'Vanilla/Unknown'}\n\n"
                
                if patterns['imports']:
                    result += "📦 Common Imports:\n"
                    for imp in list(patterns['imports'])[:3]:
                        result += f"  • {imp}\n"
                
                result += f"\n📊 Analysis based on {len(docs)} relevant code examples"
                
                return result
                
            except Exception as e:
                return f"Error analyzing patterns: {e}"
        
        return [
            Tool(
                name="search_codebase",
                description="Search the codebase for relevant code examples and patterns. Use for finding similar implementations.",
                func=search_codebase_tool
            ),
            Tool(
                name="search_with_scores", 
                description="Search codebase with similarity scores to find the most relevant matches.",
                func=search_with_scores_tool
            ),
            Tool(
                name="get_context",
                description="Get comprehensive context and examples for implementing a user story",
                func=get_context_tool
            ),
            Tool(
                name="analyze_patterns",
                description="Analyze existing patterns for specific component types (e.g., 'button', 'form', 'navigation'). Returns detailed pattern analysis.",
                func=analyze_patterns_tool
            )
        ]
    
    def generate_code(self, user_story: str) -> str:
        """Generate code based on user story using agent"""
        prompt = f"""
        User Story: {user_story}
        
        Please:
        1. Search the codebase for relevant patterns and examples
        2. Analyze the existing code style and conventions  
        3. Generate complete, production-ready code that implements the user story
        4. Ensure the code follows the project's patterns and best practices
        
        Generate the complete implementation with:
        - All necessary imports
        - Component structure
        - Styling (following project conventions)
        - Props/interfaces if needed
        - Basic functionality
        - Comments explaining key parts
        """
        
        result = self.agent_executor.invoke({"input": prompt})
        return result['output']

class SimpleRAGCodeGen:
    """Main class to orchestrate everything"""
    
    def __init__(self, openai_api_key: str):
        self.rag = CodebaseRAG(openai_api_key)
        self.agent = None
    
    def setup(self, codebase_path: str = None):
        """Setup the system - load existing or create new"""
        # Try to load existing vector store first
        if self.rag.load_existing_vectorstore():
            print("Using existing vector store")
        elif codebase_path:
            print("Creating new vector store from codebase")
            self.rag.load_and_chunk_codebase(codebase_path)
        else:
            raise ValueError("No existing vector store found and no codebase path provided")
        
        # Initialize agent
        self.agent = CodeGenerationAgent(self.rag)
        print("System ready!")
    
    def generate_ui_code(self, user_story: str) -> str:
        """Generate UI code from user story"""
        if not self.agent:
            raise ValueError("System not setup. Call setup() first.")
        
        return self.agent.generate_code(user_story)
    
    def search_code(self, query: str, search_type: str = "similarity") -> List[Document]:
        """Search existing code with different search strategies"""
        if search_type == "similarity":
            return self.rag.search_codebase(query)
        elif search_type == "mmr":
            return self.rag.max_marginal_relevance_search(query)
        elif search_type == "with_scores":
            return self.rag.similarity_search_with_score(query)
        else:
            return self.rag.search_codebase(query)
    
    def get_vectorstore_info(self):
        """Get vector store information"""
        return self.rag.get_vectorstore_info()
    
    def add_new_documents(self, file_paths: List[str]):
        """Add new documents to existing FAISS store"""
        documents = []
        
        for file_path in file_paths:
            if Path(file_path).exists():
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                
                # Add metadata
                for doc in docs:
                    doc.metadata['language'] = self._detect_language(file_path)
                    doc.metadata['file_type'] = Path(file_path).suffix.replace('.', '')
                
                # Split documents
                splitter = self.rag.text_splitters.get(
                    doc.metadata.get('language', 'default'), 
                    self.rag.text_splitters['default']
                )
                chunks = splitter.split_documents(docs)
                documents.extend(chunks)
        
        if documents:
            self.rag.add_documents_to_existing_store(documents)
            # Reinitialize agent with updated store
            self.agent = CodeGenerationAgent(self.rag)
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext_to_lang = {
            '.py': 'python',
            '.js': 'javascript', 
            '.jsx': 'javascript',
            '.ts': 'javascript',
            '.tsx': 'javascript',
            '.html': 'html',
            '.css': 'default',
            '.vue': 'javascript'
        }
        ext = Path(file_path).suffix
        return ext_to_lang.get(ext, 'default')

# Example usage with FAISS optimizations
def main():
    """Example usage of the FAISS-powered RAG system"""
    
    # Setup
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Initialize system
    code_gen = SimpleRAGCodeGen(OPENAI_API_KEY)
    
    # Setup with codebase (run once, then it will use saved FAISS index)
    CODEBASE_PATH = "path/to/your/codebase"  # Update this path
    code_gen.setup(CODEBASE_PATH)
    
    # Display vector store info
    print("\n" + "="*60)
    print("VECTOR STORE INFO:")
    print("="*60)
    print(code_gen.get_vectorstore_info())
    
    # Example user stories
    user_stories = [
        "Create a responsive login form with email validation and loading states",
        "Build a dashboard navigation sidebar with collapsible menu items", 
        "Design a product card component with image, title, price and add to cart button",
        "Implement a modal dialog for user profile editing with form validation",
        "Create a data table with sorting, filtering and pagination"
    ]
    
    # Demonstrate different search capabilities
    print("\n" + "="*60)
    print("TESTING FAISS SEARCH CAPABILITIES")
    print("="*60)
    
    # Test different search types
    test_query = "navigation menu component"
    
    print(f"\n🔍 Similarity Search for '{test_query}':")
    docs = code_gen.search_code(test_query, "similarity")
    print(f"Found {len(docs)} similar documents")
    
    print(f"\n🔍 MMR Search for '{test_query}':")
    docs_mmr = code_gen.search_code(test_query, "mmr")
    print(f"Found {len(docs_mmr)} diverse documents")
    
    print(f"\n🔍 Search with Scores for '{test_query}':")
    docs_scores = code_gen.search_code(test_query, "with_scores")
    if docs_scores:
        for doc, score in docs_scores[:2]:  # Show top 2 with scores
            print(f"Score: {score:.3f} - {doc.metadata.get('source', 'Unknown')}")
    
    # Generate code for each story
    for i, story in enumerate(user_stories, 1):
        print(f"\n{'='*60}")
        print(f"GENERATING CODE FOR: {story}")
        print('='*60)
        
        try:
            generated_code = code_gen.generate_ui_code(story)
            
            # Save to file
            output_file = f"generated_component_{i}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"User Story: {story}\n\n")
                f.write("Generated Code:\n")
                f.write("="*50 + "\n")
                f.write(generated_code)
            
            print(f"\n✅ Code generated and saved to {output_file}")
            
        except Exception as e:
            print(f"❌ Error generating code: {e}")

# Additional utility functions for FAISS
def benchmark_search_performance():
    """Benchmark FAISS vs other vector stores"""
    import time
    
    # This would compare search times between FAISS and other vector stores
    # FAISS is typically 5-10x faster for similarity search
    pass

def faiss_index_stats(faiss_index_path: str):
    """Get FAISS index statistics"""
    if not Path(faiss_index_path).exists():
        return "FAISS index not found"
    
    try:
        # Load embeddings to get FAISS index
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            faiss_index_path, 
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Get basic info
        return {
            "index_path": faiss_index_path,
            "status": "loaded",
            "type": "FAISS",
            "search_types_supported": ["similarity", "mmr", "with_scores"]
        }
    except Exception as e:
        return f"Error reading FAISS index: {e}"

if __name__ == "__main__":
    main()
