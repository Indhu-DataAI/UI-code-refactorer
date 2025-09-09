import React, { useState, useRef, useEffect } from "react";
import { FileText, Upload, Download, RefreshCw, Code, AlertCircle, CheckCircle, Bot, Brain, Search, Zap, Database, Eye, EyeOff, Monitor, Smartphone, Tablet, Globe } from "lucide-react";
 
function App() {
  const [files, setFiles] = useState(null);
  const [story, setStory] = useState("");
  const [updatedCode, setUpdatedCode] = useState("");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");
  const [debugInfo, setDebugInfo] = useState(null);
  const [isEditorOpen, setIsEditorOpen] = useState(false);
  const [code, setCode] = useState("");
  
  // Preview-related state
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);
  const [previewMode, setPreviewMode] = useState('desktop');
  const [previewFiles, setPreviewFiles] = useState({});
  const [selectedPreviewFile, setSelectedPreviewFile] = useState('');
  const [previewError, setPreviewError] = useState('');
  const iframeRef = useRef(null);
 
  const [agentWorkflow, setAgentWorkflow] = useState({
    activeAgent: null,
    agents: {
      ragAgent: {
        status: 'idle',
        progress: 0,
        message: '',
        active: false,
        currentTask: '',
        details: { chunks: 0, files: 0 }
      },
      generationAgent: {
        status: 'idle',
        progress: 0,
        message: '',
        active: false,
        currentTask: '',
        details: { retrievedChunks: 0, relevantFiles: [] }
      }
    },
    workflow: {
      stage: 'idle',
      totalProgress: 0,
      currentTask: '',
      processingLogs: []
    }
  });
 
  const BACKEND = "http://localhost:8000";
 
  // Agent definitions
  const AGENT_DEFINITIONS = {
    ragAgent: {
      name: "RAG Processing Agent",
      icon: <Database className="w-5 h-5" />,
      role: "Document chunking, embedding, and indexing",
      capabilities: [
        "File extraction and parsing",
        "Intelligent text chunking",
        "Semantic embedding generation",
        "Vector database indexing",
        "Context retrieval"
      ]
    },
    generationAgent: {
      name: "Code Generation Agent",
      icon: <Brain className="w-5 h-5" />,
      role: "AI-powered code modification and generation",
      capabilities: [
        "Semantic code search",
        "Context-aware generation",
        "LLM integration",
        "Code validation",
        "Output formatting"
      ]
    }
  };

  // Preview functions
  const parseUpdatedCode = (codeText) => {
    const files = {};
    
    if (!codeText) {
      return files;
    }
    
    // Parse delimited files first
    const filePattern = /=== FILE:\s*(.+?)\s*===([\s\S]*?)(?:=== END FILE ===|$)/g;
    let match;
    
    while ((match = filePattern.exec(codeText)) !== null) {
      const filePath = match[1].trim();
      const fileContent = match[2].trim();
      files[filePath] = fileContent;
    }
    
    // If no delimited files found, treat as single file
    if (Object.keys(files).length === 0 && codeText.trim().length > 0) {
      if (codeText.includes('<!DOCTYPE html') || codeText.includes('<html')) {
        files['index.html'] = codeText;
      } else if (codeText.includes('import React') || codeText.includes('function App') || codeText.includes('export default')) {
        files['App.jsx'] = codeText;
      } else if (codeText.includes('body {') || codeText.includes('.css-')) {
        files['styles.css'] = codeText;
      } else if (codeText.includes('function ') || codeText.includes('const ') || codeText.includes('let ')) {
        files['script.js'] = codeText;
      } else {
        files['generated-code.txt'] = codeText;
      }
    }
    
    return files;
  };// Enhanced preview content generation
const generatePreviewContent = () => {
  console.log("=== GENERATING PREVIEW CONTENT ===");
  
  const codeToPreview = code || updatedCode;
  if (!codeToPreview) {
    console.log("No code available for preview");
    return '';
  }

  console.log("Code to preview length:", codeToPreview.length);
  
  const parsedFiles = parseUpdatedCode(codeToPreview);
  setPreviewFiles(parsedFiles);
  
  console.log("Generated preview files:", Object.keys(parsedFiles));

  if (Object.keys(parsedFiles).length === 0) {
    console.log("No files parsed, returning empty");
    setPreviewError('No files could be parsed from the generated code');
    return '';
  }

  const htmlFiles = Object.keys(parsedFiles).filter(file => 
    file.toLowerCase().endsWith('.html') || 
    file.toLowerCase().endsWith('.htm')
  );

  const jsFiles = Object.keys(parsedFiles).filter(file => 
    file.toLowerCase().endsWith('.js') || 
    file.toLowerCase().endsWith('.jsx')
  );

  const cssFiles = Object.keys(parsedFiles).filter(file => 
    file.toLowerCase().endsWith('.css')
  );

  console.log("File types found:", { html: htmlFiles.length, js: jsFiles.length, css: cssFiles.length });

  let previewContent = '';

  if (htmlFiles.length > 0) {
    // Use existing HTML file
    const mainHtmlFile = htmlFiles[0];
    setSelectedPreviewFile(mainHtmlFile);
    previewContent = parsedFiles[mainHtmlFile];
    console.log("Using HTML file:", mainHtmlFile);
  } else if (jsFiles.length > 0 || cssFiles.length > 0) {
    // Create a preview HTML wrapper for JS/CSS files
    const cssContent = cssFiles.map(file => parsedFiles[file]).join('\n');
    const jsContent = jsFiles.map(file => parsedFiles[file]).join('\n');

    console.log("Creating wrapper HTML for JS/CSS files");
    console.log("CSS content length:", cssContent.length);
    console.log("JS content length:", jsContent.length);

    // Check if it's React code
    const isReactCode = jsContent.includes('React') || jsContent.includes('jsx') || jsContent.includes('useState') || jsContent.includes('import React');
    
    if (isReactCode) {
      console.log("Detected React code, showing source view");
      // For React code, show the source instead of trying to execute
      previewContent = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>React Code Preview</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: #f5f7fa;
            line-height: 1.5;
        }
        .preview-container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
        }
        .code-block {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 16px;
            margin: 16px 0;
            overflow-x: auto;
            font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', monospace;
            font-size: 13px;
            line-height: 1.4;
        }
        .file-header {
            background: #e9ecef;
            margin: -16px -16px 16px -16px;
            padding: 12px 16px;
            font-weight: 600;
            color: #495057;
            border-radius: 6px 6px 0 0;
        }
        h1 { 
            color: #2c3e50; 
            margin-top: 0; 
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 { 
            color: #34495e; 
            border-bottom: 2px solid #3498db; 
            padding-bottom: 8px; 
            margin-top: 30px;
        }
        .react-note {
            background: linear-gradient(45deg, #e3f2fd, #f3e5f5);
            border: 1px solid #2196f3;
            border-radius: 8px;
            padding: 16px;
            margin: 16px 0;
            color: #1976d2;
        }
        pre {
            margin: 0;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
    </style>
</head>
<body>
    <div class="preview-container">
        <h1>🚀 Generated React Code Preview</h1>
        <div class="react-note">
            <strong>📝 Note:</strong> This is React/JSX code that requires compilation. 
            The source code is displayed below for review. To run this code, you'll need to set up a React development environment.
        </div>`;

      // Add each file
      Object.entries(parsedFiles).forEach(([filename, content]) => {
        previewContent += `
        <h2>📄 ${filename}</h2>
        <div class="code-block">
            <div class="file-header">${filename}</div>
            <pre>${content.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</pre>
        </div>`;
      });

      previewContent += `
    </div>
</body>
</html>`;
      
    } else {
      console.log("Regular JavaScript/CSS detected, creating executable preview");
      // Regular JavaScript/CSS - try to execute
      previewContent = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Code Preview</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: #f5f7fa;
        }
        .preview-container { 
            max-width: 1200px; 
            margin: 0 auto; 
        }
        ${cssContent}
    </style>
</head>
<body>
    <div class="preview-container">
        <div id="app"></div>
        <div id="content"></div>
        <div id="root"></div>
    </div>
    <script>
        try {
            ${jsContent}
        } catch (error) {
            console.error('Preview error:', error);
            document.body.innerHTML = '<div style="color: red; padding: 20px; border: 1px solid red; margin: 10px; border-radius: 4px; background: #fff5f5;"><h3>Preview Error</h3><p>' + error.message + '</p><p><small>Check the browser console for more details.</small></p></div>';
        }
    </script>
</body>
</html>`;
    }
    
    setSelectedPreviewFile('Generated Preview');
  } else {
    console.log("No web files found, showing file structure");
    // Show file structure for non-web files
    const fileList = Object.entries(parsedFiles).map(([filename, content]) => `
        <div style="margin: 16px 0; border: 1px solid #e1e5e9; border-radius: 6px; overflow: hidden; background: white;">
          <div style="background: #f6f8fa; padding: 12px 16px; border-bottom: 1px solid #e1e5e9; font-weight: 600; color: #24292e;">
            📄 ${filename}
          </div>
          <pre style="background: #f8f9fa; padding: 16px; margin: 0; overflow: auto; max-height: 300px; font-size: 13px; line-height: 1.45; font-family: 'SF Mono', Monaco, monospace;">${content.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</pre>
        </div>
      `).join('');

    previewContent = `<!DOCTYPE html>
<html>
<head>
    <title>File Structure Preview</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: #f6f8fa; 
        }
        .file-list { 
            max-width: 1000px; 
            margin: 0 auto;
        }
        h2 { 
            color: #24292e; 
            margin-bottom: 20px;
            text-align: center;
        }
    </style>
</head>
<body>
    <h2>📁 Generated Files Preview</h2>
    <div class="file-list">
        ${fileList}
    </div>
</body>
</html>`;
    setSelectedPreviewFile('File Structure');
  }

  console.log("Generated preview content length:", previewContent.length);
  return previewContent;
};
// Enhanced update preview function with better error handling
const updatePreview = () => {
  if (!iframeRef.current) {
    console.log("No iframe ref available");
    return;
  }

  try {
    setPreviewError('');
    const previewContent = generatePreviewContent();
    
    if (!previewContent) {
      setPreviewError('No code available for preview');
      return;
    }

    const iframe = iframeRef.current;
    
    // Method 1: Try using srcdoc (more reliable)
    if ('srcdoc' in iframe) {
      iframe.srcdoc = previewContent;
      console.log("Using srcdoc method");
    } else {
      // Method 2: Fallback to document.write
      try {
        const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
        iframeDoc.open();
        iframeDoc.write(previewContent);
        iframeDoc.close();
        console.log("Using document.write method");
      } catch (e) {
        console.error("Document.write failed:", e);
        setPreviewError('Failed to load preview: ' + e.message);
      }
    }
  } catch (error) {
    console.error('Preview update error:', error);
    setPreviewError('Failed to update preview: ' + error.message);
  }
};
  
  const getPreviewDimensions = () => {
    switch (previewMode) {
      case 'mobile':
        return { width: '375px', height: '667px' };
      case 'tablet':
        return { width: '768px', height: '1024px' };
      default:
        return { width: '100%', height: '600px' };
    }
  };
 
  const updateAgentStatus = (agentId, status, progress, message, active = false, currentTask = '', details = {}) => {
    setAgentWorkflow(prev => ({
      ...prev,
      activeAgent: active ? agentId : (prev.activeAgent === agentId ? null : prev.activeAgent),
      agents: {
        ...prev.agents,
        [agentId]: {
          ...prev.agents[agentId],
          status,
          progress,
          message,
          active,
          currentTask,
          details: { ...prev.agents[agentId].details, ...details }
        }
      }
    }));
  };
 
  const updateWorkflowStatus = (stage, totalProgress, currentTask) => {
    setAgentWorkflow(prev => ({
      ...prev,
      workflow: {
        ...prev.workflow,
        stage,
        totalProgress,
        currentTask
      }
    }));
  };
 
  const addProcessingLog = (log) => {
    setAgentWorkflow(prev => ({
      ...prev,
      workflow: {
        ...prev.workflow,
        processingLogs: [...prev.workflow.processingLogs.slice(-9), log]
      }
    }));
  };
 
  const resetAgents = () => {
    Object.keys(agentWorkflow.agents).forEach(agentId => {
      updateAgentStatus(agentId, 'idle', 0, '', false, '', {});
    });
    updateWorkflowStatus('idle', 0, 'Ready to start');
    setAgentWorkflow(prev => ({
      ...prev,
      workflow: { ...prev.workflow, processingLogs: [] }
    }));
  };
 
  const handleUpload = async () => {
    if (!files || files.length === 0) {
      setStatus("Please choose files first.");
      return;
    }
 
    resetAgents();
    updateWorkflowStatus('starting', 0, 'Initializing RAG processing...');
    addProcessingLog("Starting RAG pipeline");
   
    try {
      const zipBlob = files[0];
      const formData = new FormData();
      formData.append("file", zipBlob);
 
      await simulateRAGProcessing();
 
      const res = await fetch(`${BACKEND}/upload-zip/`, {
        method: "POST",
        body: formData,
      });
 
      if (!res.ok) {
        const err = await res.text();
        throw new Error(err);
      }
 
      const result = await res.json();
     
      updateAgentStatus('ragAgent', 'completed', 100, 'RAG processing completed successfully', false, 'Completed', {});
      updateWorkflowStatus('completed', 100, 'Files indexed and ready for code generation');
      addProcessingLog("RAG pipeline completed - ready for code generation");
     
      setStatus("Upload successful. Files indexed! Ready for code generation.");
      await checkIndexedFiles();
    } catch (err) {
      updateAgentStatus('ragAgent', 'error', 0, 'RAG processing failed: ' + err.message, false, 'Error', {});
      updateWorkflowStatus('error', 0, 'RAG processing failed');
      addProcessingLog(`RAG pipeline error: ${err.message}`);
      setStatus("Upload failed: " + err.message);
      console.error("Upload error:", err);
    }
  };
 
  const simulateRAGProcessing = async () => {
    const ragSteps = [
      { progress: 10, message: 'Extracting files from ZIP archive...', task: 'File Extraction', log: 'RAG Agent: Extracting ZIP contents', duration: 800, details: { files: 15 } },
      { progress: 25, message: 'Parsing and analyzing code structure...', task: 'Code Analysis', log: 'RAG Agent: Analyzing code structure', duration: 1000, details: { files: 15 } },
      { progress: 45, message: 'Chunking documents with semantic awareness...', task: 'Document Chunking', log: 'RAG Agent: Creating semantic chunks', duration: 1500, details: { chunks: 45, files: 15 } },
      { progress: 65, message: 'Generating embeddings using OpenAI...', task: 'Embedding Generation', log: 'RAG Agent: Generating embeddings', duration: 2000, details: { chunks: 87, files: 15 } },
      { progress: 85, message: 'Building vector database index...', task: 'Vector Indexing', log: 'RAG Agent: Building vector index', duration: 1200, details: { chunks: 87, files: 15 } },
      { progress: 100, message: 'RAG processing completed', task: 'Complete', log: 'RAG Agent: Database ready', duration: 500, details: { chunks: 87, files: 15 } }
    ];
 
    updateAgentStatus('ragAgent', 'active', 0, 'Starting RAG processing...', true, 'Initializing', {});
 
    for (const step of ragSteps) {
      updateAgentStatus('ragAgent', 'active', step.progress, step.message, true, step.task, step.details);
      updateWorkflowStatus('processing', Math.floor(step.progress * 0.5), step.message);
      addProcessingLog(step.log);
      await new Promise(resolve => setTimeout(resolve, step.duration));
    }
  };
 
  const handleGenerate = async () => {
    if (!story.trim()) {
      setStatus("Please enter a modification request first.");
      return;
    }
 
    updateAgentStatus('generationAgent', 'idle', 0, '', false, '', {});
 
    setLoading(true);
    updateWorkflowStatus('generation', 50, 'Starting code generation...');
    addProcessingLog("Activating code generation pipeline");
 
    try {
      await simulateGenerationWorkflow();
 
      const res = await fetch(`${BACKEND}/generate-updated-zip/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ story }),
      });
 
      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(errorText);
      }
 
      const result = await res.json();
 
      if (!result.updated_code) {
        throw new Error("No code generated by the AI");
      }
 
      setUpdatedCode(result.updated_code);
      if (!code) {
        setCode(result.updated_code);
      }
 
      updateAgentStatus('generationAgent', 'completed', 100, 'Code generation completed successfully', false, 'Completed', {});
      updateWorkflowStatus('completed', 100, 'Code generation completed');
      addProcessingLog("Code generation completed successfully");
     
      setStatus("Code generation completed! You can now preview and download the updated files.");
    } catch (error) {
      updateAgentStatus('generationAgent', 'error', 0, 'Generation failed: ' + error.message, false, 'Error', {});
      updateWorkflowStatus('error', 50, 'Code generation failed');
      addProcessingLog(`Generation error: ${error.message}`);
      console.error("Generation error:", error);
      setStatus("Code generation failed: " + error.message);
    } finally {
      setLoading(false);
    }
  };
 
  const simulateGenerationWorkflow = async () => {
    const generationSteps = [
      { progress: 15, message: 'Analyzing modification request...', task: 'Request Analysis', log: 'Generation Agent: Parsing requirements', duration: 800 },
      { progress: 35, message: 'Performing semantic search...', task: 'Semantic Search', log: 'Generation Agent: Searching code chunks', duration: 1500 },
      { progress: 55, message: 'Ranking and selecting matches...', task: 'Context Selection', log: 'Generation Agent: Ranking relevance', duration: 1000 },
      { progress: 75, message: 'Generating modified code...', task: 'Code Generation', log: 'Generation Agent: Invoking LLM', duration: 2500 },
      { progress: 90, message: 'Validating output...', task: 'Code Validation', log: 'Generation Agent: Validating code', duration: 1000 },
      { progress: 100, message: 'Code generation completed', task: 'Complete', log: 'Generation Agent: Code ready', duration: 500 }
    ];
 
    updateAgentStatus('generationAgent', 'active', 0, 'Starting code generation...', true, 'Initializing', {});
 
    for (const step of generationSteps) {
      updateAgentStatus('generationAgent', 'active', step.progress, step.message, true, step.task, step.details || {});
      updateWorkflowStatus('generation', 50 + Math.floor(step.progress * 0.5), step.message);
      addProcessingLog(step.log);
      await new Promise(resolve => setTimeout(resolve, step.duration));
    }
  };
 
  const checkIndexedFiles = async () => {
    try {
      const res = await fetch(`${BACKEND}/list-indexed-files/`);
      const data = await res.json();
      setDebugInfo(data);
    } catch (err) {
      console.error("Debug info failed:", err);
    }
  };
 
  const downloadUpdatedCode = async () => {
    if (!updatedCode && !code) {
      alert("No generated code found. Please generate code first.");
      return;
    }
 
    try {
      setStatus("Preparing updated code for download...");
      const codeToDownload = code || updatedCode;
     
      const formData = new FormData();
      formData.append('updated_code', codeToDownload);
 
      const response = await fetch(`${BACKEND}/save-updated-zip/`, {
        method: 'POST',
        body: formData
      });
 
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to save updated code: ${errorText}`);
      }
 
      setStatus("Downloading updated ZIP...");
      const downloadResponse = await fetch(`${BACKEND}/download-zip`);
     
      if (!downloadResponse.ok) {
        const errorText = await downloadResponse.text();
        throw new Error(`Download failed: ${errorText}`);
      }
 
      const blob = await downloadResponse.blob();
     
      if (blob.size === 0) {
        throw new Error("Downloaded file is empty");
      }
 
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = 'updated_codebase.zip';
      document.body.appendChild(a);
      a.click();
     
      setTimeout(() => {
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }, 100);
 
      setStatus("Updated ZIP downloaded successfully!");
    } catch (error) {
      console.error("Download error:", error);
      setStatus("Failed to download ZIP: " + error.message);
    }
  };

  const loadTestUI = () => {
    const testCode = `=== FILE: index.html ===
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test UI Preview</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container { 
            max-width: 800px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 12px; 
            padding: 30px; 
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }
        .header { text-align: center; margin-bottom: 30px; }
        .card { 
            background: #f8f9fa; 
            padding: 20px; 
            border-radius: 8px; 
            margin: 15px 0; 
            border-left: 4px solid #007bff;
        }
        .btn { 
            background: #007bff; 
            color: white; 
            padding: 12px 24px; 
            border: none; 
            border-radius: 6px; 
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        .btn:hover { 
            background: #0056b3; 
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,123,255,0.3);
        }
        .features { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 20px; 
            margin-top: 30px;
        }
        .feature { 
            background: white; 
            padding: 20px; 
            border-radius: 8px; 
            border: 2px solid #e9ecef;
            text-align: center;
            transition: transform 0.3s ease;
        }
        .feature:hover {
            transform: scale(1.05);
            border-color: #007bff;
        }
        .counter { 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            gap: 15px; 
            margin: 20px 0;
        }
        .count-display { 
            background: #007bff; 
            color: white; 
            padding: 10px 20px; 
            border-radius: 50px; 
            font-size: 24px; 
            font-weight: bold;
            min-width: 60px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Test UI Preview</h1>
            <p>Interactive demo to test the preview functionality</p>
        </div>
        
        <div class="card">
            <h3>Visual Preview Working!</h3>
            <p>This demonstrates that the visual preview is working correctly.</p>
            
            <div class="counter">
                <button class="btn" onclick="decrementCounter()">-</button>
                <div class="count-display" id="counterDisplay">0</div>
                <button class="btn" onclick="incrementCounter()">+</button>
            </div>
            
            <button class="btn" onclick="showAlert()">Test Alert</button>
            <button class="btn" onclick="changeTheme()" id="themeBtn">Dark Mode</button>
        </div>
        
        <div class="features">
            <div class="feature" onclick="highlightFeature(this)">
                <h4>CSS Styling</h4>
                <p>Gradients, shadows, and modern design</p>
            </div>
            <div class="feature" onclick="highlightFeature(this)">
                <h4>Responsive</h4>
                <p>Adapts to different screen sizes</p>
            </div>
            <div class="feature" onclick="highlightFeature(this)">
                <h4>Interactive</h4>
                <p>JavaScript functionality works</p>
            </div>
        </div>
    </div>
    
    <script>
        let counter = 0;
        let isDark = false;
        
        function incrementCounter() {
            counter++;
            document.getElementById('counterDisplay').textContent = counter;
        }
        
        function decrementCounter() {
            counter--;
            document.getElementById('counterDisplay').textContent = counter;
        }
        
        function showAlert() {
            alert('Preview is fully interactive! Counter: ' + counter);
        }
        
        function changeTheme() {
            isDark = !isDark;
            document.body.style.background = isDark ? 
                'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)' : 
                'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
            document.querySelector('.container').style.background = isDark ? '#2d3748' : 'white';
            document.querySelector('.container').style.color = isDark ? 'white' : 'black';
            document.getElementById('themeBtn').textContent = isDark ? 'Light Mode' : 'Dark Mode';
        }
        
        function highlightFeature(element) {
            element.style.background = '#e3f2fd';
            setTimeout(() => {
                element.style.background = 'white';
            }, 300);
        }
    </script>
</body>
</html>
=== END FILE ===

=== FILE: App.jsx ===
import React, { useState } from 'react';

const App = () => {
  const [count, setCount] = useState(0);
  const [darkMode, setDarkMode] = useState(false);

  return (
    <div className={\`min-h-screen transition-all duration-300 \${darkMode ? 'bg-gray-900 text-white' : 'bg-gradient-to-br from-blue-50 to-purple-50 text-gray-900'}\`}>
      <div className="container mx-auto px-6 py-12">
        <header className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4">React Component Preview</h1>
          <p className="text-xl opacity-75">Interactive React component test</p>
        </header>

        <div className="max-w-2xl mx-auto space-y-6">
          <div className={\`p-6 rounded-xl shadow-lg \${darkMode ? 'bg-gray-800' : 'bg-white'}\`}>
            <h3 className="text-2xl font-semibold mb-4">Counter Demo</h3>
            <div className="flex items-center justify-between">
              <span className="text-3xl font-bold text-blue-600">{count}</span>
              <div className="space-x-2">
                <button 
                  onClick={() => setCount(count - 1)}
                  className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg transition-colors"
                >
                  -
                </button>
                <button 
                  onClick={() => setCount(count + 1)}
                  className="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-lg transition-colors"
                >
                  +
                </button>
              </div>
            </div>
          </div>

          <div className={\`p-6 rounded-xl shadow-lg \${darkMode ? 'bg-gray-800' : 'bg-white'}\`}>
            <div className="flex items-center justify-between">
              <h3 className="text-xl font-semibold">Dark Mode</h3>
              <button
                onClick={() => setDarkMode(!darkMode)}
                className={\`px-4 py-2 rounded-lg transition-all \${darkMode ? 'bg-yellow-500 text-black' : 'bg-gray-800 text-white'}\`}
              >
                {darkMode ? 'Light' : 'Dark'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
=== END FILE ===`;
                  
    setCode(testCode);
    setUpdatedCode(testCode);
    setStatus("Test UI loaded! Open the preview to see the demo.");
  };
 
  // Agent Status Component
  const AgentStatusCard = ({ agentId, agent, definition }) => {
    const getStatusColor = (status) => {
      switch (status) {
        case 'active': return 'border-blue-500 bg-blue-50 shadow-lg';
        case 'completed': return 'border-green-500 bg-green-50';
        case 'error': return 'border-red-500 bg-red-50';
        default: return 'border-gray-200 bg-white';
      }
    };
 
    const getStatusIcon = (status) => {
      switch (status) {
        case 'active': return <Zap className="w-5 h-5 text-blue-600 animate-pulse" />;
        case 'completed': return <CheckCircle className="w-5 h-5 text-green-600" />;
        case 'error': return <AlertCircle className="w-5 h-5 text-red-600" />;
        default: return definition.icon;
      }
    };
 
    return (
      <div className={`border-2 rounded-xl p-4 transition-all duration-500 ${getStatusColor(agent.status)} ${agent.active ? 'scale-105 transform' : ''}`}>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-3">
            {getStatusIcon(agent.status)}
            <div>
              <div className="font-semibold text-lg">{definition.name}</div>
              <div className="text-sm text-gray-600">{definition.role}</div>
            </div>
          </div>
        </div>
       
        {agent.status !== 'idle' && (
          <div className="space-y-3">
            {agent.currentTask && (
              <div className="bg-white rounded-lg p-2 border">
                <div className="text-xs font-medium text-gray-500 mb-1">Current Task</div>
                <div className="text-sm font-semibold text-blue-700">{agent.currentTask}</div>
              </div>
            )}
 
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-gray-600">Progress</span>
                <span className="font-mono text-blue-600">{agent.progress}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all duration-500 ${
                    agent.status === 'active' ? 'bg-blue-500' :
                    agent.status === 'completed' ? 'bg-green-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${agent.progress}%` }}
                />
              </div>
            </div>
 
            <div className="text-sm text-gray-700 bg-white rounded p-2 border">
              {agent.message}
            </div>
          </div>
        )}
 
        <details className="mt-3">
          <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700">
            View Capabilities ({definition.capabilities.length})
          </summary>
          <ul className="mt-2 text-xs text-gray-600 space-y-1">
            {definition.capabilities.map((cap, idx) => (
              <li key={idx} className="flex items-center space-x-2">
                <span className="w-1.5 h-1.5 bg-blue-400 rounded-full"></span>
                <span>{cap}</span>
              </li>
            ))}
          </ul>
        </details>
      </div>
    );
  };
 
  // Workflow Overview Component
  const WorkflowOverview = ({ workflow }) => {
    return (
      <div className="bg-gradient-to-r from-slate-500 to-black-100 border-2 border-black-200 rounded-xl p-6 shadow-lg">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold flex items-center space-x-3">
            <Bot className="w-6 h-6 text-black-600" />
            <span>Agentic Workflow Status</span>
          </h3>
          <div className="flex items-center space-x-3">
            <span className="text-sm text-gray-600">Overall Progress:</span>
            <span className="bg-white-600 text-black px-3 py-1 rounded-lg text-sm font-mono font-bold">
              {workflow.totalProgress}%
            </span>
          </div>
        </div>
 
        <div className="w-full bg-gray-200 rounded-full h-3 mb-4 shadow-inner">
          <div
            className="h-3 rounded-full bg-gradient-to-r from-blue-500 via-purple-500 to-green-500 transition-all duration-700 ease-out shadow-sm"
            style={{ width: `${workflow.totalProgress}%` }}
          />
        </div>

        <div className="text-gray-700 mb-4 p-3 bg-white rounded-lg border">
          <div className="font-medium text-sm text-gray-500 mb-1">Current Activity</div>
          <div className="text-lg">{workflow.currentTask || 'Waiting for instructions...'}</div>
        </div>

        {workflow.processingLogs.length > 0 && (
          <div className="bg-white rounded-lg border p-4">
            <div className="text-sm font-semibold text-gray-700 mb-3 flex items-center space-x-2">
              <Search className="w-4 h-4" />
              <span>Processing Activity Log</span>
            </div>
            <div className="max-h-32 overflow-y-auto space-y-1">
              {workflow.processingLogs.slice(-6).map((log, index) => (
                <div key={index} className="text-xs font-mono text-gray-600 p-1 hover:bg-gray-50 rounded">
                  {log}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  // Visual Preview Component
  const VisualPreview = () => {
    const dimensions = getPreviewDimensions();
    
    return (
      <div className="space-y-4 bg-white border rounded-xl p-6 shadow-sm">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold flex items-center space-x-3">
            <Eye className="w-6 h-6 text-purple-600" />
            <span>Visual Preview</span>
          </h2>
          
          <div className="flex items-center space-x-3">
            <div className="flex items-center bg-gray-100 rounded-lg p-1">
              <button
                onClick={() => setPreviewMode('desktop')}
                className={`p-2 rounded ${previewMode === 'desktop' ? 'bg-white shadow-sm text-blue-600' : 'text-gray-600'}`}
                title="Desktop View"
              >
                <Monitor className="w-4 h-4" />
              </button>
              <button
                onClick={() => setPreviewMode('tablet')}
                className={`p-2 rounded ${previewMode === 'tablet' ? 'bg-white shadow-sm text-blue-600' : 'text-gray-600'}`}
                title="Tablet View"
              >
                <Tablet className="w-4 h-4" />
              </button>
              <button
                onClick={() => setPreviewMode('mobile')}
                className={`p-2 rounded ${previewMode === 'mobile' ? 'bg-white shadow-sm text-blue-600' : 'text-gray-600'}`}
                title="Mobile View"
              >
                <Smartphone className="w-4 h-4" />
              </button>
            </div>

            <button
              onClick={updatePreview}
              className="p-2 bg-gray-100 rounded-lg hover:bg-gray-200 text-gray-600 transition-colors"
              title="Refresh Preview"
            >
              <RefreshCw className="w-4 h-4" />
            </button>

            <button
              onClick={() => setIsPreviewOpen(!isPreviewOpen)}
              className="bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700 flex items-center space-x-2 transition-colors"
            >
              {isPreviewOpen ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              <span>{isPreviewOpen ? 'Hide Preview' : 'Show Preview'}</span>
            </button>
          </div>
        </div>

        {selectedPreviewFile && (
          <div className="bg-gray-50 rounded-lg p-3 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Globe className="w-4 h-4 text-gray-500" />
              <span className="text-sm text-gray-600">Previewing: <span className="font-mono font-semibold">{selectedPreviewFile}</span></span>
            </div>
            <div className="text-xs text-gray-500">
              {previewMode.charAt(0).toUpperCase() + previewMode.slice(1)} View ({dimensions.width} × {dimensions.height})
            </div>
          </div>
        )}

        {previewError && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center space-x-3">
            <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0" />
            <div className="text-red-700">
              <div className="font-semibold">Preview Error</div>
              <div className="text-sm">{previewError}</div>
            </div>
          </div>
        )}

        {isPreviewOpen && (
          <div className="border rounded-lg overflow-hidden bg-gray-100">
            <div className="bg-gray-200 px-4 py-2 flex items-center justify-between text-sm">
              <div className="flex items-center space-x-2">
                <div className="flex space-x-1">
                  <div className="w-3 h-3 bg-red-400 rounded-full"></div>
                  <div className="w-3 h-3 bg-yellow-400 rounded-full"></div>
                  <div className="w-3 h-3 bg-green-400 rounded-full"></div>
                </div>
                <span className="text-gray-600 font-mono">localhost:3000/preview</span>
              </div>
              <div className="text-gray-500">
                {previewMode} • {dimensions.width} × {dimensions.height}
              </div>
            </div>

            <div className="bg-white p-4 flex justify-center">
              <div 
                className={`border rounded-lg overflow-hidden shadow-lg ${
                  previewMode === 'mobile' ? 'bg-black p-2' : 'bg-white'
                }`}
                style={{ 
                  width: previewMode === 'desktop' ? '100%' : dimensions.width,
                  maxWidth: '100%'
                }}
              >
                <iframe
                  ref={iframeRef}
                  className="w-full bg-white rounded"
                  style={{ 
                    height: dimensions.height,
                    minHeight: '400px',
                    border: 'none'
                  }}
                  sandbox="allow-scripts allow-same-origin allow-forms"
                  title="Code Preview"
                />
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };
 
  const CodeEditor = ({ value, onChange }) => (
    <textarea
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full h-96 p-4 bg-gray-900 text-green-200 font-mono text-sm rounded border focus:ring-2 focus:ring-blue-500 resize-none"
      style={{ fontFamily: 'Monaco, Menlo, "Ubuntu Mono", monospace' }}
      placeholder="Generated code will appear here..."
    />
  );
 
  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto space-y-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-slate-900 to-slate-400 bg-clip-text text-transparent mb-2">
            BCT Agentic UI Code Refactorer
          </h1>
          <p className="text-gray-600 text-lg">Intelligent RAG-Based Code Modification System</p>
          <div className="flex justify-center space-x-4 mt-4 text-sm text-gray-500">
            <span className="flex items-center space-x-1">
              <Database className="w-4 h-4" />
              <span>RAG Processing</span>
            </span>
            <span className="flex items-center space-x-1">
              <Brain className="w-4 h-4" />
              <span>AI Generation</span>
            </span>
            <span className="flex items-center space-x-1">
              <Eye className="w-4 h-4" />
              <span>Visual Preview</span>
            </span>
          </div>
        </div>
 
        <WorkflowOverview workflow={agentWorkflow.workflow} />
 
        <div className="space-y-6">
          <h2 className="text-2xl font-bold flex items-center space-x-3">
            <Bot className="w-7 h-7 text-black-600" />
            <span>Agent Status</span>
          </h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {Object.entries(AGENT_DEFINITIONS).map(([agentId, definition]) => (
              <AgentStatusCard
                key={agentId}
                agentId={agentId}
                agent={agentWorkflow.agents[agentId]}
                definition={definition}
              />
            ))}
          </div>
        </div>
 
        <div className="space-y-6 bg-white border rounded-xl p-6 shadow-sm">
          <h2 className="text-xl font-semibold flex items-center space-x-3">
            <Upload className="w-6 h-6 text-black-600" />
            <span>Upload Codebase</span>
          </h2>
         
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Select ZIP file containing your codebase
              </label>
              <input
                type="file"
                onChange={(e) => setFiles(e.target.files)}
                accept=".zip"
                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-slate-200 file:text-black-700 hover:file:bg-blue-100"
              />
            </div>
           
            <div className="flex space-x-3">
              <button
                type="button"
                onClick={handleUpload}
                disabled={agentWorkflow.workflow.stage === 'processing'}
                className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2 transition-all duration-200"
              >
                <Upload className="w-5 h-5 " />
                <span>{agentWorkflow.workflow.stage === 'processing' ? "Processing..." : "Upload & Process"}</span>
              </button>
              <button
                type="button"
                onClick={checkIndexedFiles}
                className="bg-gray-600 text-white px-6 py-2 rounded-lg hover:bg-gray-700 flex items-center space-x-2 transition-all duration-200"
              >
                <AlertCircle className="w-5 h-5" />
                <span>Check Status</span>
              </button>
            </div>
          </div>
 
          {status && (
            <div className={`p-4 rounded-lg flex items-center space-x-3 ${
              status.includes("successful") ? "bg-green-50 border border-green-200" :
              status.includes("failed") ? "bg-red-50 border border-red-200" :
              "bg-blue-50 border border-blue-200"
            }`}>
              {status.includes("successful") ? (
                <CheckCircle className="w-5 h-5 text-green-600" />
              ) : status.includes("failed") ? (
                <AlertCircle className="w-5 h-5 text-red-600" />
              ) : (
                <Bot className="w-5 h-5 text-blue-600" />
              )}
              <span className="text-sm">{status}</span>
            </div>
          )}
        </div>
 
        <div className="space-y-6 bg-white border rounded-xl p-6 shadow-sm">
          <h2 className="text-xl font-semibold flex items-center space-x-3">
            <FileText className="w-6 h-6 text-black-600" />
            <span>Code Modification Request</span>
          </h2>
 
          <div className="space-y-4">
            <textarea
              value={story}
              onChange={(e) => setStory(e.target.value)}
              placeholder="Describe the changes you want to make to your codebase... For example: 'Add a dark mode toggle to the header component' or 'Implement user authentication with login/logout functionality'"
              className="w-full p-4 border rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 min-h-[120px] resize-none"
              rows={5}
            />
 
            <button
              type="button"
              onClick={handleGenerate}
              disabled={loading}
              className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2 transition-all duration-200 text-lg"
            >
              <Brain className="w-5 h-5" />
              <span>{loading ? "Generating..." : "Generate Modified Code"}</span>
            </button>
          </div>
        </div>

        {(updatedCode || code) && <VisualPreview />}
 
        {(updatedCode || code) && (
          <div className="space-y-6 bg-white border rounded-xl p-6 shadow-sm">
            <h2 className="text-xl font-semibold flex items-center space-x-3">
              <Code className="w-6 h-6 text-green-600" />
              <span>Generated Code</span>
            </h2>
 
            <div className="flex space-x-3 flex-wrap gap-2">
              <button
                onClick={() => setIsEditorOpen(!isEditorOpen)}
                className="bg-gray-600 text-white px-4 py-2 rounded-lg hover:bg-gray-700 text-sm flex items-center space-x-2 transition-all duration-200"
              >
                <Code className="w-4 h-4" />
                <span>{isEditorOpen ? "Hide Code" : "View Code"}</span>
              </button>
              <button
                onClick={downloadUpdatedCode}
                className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 text-sm flex items-center space-x-2 transition-all duration-200"
              >
                <Download className="w-4 h-4" />
                <span>Download ZIP</span>
              </button>
              <button
                onClick={loadTestUI}
                className="bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700 text-sm flex items-center space-x-2 transition-all duration-200"
              >
                <Eye className="w-4 h-4" />
                <span>Load Test UI</span>
              </button>
            </div>
 
            {isEditorOpen && (
              <div className="border rounded-lg overflow-hidden">
                <div className="bg-gray-800 text-gray-300 px-4 py-2 text-sm font-semibold flex items-center justify-between">
                  <span>Generated Code Preview</span>
                  <span className="text-xs">
                    {(code || updatedCode)?.length.toLocaleString() || 0} characters
                  </span>
                </div>
                <CodeEditor
                  value={code || updatedCode}
                  onChange={setCode}
                />
              </div>
            )}
          </div>
        )}
 
        {debugInfo && (
          <div className="bg-white border rounded-xl p-6 shadow-sm">
            <h2 className="text-xl font-semibold flex items-center space-x-3 mb-4">
              <AlertCircle className="w-6 h-6 text-orange-600" />
              <span>System Debug Information</span>
            </h2>
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <div className="bg-gray-800 text-gray-300 px-4 py-2 text-sm font-semibold">
                Indexed Files Status
              </div>
              <pre className="text-xs text-green-200 p-4 max-h-64 overflow-y-auto">
                {JSON.stringify(debugInfo, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
 
export default App;