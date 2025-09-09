import React, { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// Error boundary for debugging
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('App Error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ 
          padding: '20px', 
          background: '#fff5f5', 
          color: '#e53e3e', 
          border: '1px solid #fed7d7',
          borderRadius: '8px',
          margin: '20px'
        }}>
          <h2>Something went wrong</h2>
          <details>
            <summary>Error Details</summary>
            <pre style={{ fontSize: '12px', overflow: 'auto' }}>
              {this.state.error?.toString()}
            </pre>
          </details>
        </div>
      );
    }

    return this.props.children;
  }
}

const root = createRoot(document.getElementById('root'));

// Check if root element exists
if (!root) {
  console.error('Root element not found');
  document.body.innerHTML = '<div style="padding: 20px; color: red;">Root element not found. Check index.html</div>';
} else {
  try {
    root.render(
      <StrictMode>
        <ErrorBoundary>
          <App />
        </ErrorBoundary>
      </StrictMode>
    );
    console.log('App rendered successfully');
  } catch (error) {
    console.error('Render error:', error);
    document.body.innerHTML = `<div style="padding: 20px; color: red;">Render failed: ${error.message}</div>`;
  }
}
