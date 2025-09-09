import React from 'react';

function TestApp() {
  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      padding: '20px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center'
    }}>
      <div style={{
        background: 'white',
        borderRadius: '12px',
        padding: '40px',
        boxShadow: '0 10px 25px rgba(0,0,0,0.1)',
        maxWidth: '600px',
        textAlign: 'center'
      }}>
        <h1 style={{ 
          color: '#333', 
          marginBottom: '20px',
          fontSize: '2.5rem'
        }}>
          ✅ UI Preview is Working!
        </h1>
        <p style={{ 
          color: '#666', 
          fontSize: '1.2rem',
          marginBottom: '30px'
        }}>
          Your React application is rendering correctly. The white page issue has been fixed.
        </p>
        <div style={{
          background: '#f8f9fa',
          padding: '20px',
          borderRadius: '8px',
          marginBottom: '20px'
        }}>
          <h3 style={{ color: '#495057', marginBottom: '10px' }}>✨ What was fixed:</h3>
          <ul style={{ 
            textAlign: 'left', 
            color: '#6c757d',
            lineHeight: '1.6'
          }}>
            <li>Removed conflicting CSS styles from index.css</li>
            <li>Fixed body display:flex centering that was hiding content</li>
            <li>Added proper error boundary for debugging</li>
            <li>Ensured root container takes full height</li>
          </ul>
        </div>
        <button 
          onClick={() => alert('Interactive features are working!')}
          style={{
            background: '#007bff',
            color: 'white',
            border: 'none',
            padding: '12px 24px',
            borderRadius: '6px',
            fontSize: '16px',
            cursor: 'pointer'
          }}
        >
          Test Interactivity
        </button>
      </div>
    </div>
  );
}

export default TestApp;