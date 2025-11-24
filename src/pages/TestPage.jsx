import React from 'react'

const TestPage = () => {
  return (
    <div style={{ 
      padding: '50px', 
      background: '#f0fdf4', 
      minHeight: '100vh',
      color: '#1f2937',
      fontSize: '24px'
    }}>
      <h1 style={{ color: '#22c55e' }}>GreenGo Test Page</h1>
      <p>If you can see this, React is working!</p>
      <div style={{ 
        background: 'white', 
        padding: '20px', 
        borderRadius: '10px',
        marginTop: '20px',
        boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
      }}>
        <p>This is a test to verify the app is rendering.</p>
      </div>
    </div>
  )
}

export default TestPage

