import React from 'react'
import { Routes, Route } from 'react-router-dom'
import LandingPage from './pages/LandingPage'
import DestinationPage from './pages/DestinationPage'
import RoutesPage from './pages/RoutesPage'
import DrivePage from './pages/DrivePage'
import HistoryPage from './pages/HistoryPage'
import SettingsPage from './pages/SettingsPage'
import TestPage from './pages/TestPage'

function App() {
  // Suppress console warnings in production
  if (import.meta.env.PROD) {
    const originalWarn = console.warn
    console.warn = (...args) => {
      if (!args[0]?.includes('React Router Future Flag')) {
        originalWarn(...args)
      }
    }
  }

  return (
    <div className="app" style={{ minHeight: '100vh', width: '100%', background: '#f9fafb' }}>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/test" element={<TestPage />} />
        <Route path="/destination" element={<DestinationPage />} />
        <Route path="/routes" element={<RoutesPage />} />
        <Route path="/drive" element={<DrivePage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/settings" element={<SettingsPage />} />
      </Routes>
    </div>
  )
}

export default App

