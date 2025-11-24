import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Clock, MapPin, ArrowRight, Trash2 } from 'lucide-react'
import NavigationBar from '../components/NavigationBar'
import { storage } from '../utils/storage'
import './HistoryPage.css'

const HistoryPage = () => {
  const navigate = useNavigate()
  const [history, setHistory] = useState([])

  useEffect(() => {
    loadHistory()
    // Refresh history when page becomes visible
    const handleFocus = () => loadHistory()
    window.addEventListener('focus', handleFocus)
    return () => window.removeEventListener('focus', handleFocus)
  }, [])

  const loadHistory = () => {
    const savedHistory = storage.getJourneys()
    // Also check legacy key for backwards compatibility
    const legacyHistory = JSON.parse(localStorage.getItem('greengo_history') || '[]')
    const allHistory = [...savedHistory, ...legacyHistory]
    // Remove duplicates based on id
    const uniqueHistory = allHistory.filter((item, index, self) => 
      index === self.findIndex(t => t.id === item.id)
    )
    setHistory(uniqueHistory)
  }

  const handleRouteSelect = (journey) => {
    navigate('/destination', {
      state: {
        currentLocation: journey.from,
        destination: journey.to,
      },
    })
  }

  const handleDelete = (id) => {
    const updatedHistory = history.filter((item) => item.id !== id)
    setHistory(updatedHistory)
    // Update both storage keys
    localStorage.setItem('greengo_journeys', JSON.stringify(updatedHistory))
    localStorage.setItem('greengo_history', JSON.stringify(updatedHistory))
  }

  const formatDate = (dateString) => {
    if (!dateString) return 'Unknown date'
    const date = new Date(dateString)
    if (isNaN(date.getTime())) return 'Unknown date'
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  return (
    <div className="history-page">
      <div className="history-header">
        <h1>GreenGo</h1>
        <p className="header-subtitle">Journey History</p>
      </div>

      <div className="history-content">
        {history.length === 0 ? (
          <div className="empty-state">
            <Clock size={64} className="empty-icon" />
            <h2>No Journey History</h2>
            <p>Your completed journeys will appear here</p>
            <button
              className="start-journey-button"
              onClick={() => navigate('/destination')}
            >
              Start Your First Journey
            </button>
          </div>
        ) : (
          <div className="history-list">
            {history.map((journey) => (
              <div key={journey.id} className="history-card">
                <div className="history-card-header">
                  <div className="history-date">
                    <Clock size={16} />
                    <span>{formatDate(journey.date || journey.timestamp)}</span>
                  </div>
                  <button
                    className="delete-button"
                    onClick={() => handleDelete(journey.id)}
                  >
                    <Trash2 size={18} />
                  </button>
                </div>
                <div className="history-route">
                  <div className="route-point">
                    <MapPin size={18} className="route-icon" />
                    <span>{journey.from}</span>
                  </div>
                  <ArrowRight size={20} className="route-arrow" />
                  <div className="route-point">
                    <MapPin size={18} className="route-icon destination" />
                    <span>{journey.to}</span>
                  </div>
                </div>
                <div className="history-stats">
                  <div className="stat-item">
                    <span className="stat-label">Distance</span>
                    <span className="stat-value">{journey.distance}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Duration</span>
                    <span className="stat-value">{journey.duration}</span>
                  </div>
                </div>
                <button
                  className="use-route-button"
                  onClick={() => handleRouteSelect(journey)}
                >
                  Use This Route Again
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      <NavigationBar />
    </div>
  )
}

export default HistoryPage

