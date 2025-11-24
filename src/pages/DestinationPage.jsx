import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { MapPin, Navigation, Search, Star, Clock, TrendingUp } from 'lucide-react'
import NavigationBar from '../components/NavigationBar'
import { storage } from '../utils/storage'
import './DestinationPage.css'

const DestinationPage = () => {
  const navigate = useNavigate()
  const [currentLocation, setCurrentLocation] = useState('')
  const [destination, setDestination] = useState('')
  const [isDetecting, setIsDetecting] = useState(false)
  const [favorites] = useState(storage.getFavoriteDestinations())
  const [recentSearches] = useState(storage.getRecentSearches())
  const [suggestions, setSuggestions] = useState([])

  useEffect(() => {
    // Try to get current location
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords
          setCurrentLocation(`Current Location (${latitude.toFixed(4)}, ${longitude.toFixed(4)})`)
        },
        () => {
          setCurrentLocation('Kampala, Uganda')
        }
      )
    } else {
      setCurrentLocation('Kampala, Uganda')
    }
  }, [])

  const handleDetectLocation = () => {
    setIsDetecting(true)
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords
          setCurrentLocation(`Current Location (${latitude.toFixed(4)}, ${longitude.toFixed(4)})`)
          setIsDetecting(false)
        },
        (error) => {
          console.error('Geolocation error:', error)
          setIsDetecting(false)
        }
      )
    } else {
      setIsDetecting(false)
    }
  }

  const handleDestinationChange = (value) => {
    setDestination(value)
    // In production, this would call a geocoding API
    if (value.length > 2) {
      // Simulate suggestions
      const mockSuggestions = [
        { name: `${value} - Kampala`, type: 'location' },
        { name: `${value} Mall`, type: 'shopping' },
        { name: `${value} Road`, type: 'street' },
      ]
      setSuggestions(mockSuggestions.slice(0, 3))
    } else {
      setSuggestions([])
    }
  }

  const handleStartJourney = () => {
    if (destination.trim() && currentLocation.trim()) {
      // Save to recent searches
      storage.saveRecentSearch({
        query: destination,
        location: currentLocation,
      })
      
      navigate('/routes', {
        state: {
          currentLocation,
          destination,
        },
      })
    }
  }

  const handleSuggestionClick = (suggestion) => {
    setDestination(suggestion.name)
    setSuggestions([])
  }

  const handleFavoriteClick = (fav) => {
    setDestination(fav.name)
  }

  const handleRecentClick = (recent) => {
    setDestination(recent.query)
  }

  return (
    <div className="destination-page-new">
      <div className="destination-header-new">
        <h1>Plan Your Route</h1>
        <p>Enter your destination to find the best route with traffic light predictions</p>
      </div>

      <div className="destination-content-new">
        <div className="input-card-new">
          <div className="input-group">
            <label className="input-label-new">
              <Navigation size={18} />
              Current Location
            </label>
            <div className="input-wrapper-new">
              <input
                type="text"
                className="location-input-new"
                value={currentLocation}
                onChange={(e) => setCurrentLocation(e.target.value)}
                placeholder="Enter your current location"
              />
              <button
                className="detect-button-new"
                onClick={handleDetectLocation}
                disabled={isDetecting}
              >
                <MapPin size={16} />
                {isDetecting ? 'Detecting...' : 'Detect'}
              </button>
            </div>
          </div>

          <div className="input-group">
            <label className="input-label-new">
              <Search size={18} />
              Destination
            </label>
            <div className="input-wrapper-new destination-wrapper">
              <input
                type="text"
                className="destination-input-new"
                value={destination}
                onChange={(e) => handleDestinationChange(e.target.value)}
                placeholder="Where do you want to go?"
                onKeyPress={(e) => e.key === 'Enter' && handleStartJourney()}
              />
              {suggestions.length > 0 && (
                <div className="suggestions-dropdown">
                  {suggestions.map((suggestion, index) => (
                    <div
                      key={index}
                      className="suggestion-item"
                      onClick={() => handleSuggestionClick(suggestion)}
                    >
                      <MapPin size={16} />
                      <span>{suggestion.name}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <button
            className="start-journey-button-new"
            onClick={handleStartJourney}
            disabled={!destination.trim() || !currentLocation.trim()}
          >
            Find Routes
          </button>
        </div>

        {favorites.length > 0 && (
          <div className="section-new">
            <div className="section-header">
              <Star size={20} className="section-icon" />
              <h3>Favorite Destinations</h3>
            </div>
            <div className="favorites-grid">
              {favorites.map((fav, index) => (
                <div
                  key={index}
                  className="favorite-card"
                  onClick={() => handleFavoriteClick(fav)}
                >
                  <Star size={20} className="star-icon" />
                  <div className="favorite-info">
                    <div className="favorite-name">{fav.name}</div>
                    <div className="favorite-location">{fav.location}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {recentSearches.length > 0 && (
          <div className="section-new">
            <div className="section-header">
              <Clock size={20} className="section-icon" />
              <h3>Recent Searches</h3>
            </div>
            <div className="recent-list-new">
              {recentSearches.slice(0, 5).map((recent, index) => (
                <div
                  key={index}
                  className="recent-item-new"
                  onClick={() => handleRecentClick(recent)}
                >
                  <Clock size={18} />
                  <div className="recent-info-new">
                    <div className="recent-query">{recent.query}</div>
                    <div className="recent-location">{recent.location}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="section-new">
          <div className="section-header">
            <TrendingUp size={20} className="section-icon" />
            <h3>Quick Tips</h3>
          </div>
          <div className="tips-grid">
            <div className="tip-card">
              <div className="tip-icon">🚦</div>
              <div className="tip-content">
                <h4>Smart Routing</h4>
                <p>Choose routes with optimal traffic light timing</p>
              </div>
            </div>
            <div className="tip-card">
              <div className="tip-icon">⚡</div>
              <div className="tip-content">
                <h4>Save Fuel</h4>
                <p>Follow speed recommendations to reduce consumption</p>
              </div>
            </div>
            <div className="tip-card">
              <div className="tip-icon">🌱</div>
              <div className="tip-content">
                <h4>Eco-Friendly</h4>
                <p>Reduce emissions with intelligent routing</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <NavigationBar />
    </div>
  )
}

export default DestinationPage
