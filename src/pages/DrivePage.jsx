import React, { useState, useEffect, useRef } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { MapContainer, TileLayer, Marker, Polyline, useMap, Popup } from 'react-leaflet'
import L from 'leaflet'
import { Navigation, Pause, RefreshCw, MapPin, Gauge, Clock } from 'lucide-react'
import NavigationBar from '../components/NavigationBar'
import { predictTrafficLight } from '../services/api'
import { storage } from '../utils/storage'
import { routeService } from '../services/routeService'
import './DrivePage.css'
import 'leaflet/dist/leaflet.css'

// Fix for default marker icons
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
})

// Custom vehicle icon
const vehicleIcon = L.divIcon({
  className: 'vehicle-marker-new',
  html: '<div class="vehicle-icon-new">🚗</div>',
  iconSize: [50, 50],
  iconAnchor: [25, 25],
})

// Custom traffic light icon
const createTrafficLightIcon = (status, countdown) => {
  const colors = {
    Green: '#22c55e',
    Yellow: '#eab308',
    Red: '#ef4444',
  }
  return L.divIcon({
    className: 'traffic-light-marker-new',
    html: `<div class="traffic-light-icon-new" style="background: ${colors[status] || colors.Green}">
      <div class="traffic-light-status-new">${status[0]}</div>
      <div class="traffic-light-countdown">${countdown}s</div>
    </div>`,
    iconSize: [50, 50],
    iconAnchor: [25, 25],
  })
}

// Map updater component
function MapUpdater({ center, zoom }) {
  const map = useMap()
  useEffect(() => {
    if (center && zoom) {
      map.setView(center, zoom)
    }
  }, [center, zoom, map])
  return null
}

const DrivePage = () => {
  const location = useLocation()
  const navigate = useNavigate()
  const [currentPosition, setCurrentPosition] = useState([0.3476, 32.5825])
  const [route, setRoute] = useState(null)
  const [trafficLights, setTrafficLights] = useState([])
  const [isDriving, setIsDriving] = useState(false)
  const [currentSpeed, setCurrentSpeed] = useState(0)
  const [recommendedSpeed, setRecommendedSpeed] = useState(45)
  const [speedLimit, setSpeedLimit] = useState(50)
  const [nextLight, setNextLight] = useState(null)
  const [distanceToNextLight, setDistanceToNextLight] = useState(0)
  const [destinationName, setDestinationName] = useState('')
  const [currentRoadName, setCurrentRoadName] = useState('')
  const [routeInfo, setRouteInfo] = useState(null)
  const [apiPredictions, setApiPredictions] = useState({})
  const [isLoadingPrediction, setIsLoadingPrediction] = useState(false)
  const [isJourneyComplete, setIsJourneyComplete] = useState(false)
  const [apiCallCount, setApiCallCount] = useState(0)
  const [currentAcceleration, setCurrentAcceleration] = useState(0)
  const [environment] = useState({
    weatherCondition: 'Partly cloudy',
    visibilityMeters: 8000,
  })
  const intervalRef = useRef(null)
  const positionIndexRef = useRef(0)
  const lastApiCallRef = useRef(0)
  const prevSpeedRef = useRef(0)

  // Initialize route from state or create default
  useEffect(() => {
    if (location.state?.route) {
      const routeData = location.state.route
      setRoute(routeData)
      setCurrentPosition(routeData.start)
      setTrafficLights(routeData.trafficLightsArray || routeData.trafficLights || [])
      setDestinationName(location.state.destination || 'Destination')
      setRouteInfo({
        distance: routeData.distance,
        duration: routeData.duration,
        trafficLightsCount: routeData.trafficLights,
      })
      // Remember destination name
      if (location.state.destination) {
        storage.saveRecentSearch({
          query: location.state.destination,
          location: location.state.currentLocation || 'Current Location',
        })
      }
    } else {
      initializeDefaultRoute()
    }
  }, [location.state])

  const initializeDefaultRoute = () => {
    const start = [0.3476, 32.5825]
    const end = [0.3650, 32.6100]
    const routeOptions = routeService.generateRouteOptions(start, end)
    const defaultRoute = {
      ...routeOptions[0],
      start,
      end,
      fullPath: [start, ...routeOptions[0].waypoints, end],
      trafficLights: routeService.generateTrafficLights(routeOptions[0]),
    }
      setRoute(defaultRoute)
      setCurrentPosition(start)
      setTrafficLights(defaultRoute.trafficLightsArray || defaultRoute.trafficLights || [])
    setDestinationName('Kampala City Center')
    setRouteInfo({
      distance: defaultRoute.distance,
      duration: defaultRoute.duration,
      trafficLightsCount: defaultRoute.trafficLights.length,
    })
  }

  const startDriving = () => {
    if (!route) return
    setIsDriving(true)
    positionIndexRef.current = 0
    intervalRef.current = setInterval(() => {
      updateDrivingState()
    }, 1000)
  }

  const stopDriving = () => {
    setIsDriving(false)
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
  }

  const updateDrivingState = async () => {
    if (!route || positionIndexRef.current >= route.fullPath.length - 1) {
      stopDriving()
      saveJourneyToHistory()
      setIsJourneyComplete(true)
      return
    }

    // Move vehicle
    positionIndexRef.current += 0.15
    const nextIndex = Math.min(Math.floor(positionIndexRef.current), route.fullPath.length - 1)
    const nextPos = route.fullPath[nextIndex]
    setCurrentPosition(nextPos)

    // Update speed (simulated)
    const newSpeed = Math.max(0, 35 + Math.random() * 15)
    const accelerationValue = newSpeed - prevSpeedRef.current
    prevSpeedRef.current = newSpeed
    const roundedSpeed = Math.round(newSpeed)
    setCurrentSpeed(roundedSpeed)
    setCurrentAcceleration(accelerationValue)

    // Update road name (simulated)
    const roadNames = ['Jinja Road', 'Kampala Road', 'Entebbe Road', 'Gulu Road']
    setCurrentRoadName(roadNames[Math.floor(Math.random() * roadNames.length)])

    // Find nearest traffic light
    const nearestLight = findNearestTrafficLight(nextPos)
    if (nearestLight) {
      await updateTrafficLightPrediction(nearestLight, nextPos, roundedSpeed, accelerationValue)
    }
  }

  const findNearestTrafficLight = (position) => {
    if (!position || !Array.isArray(position) || position.length < 2) {
      return null
    }
    
    if (!trafficLights || trafficLights.length === 0) {
      return null
    }

    let nearest = null
    let minDistance = Infinity

    trafficLights.forEach((light) => {
      if (!light || !light.position || !Array.isArray(light.position)) {
        return
      }
      
      try {
        const distance = routeService.calculateDistance(position, light.position)
        if (distance < minDistance && distance < 1000) {
          minDistance = distance
          nearest = { ...light, distance }
        }
      } catch (error) {
        console.warn('Error calculating distance to light:', error)
      }
    })

    return nearest
  }

  const updateTrafficLightPrediction = async (light, currentPos, currentSpeed, acceleration = 0) => {
    const distance = light.distance || routeService.calculateDistance(currentPos, light.position)
    const speedMs = Math.max(0.1, currentSpeed * 1000 / 3600)
    const eta = distance / speedMs

    setDistanceToNextLight(Math.round(distance))
    setNextLight(light)

    // Call API every 3 seconds to avoid too many requests
    const now = Date.now()
    if (now - lastApiCallRef.current > 3000 && !isLoadingPrediction) {
      setIsLoadingPrediction(true)
      lastApiCallRef.current = now

      try {
        // Prepare payload with the context the API expects
        const lat = currentPos?.[0] ?? currentPosition[0]
        const lon = currentPos?.[1] ?? currentPosition[1]
        const intersectionId = Number(String(light.id).replace(/\D/g, '')) || route?.id || 0

        const apiPayload = {
          timestamp: new Date().toISOString(), // ISO string
          current_light: String(light.status || 'Green'), // String: "Green", "Yellow", or "Red"
          eta_to_light_s: Number(Math.max(0, eta)), // Float
          distance_to_next_light_m: Number(distance), // Float
          vehicle_count: Number(route?.vehicleDensity === 'High' ? 8 : route?.vehicleDensity === 'Medium' ? 5 : 3), // Integer
          pedestrian_count: Number(Math.floor(Math.random() * 5) + 1), // Integer
          lead_vehicle_speed_kmh: Number(currentSpeed), // Float
          speed_limit_kmh: Number(speedLimit), // Float
          current_speed_kmh: Number(currentSpeed),
          acceleration: Number(acceleration),
          latitude: Number(lat),
          longitude: Number(lon),
          position: currentPos,
          weather_condition: environment.weatherCondition,
          visibility_meters: environment.visibilityMeters,
          road_condition: route?.roadCondition,
          intersection_id: intersectionId,
          route_id: route?.id ?? 0,
        }
        
        // Validate all required fields are present
        const requiredFields = [
          'timestamp', 'current_light', 'eta_to_light_s', 'distance_to_next_light_m',
          'vehicle_count', 'pedestrian_count', 'lead_vehicle_speed_kmh', 'speed_limit_kmh'
        ]
        const missingFields = requiredFields.filter(field => apiPayload[field] === undefined || apiPayload[field] === null)
        
        if (missingFields.length > 0) {
          console.error('❌ Missing required fields:', missingFields)
          throw new Error(`Missing required fields: ${missingFields.join(', ')}`)
        }
        
        console.log('🚀 Calling API with payload:', apiPayload)
        setApiCallCount(prev => prev + 1)
        
        const prediction = await predictTrafficLight(apiPayload)
        
        console.log('✅ API Response received:', prediction)

        // Update with real API response
        if (prediction.recommended_speed !== undefined) {
          setRecommendedSpeed(Math.max(0, Math.min(speedLimit, Math.round(prediction.recommended_speed))))
        }
        if (prediction.next_light_change_s !== undefined) {
          const newCountdown = Math.max(0, Math.round(prediction.next_light_change_s))
          setTrafficLights(prev => prev.map(l => 
            l.id === light.id ? { ...l, countdown: newCountdown } : l
          ))
          if (light.id === nextLight?.id) {
            setNextLight({ ...light, countdown: newCountdown })
          }
        }
        if (prediction.next_light_status) {
          setTrafficLights(prev => prev.map(l => 
            l.id === light.id ? { ...l, status: prediction.next_light_status } : l
          ))
          if (light.id === nextLight?.id) {
            setNextLight({ ...light, status: prediction.next_light_status })
          }
        }

        setApiPredictions(prev => ({
          ...prev,
          [light.id]: prediction,
        }))
      } catch (error) {
        console.error('API prediction error:', error)
        // Fallback: use simulated countdown
        if (light && light.id) {
          const simulatedCountdown = Math.max(0, Math.round(eta - 5))
          setTrafficLights(prev => prev.map(l => 
            l.id === light.id ? { ...l, countdown: simulatedCountdown } : l
          ))
        }
      } finally {
        setIsLoadingPrediction(false)
      }
    } else {
      // Update countdown locally (when not calling API)
      if (light && light.id) {
        setTrafficLights(prev => prev.map(l => {
          if (l.id === light.id && l.countdown > 0) {
            return { ...l, countdown: Math.max(0, l.countdown - 1) }
          }
          return l
        }))
        if (nextLight?.id === light.id && nextLight.countdown > 0) {
          setNextLight({ ...light, countdown: Math.max(0, nextLight.countdown - 1) })
        }
      }
    }
  }

  const saveJourneyToHistory = () => {
    if (route && destinationName) {
      storage.saveJourney({
        from: location.state?.currentLocation || 'Current Location',
        to: destinationName,
        distance: `${route.distance} km`,
        duration: `${route.duration} min`,
        route: route,
      })
    }
  }

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [])

  if (!route) {
    return (
      <div className="drive-page-new">
        <div className="loading-state">Loading route...</div>
        <NavigationBar />
      </div>
    )
  }

  const mapCenter = currentPosition

  return (
    <div className="drive-page-new">
      <div className="drive-header-new">
        <div className="header-content">
          <div>
            <h2>{destinationName}</h2>
            <p className="road-name">{currentRoadName || 'Calculating route...'}</p>
          </div>
          <div className="header-stats">
            <div className="header-stat">
              <span className="stat-label">Distance</span>
              <span className="stat-value">{routeInfo?.distance} km</span>
            </div>
            <div className="header-stat">
              <span className="stat-label">ETA</span>
              <span className="stat-value">{routeInfo?.duration} min</span>
            </div>
          </div>
        </div>
      </div>

      <div className="drive-main">
        <div className="map-section-large">
          <MapContainer
            center={mapCenter}
            zoom={14}
            style={{ height: '100%', width: '100%' }}
            zoomControl={true}
          >
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            <MapUpdater center={mapCenter} zoom={14} />
            <Polyline positions={route.fullPath} color="#22c55e" weight={6} />
            <Marker position={currentPosition} icon={vehicleIcon}>
              <Popup>
                <div>
                  <strong>Your Vehicle</strong><br />
                  Speed: {currentSpeed} km/h<br />
                  Road: {currentRoadName}
                </div>
              </Popup>
            </Marker>
            {trafficLights.map((light) => (
              <Marker
                key={light.id}
                position={light.position}
                icon={createTrafficLightIcon(light.status, light.countdown)}
              >
                <Popup>
                  <div>
                    <strong>{light.name || 'Traffic Light'}</strong><br />
                    Status: {light.status}<br />
                    Countdown: {light.countdown}s
                  </div>
                </Popup>
              </Marker>
            ))}
          </MapContainer>
        </div>

        <div className="drive-sidebar">
          <div className="speed-panel">
            <div className="speed-display">
              <div className="speed-main">
                <Gauge size={32} />
                <div>
                  <div className="speed-value">{recommendedSpeed}</div>
                  <div className="speed-label">Recommended Speed (km/h)</div>
                </div>
              </div>
              <div className="speed-limit">
                <div className="limit-value">{speedLimit}</div>
                <div className="limit-label">Speed Limit</div>
              </div>
            </div>
            <div className="current-speed">
              <span>Current: {currentSpeed} km/h</span>
            </div>
          </div>

          {nextLight && (
            <div className="next-light-panel">
              <h3>Next Traffic Light</h3>
              <div className={`light-status-badge ${nextLight.status.toLowerCase()}`}>
                <div className="light-status-text">{nextLight.status}</div>
                <div className="light-countdown">{nextLight.countdown}s</div>
              </div>
              <div className="light-details">
                <div className="light-detail-item">
                  <MapPin size={16} />
                  <span>{distanceToNextLight} m away</span>
                </div>
                <div className="light-detail-item">
                  <Clock size={16} />
                  <span>ETA: {Math.round(distanceToNextLight / (currentSpeed * 1000 / 3600))}s</span>
                </div>
              </div>
            </div>
          )}

          <div className="route-info-panel">
            <h3>Route Information</h3>
            <div className="info-item">
              <span className="info-label">Destination:</span>
              <span className="info-value">{destinationName}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Distance:</span>
              <span className="info-value">{routeInfo?.distance} km</span>
            </div>
            <div className="info-item">
              <span className="info-label">Traffic Lights:</span>
              <span className="info-value">{routeInfo?.trafficLightsCount} ahead</span>
            </div>
            <div className="info-item">
              <span className="info-label">Road Condition:</span>
              <span className="info-value">{route.roadCondition}</span>
            </div>
          </div>

          <button
            className="update-prediction-button"
            onClick={() => {
              const nearest = findNearestTrafficLight(currentPosition)
              if (nearest) {
                updateTrafficLightPrediction(nearest, currentPosition, currentSpeed, currentAcceleration)
              }
            }}
            disabled={isLoadingPrediction}
          >
            <RefreshCw size={18} className={isLoadingPrediction ? 'spinning' : ''} />
            {isLoadingPrediction ? 'Updating...' : 'Update Prediction'}
          </button>
        </div>
      </div>

      <div className="drive-controls">
        {!isDriving && !isJourneyComplete ? (
          <button className="start-button-new" onClick={startDriving}>
            <Navigation size={20} />
            Start Journey
          </button>
        ) : isDriving && !isJourneyComplete ? (
          <button className="stop-button-new" onClick={stopDriving}>
            <Pause size={20} />
            Pause Journey
          </button>
        ) : null}
      </div>

      {isJourneyComplete && (
        <div className="journey-complete-overlay">
          <div className="journey-complete-card">
            <div className="complete-icon">✓</div>
            <h2>Journey Complete!</h2>
            <p>Your journey has been saved to history</p>
            <div className="journey-stats-summary">
              <div className="summary-stat">
                <span className="summary-label">API Calls Made</span>
                <span className="summary-value">{apiCallCount}</span>
              </div>
              <div className="summary-stat">
                <span className="summary-label">Distance</span>
                <span className="summary-value">{routeInfo?.distance}</span>
              </div>
            </div>
            <button 
              className="new-journey-button" 
              onClick={() => navigate('/destination')}
            >
              <Navigation size={20} />
              Start Another Journey
            </button>
          </div>
        </div>
      )}

      <NavigationBar />
    </div>
  )
}

export default DrivePage
