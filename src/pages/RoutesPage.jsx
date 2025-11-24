import React, { useState, useEffect } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { MapContainer, TileLayer, Polyline, useMap } from 'react-leaflet'
import L from 'leaflet'
import { Clock, MapPin, Zap, Car, Circle, Leaf, CheckCircle } from 'lucide-react'
import NavigationBar from '../components/NavigationBar'
import { routeService } from '../services/routeService'
import './RoutesPage.css'
import 'leaflet/dist/leaflet.css'

// Fix for default marker icons
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
})

function MapViewer({ route }) {
  const map = useMap()
  
  useEffect(() => {
    if (route && route.waypoints && route.waypoints.length > 0) {
      const allPoints = [
        [route.start[0], route.start[1]],
        ...route.waypoints.map(wp => [wp[0], wp[1]]),
        [route.end[0], route.end[1]]
      ]
      const bounds = L.latLngBounds(allPoints)
      map.fitBounds(bounds, { padding: [50, 50] })
    }
  }, [route, map])
  
  return null
}

const RoutesPage = () => {
  const location = useLocation()
  const navigate = useNavigate()
  const [routes, setRoutes] = useState([])
  const [selectedRoute, setSelectedRoute] = useState(null)
  const [mapCenter, setMapCenter] = useState([0.3476, 32.5825]) // Kampala

  useEffect(() => {
    try {
      // Parse locations (in production, use geocoding API)
      const start = [0.3476, 32.5825] // Default Kampala
      const end = [0.3650, 32.6100] // Default destination
      
      // Generate route options
      const routeOptions = routeService.generateRouteOptions(start, end)
      
      // Add start and end points to each route
      const routesWithPoints = routeOptions.map(route => ({
        ...route,
        start,
        end,
        fullPath: [
          start,
          ...route.waypoints,
          end
        ],
        trafficLightsArray: routeService.generateTrafficLights(route),
        // Keep the original trafficLights count
        trafficLights: route.trafficLights,
      }))
      
      setRoutes(routesWithPoints)
      setSelectedRoute(routesWithPoints[0])
      setMapCenter(start)
    } catch (error) {
      console.error('Error initializing routes:', error)
      // Fallback to default routes
      const start = [0.3476, 32.5825]
      const end = [0.3650, 32.6100]
      const routeOptions = routeService.generateRouteOptions(start, end)
      const routesWithPoints = routeOptions.map(route => ({
        ...route,
        start,
        end,
        fullPath: [start, ...route.waypoints, end],
        trafficLightsArray: routeService.generateTrafficLights(route),
        // Keep the original trafficLights count
        trafficLights: route.trafficLights,
      }))
      setRoutes(routesWithPoints)
      setSelectedRoute(routesWithPoints[0])
    }
  }, [location.state])

  const handleRouteSelect = (route) => {
    setSelectedRoute(route)
  }

  const handleStartJourney = () => {
    if (selectedRoute) {
      navigate('/drive', {
        state: {
          route: selectedRoute,
          currentLocation: location.state?.currentLocation || 'Kampala',
          destination: location.state?.destination || 'Destination',
        },
      })
    }
  }

  if (routes.length === 0) {
    return (
      <div className="routes-page">
        <div className="loading-state">Loading routes...</div>
        <NavigationBar />
      </div>
    )
  }

  return (
    <div className="routes-page">
      <div className="routes-header">
        <h1>Choose Your Route</h1>
        <p>{location.state?.destination || 'Destination'}</p>
      </div>

      <div className="routes-content">
        <div className="routes-list">
          {routes.map((route) => (
            <div
              key={route.id}
              className={`route-card ${selectedRoute?.id === route.id ? 'selected' : ''}`}
              onClick={() => handleRouteSelect(route)}
            >
              <div className="route-header">
                <div className="route-indicator" style={{ backgroundColor: route.color }}></div>
                <div className="route-info-header">
                  <h3>{route.name}</h3>
                  {selectedRoute?.id === route.id && (
                    <CheckCircle size={20} className="selected-icon" />
                  )}
                </div>
              </div>

              <div className="route-stats">
                <div className="stat">
                  <Clock size={18} />
                  <div>
                    <div className="stat-value">{route.duration} min</div>
                    <div className="stat-label">Duration</div>
                  </div>
                </div>
                <div className="stat">
                  <MapPin size={18} />
                  <div>
                    <div className="stat-value">{route.distance} km</div>
                    <div className="stat-label">Distance</div>
                  </div>
                </div>
                <div className="stat">
                  <Circle size={18} />
                  <div>
                    <div className="stat-value">{route.trafficLights}</div>
                    <div className="stat-label">Traffic Lights</div>
                  </div>
                </div>
              </div>

              <div className="route-details">
                <div className="detail-row">
                  <Car size={16} />
                  <span>Vehicle Density: <strong>{route.vehicleDensity}</strong></span>
                </div>
                <div className="detail-row">
                  <Leaf size={16} />
                  <span>Fuel Efficiency: <strong>{route.fuelEfficiency}%</strong></span>
                </div>
                <div className="detail-row">
                  <Zap size={16} />
                  <span>Road Condition: <strong>{route.roadCondition}</strong></span>
                </div>
              </div>

              <div className="route-preview-map">
                <MapContainer
                  center={mapCenter}
                  zoom={13}
                  style={{ height: '150px', width: '100%' }}
                  zoomControl={false}
                  dragging={false}
                  touchZoom={false}
                  doubleClickZoom={false}
                  scrollWheelZoom={false}
                >
                  <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  />
                  <MapViewer route={route} />
                  <Polyline positions={route.fullPath} color={route.color} weight={4} />
                </MapContainer>
              </div>
            </div>
          ))}
        </div>

        <div className="routes-sidebar">
          <div className="map-container-large">
            <MapContainer
              center={selectedRoute ? [
                (selectedRoute.start[0] + selectedRoute.end[0]) / 2,
                (selectedRoute.start[1] + selectedRoute.end[1]) / 2
              ] : mapCenter}
              zoom={13}
              style={{ height: '100%', width: '100%' }}
            >
              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
              {selectedRoute && (
                <>
                  <MapViewer route={selectedRoute} />
                  <Polyline
                    positions={selectedRoute.fullPath}
                    color={selectedRoute.color}
                    weight={6}
                  />
                </>
              )}
            </MapContainer>
          </div>

          <div className="selected-route-summary">
            {selectedRoute && (
              <>
                <h3>Route Summary</h3>
                <div className="summary-grid">
                  <div className="summary-item">
                    <div className="summary-label">Distance</div>
                    <div className="summary-value">{selectedRoute.distance} km</div>
                  </div>
                  <div className="summary-item">
                    <div className="summary-label">Duration</div>
                    <div className="summary-value">{selectedRoute.duration} min</div>
                  </div>
                  <div className="summary-item">
                    <div className="summary-label">Traffic Lights</div>
                    <div className="summary-value">{selectedRoute.trafficLights}</div>
                  </div>
                  <div className="summary-item">
                    <div className="summary-label">Fuel Efficiency</div>
                    <div className="summary-value">{selectedRoute.fuelEfficiency}%</div>
                  </div>
                </div>
                <button className="start-route-button" onClick={handleStartJourney}>
                  Start This Route
                </button>
              </>
            )}
          </div>
        </div>
      </div>

      <NavigationBar />
    </div>
  )
}

export default RoutesPage

