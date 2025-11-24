// Route calculation and simulation service
export const routeService = {
  // Generate multiple route options
  generateRouteOptions: (start, end) => {
    // In production, this would call a real routing API (OSRM, Google Maps, etc.)
    // For now, we'll simulate multiple routes with different characteristics
    
    const baseLat = start[0]
    const baseLng = start[1]
    const endLat = end[0]
    const endLng = end[1]
    
    const latDiff = endLat - baseLat
    const lngDiff = endLng - baseLng
    
    // Route 1: Fastest (more traffic lights, but shorter)
    const route1 = {
      id: 1,
      name: 'Fastest Route',
      waypoints: [
        [baseLat + latDiff * 0.25, baseLng + lngDiff * 0.25],
        [baseLat + latDiff * 0.5, baseLng + lngDiff * 0.5],
        [baseLat + latDiff * 0.75, baseLng + lngDiff * 0.75],
      ],
      distance: 8.2,
      duration: 15,
      trafficLights: 6,
      vehicleDensity: 'High',
      roadCondition: 'Good',
      fuelEfficiency: 85,
      color: '#22c55e',
    }
    
    // Route 2: Balanced (moderate traffic lights, good balance)
    const route2 = {
      id: 2,
      name: 'Balanced Route',
      waypoints: [
        [baseLat + latDiff * 0.2, baseLng + lngDiff * 0.3],
        [baseLat + latDiff * 0.45, baseLng + lngDiff * 0.55],
        [baseLat + latDiff * 0.7, baseLng + lngDiff * 0.8],
      ],
      distance: 9.1,
      duration: 18,
      trafficLights: 4,
      vehicleDensity: 'Medium',
      roadCondition: 'Excellent',
      fuelEfficiency: 92,
      color: '#3b82f6',
    }
    
    // Route 3: Eco-Friendly (fewer traffic lights, better fuel efficiency)
    const route3 = {
      id: 3,
      name: 'Eco-Friendly Route',
      waypoints: [
        [baseLat + latDiff * 0.15, baseLng + lngDiff * 0.2],
        [baseLat + latDiff * 0.4, baseLng + lngDiff * 0.5],
        [baseLat + latDiff * 0.65, baseLng + lngDiff * 0.75],
      ],
      distance: 10.5,
      duration: 22,
      trafficLights: 3,
      vehicleDensity: 'Low',
      roadCondition: 'Good',
      fuelEfficiency: 95,
      color: '#10b981',
    }
    
    return [route1, route2, route3]
  },

  // Generate traffic lights along a route
  generateTrafficLights: (route) => {
    const lights = []
    route.waypoints.forEach((waypoint, index) => {
      // Add traffic lights at waypoints
      lights.push({
        id: `light-${route.id}-${index}`,
        position: waypoint,
        status: ['Green', 'Yellow', 'Red'][index % 3],
        countdown: [30, 15, 45][index % 3],
        name: `Traffic Light ${index + 1}`,
      })
    })
    return lights
  },

  // Calculate distance between two points (Haversine formula)
  calculateDistance: (point1, point2) => {
    const R = 6371e3 // Earth radius in meters
    const φ1 = point1[0] * Math.PI / 180
    const φ2 = point2[0] * Math.PI / 180
    const Δφ = (point2[0] - point1[0]) * Math.PI / 180
    const Δλ = (point2[1] - point1[1]) * Math.PI / 180

    const a = Math.sin(Δφ / 2) * Math.sin(Δφ / 2) +
      Math.cos(φ1) * Math.cos(φ2) *
      Math.sin(Δλ / 2) * Math.sin(Δλ / 2)
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))

    return R * c // Distance in meters
  },
}

