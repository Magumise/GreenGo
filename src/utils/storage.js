// Utility for storing and retrieving app data
export const storage = {
  // Store journey data
  saveJourney: (journey) => {
    const journeys = storage.getJourneys()
    journeys.unshift({
      ...journey,
      id: Date.now(),
      timestamp: new Date().toISOString(),
    })
    localStorage.setItem('greengo_journeys', JSON.stringify(journeys.slice(0, 100)))
  },

  getJourneys: () => {
    return JSON.parse(localStorage.getItem('greengo_journeys') || '[]')
  },

  // Store favorite destinations
  saveFavoriteDestination: (destination) => {
    const favorites = storage.getFavoriteDestinations()
    if (!favorites.find(f => f.name === destination.name)) {
      favorites.push(destination)
      localStorage.setItem('greengo_favorites', JSON.stringify(favorites))
    }
  },

  getFavoriteDestinations: () => {
    return JSON.parse(localStorage.getItem('greengo_favorites') || '[]')
  },

  // Store recent searches
  saveRecentSearch: (search) => {
    const recent = storage.getRecentSearches()
    const existing = recent.findIndex(r => r.query === search.query)
    if (existing >= 0) {
      recent.splice(existing, 1)
    }
    recent.unshift({ ...search, timestamp: Date.now() })
    localStorage.setItem('greengo_recent_searches', JSON.stringify(recent.slice(0, 20)))
  },

  getRecentSearches: () => {
    return JSON.parse(localStorage.getItem('greengo_recent_searches') || '[]')
  },

  // Store route preferences
  saveRoutePreference: (preference) => {
    localStorage.setItem('greengo_route_preference', JSON.stringify(preference))
  },

  getRoutePreference: () => {
    return JSON.parse(localStorage.getItem('greengo_route_preference') || '{}')
  },
}

