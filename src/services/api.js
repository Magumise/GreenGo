import axios from 'axios'
import { buildFeatureVector } from './featureBuilder'

const DIRECT_API_URL = 'https://greengo-api-915779460150.us-east1.run.app'
const DEFAULT_PROXY = 'https://cors.isomorphic-git.org/'

const resolveApiBase = () => {
  if (import.meta.env.DEV) {
    return '/api'
  }

  const envUrl = (import.meta.env.VITE_API_BASE_URL || '').trim()
  if (envUrl) {
    return envUrl
  }

  // Fallback for production deployments where the API blocks browser origins
  return `${DEFAULT_PROXY}${DIRECT_API_URL}`
}

const API_BASE_URL = resolveApiBase()

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
})

export const predictTrafficLight = async (data) => {
  try {
    const featureVector = buildFeatureVector(data)
    const payload = {
      instances: [featureVector],
    }
    
    console.log('📤 Sending to API (instances format):', payload)
    
    const response = await api.post('/predict', payload, {
      headers: {
        'Content-Type': 'application/json',
      },
    })
    
    console.log('📥 Raw API Response:', response.data)
    
    // Handle API response - it returns a prediction value
    // Could be: {"predictions": [value]} or just a number or object
    let predictionValue = null
    
    if (response.data) {
      if (response.data.predictions && Array.isArray(response.data.predictions)) {
        const firstPrediction = response.data.predictions[0]
        predictionValue = Array.isArray(firstPrediction) ? firstPrediction[0] : firstPrediction
      } else if (typeof response.data === 'number') {
        predictionValue = response.data
      } else if (response.data.prediction !== undefined) {
        predictionValue = response.data.prediction
      } else if (response.data.recommended_speed !== undefined) {
        predictionValue = response.data.recommended_speed
      } else {
        predictionValue = response.data
      }
    }
    
    // The API returns the recommended speed prediction
    // Convert to our expected format
    const result = {
      recommended_speed: typeof predictionValue === 'number'
        ? Math.max(0, Math.min(Number(data.speed_limit_kmh ?? 50), predictionValue))
        : Number(data.lead_vehicle_speed_kmh ?? 45),
      next_light_change_s: Number(data.eta_to_light_s ?? 30),
      next_light_status: String(data.current_light ?? 'Green'),
      confidence: 0.8,
      raw_prediction: predictionValue,
      is_from_api: true,
    }
    
    console.log('✅ Processed API Response:', result)
    return result
    
  } catch (error) {
    console.error('❌ API Error:', error)
    
    // Log detailed error information
    if (error.response) {
      console.error('Response status:', error.response.status)
      console.error('Response data:', error.response.data)
    }
    if (error.request) {
      console.error('Request made but no response:', error.request)
    }
    
    // Return fallback with same structure
    return {
      recommended_speed: Number(data.lead_vehicle_speed_kmh || 45),
      next_light_change_s: Number(data.eta_to_light_s || 30),
      next_light_status: String(data.current_light || 'Green'),
      confidence: 0.5,
      is_fallback: true,
    }
  }
}

export const getHealth = async () => {
  try {
    const response = await api.get('/health')
    return response.data
  } catch (error) {
    console.error('Health check error:', error)
    throw error
  }
}

export default api

