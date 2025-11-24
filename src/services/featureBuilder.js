const FEATURE_NAMES = [
  'vehicle_count',
  'pedestrian_count',
  'day_of_week',
  'latitude',
  'longitude',
  'acceleration',
  'visibility_meters',
  'road_slipperiness',
  'distance_to_next_light_m',
  'eta_to_light_s',
  'lead_vehicle_speed_kmh',
  'headway_seconds',
  'hour',
  'minute',
  'hour_sin',
  'hour_cos',
  'minute_sin',
  'minute_cos',
  'morning_rush',
  'evening_rush',
  'weekend',
  'light_timing_code',
  'is_green',
  'is_red',
  'required_speed_ms',
  'timing_urgency',
  'eta_squared',
  'distance_squared',
  'eta_distance_interaction',
  'total_traffic',
  'traffic_density',
  'rush_traffic',
  'speed_ratio',
  'speed_deficit',
  'eta_rolling_mean_3',
  'eta_rolling_mean_5',
  'hour_traffic',
  'rush_urgency',
  'current_light_Red',
  'current_light_Yellow',
  'weather_condition_Rainy',
  'weather_condition_Snowy',
  'weather_condition_Sunny',
  'time_of_day_Evening',
  'time_of_day_Morning',
  'time_of_day_Night',
  'visibility_category_Low',
  'visibility_category_Moderate',
  'slipperiness_category_High',
  'slipperiness_category_Moderate',
  'slipperiness_category_Slight',
  'predicted_state_at_arrival_Red',
  'predicted_state_at_arrival_Yellow',
  'congestion_level_Moderate',
]

const etaHistory = []

const clampNumber = (value, fallback = 0) => {
  const num = Number(value)
  return Number.isFinite(num) ? num : fallback
}

const toRadians = (deg) => (deg * Math.PI) / 180

const getRollingAverage = (history, value, windowSize) => {
  history.push(value)
  if (history.length > windowSize * 2) {
    history.splice(0, history.length - windowSize * 2)
  }
  const slice = history.slice(-windowSize)
  if (!slice.length) return value
  const sum = slice.reduce((acc, curr) => acc + curr, 0)
  return sum / slice.length
}

const mapRoadConditionToSlipperiness = (condition) => {
  if (!condition) return 0.2
  const normalized = String(condition).toLowerCase()
  if (normalized.includes('wet') || normalized.includes('snow') || normalized.includes('ice')) {
    return 0.8
  }
  if (normalized.includes('moderate') || normalized.includes('gravel')) {
    return 0.5
  }
  return 0.2
}

const mapWeatherVisibility = (weatherCondition, explicitVisibility) => {
  const base = clampNumber(explicitVisibility, NaN)
  if (Number.isFinite(base)) return base
  const normalized = String(weatherCondition || '').toLowerCase()
  if (normalized.includes('fog') || normalized.includes('storm')) return 500
  if (normalized.includes('rain')) return 2000
  if (normalized.includes('snow')) return 1500
  return 8000
}

const getTimeOfDayFlags = (hour) => {
  return {
    morning: hour >= 5 && hour < 11 ? 1 : 0,
    evening: hour >= 17 && hour < 21 ? 1 : 0,
    night: hour >= 21 || hour < 5 ? 1 : 0,
  }
}

const getWeatherFlags = (condition) => {
  const normalized = String(condition || '').toLowerCase()
  return {
    rainy: normalized.includes('rain') ? 1 : 0,
    snowy: normalized.includes('snow') ? 1 : 0,
    sunny: normalized.includes('sun') || normalized.includes('clear') ? 1 : 0,
  }
}

const getVisibilityFlags = (visibility) => {
  return {
    low: visibility < 1000 ? 1 : 0,
    moderate: visibility >= 1000 && visibility < 4000 ? 1 : 0,
  }
}

const getSlipperinessFlags = (slipperiness) => {
  return {
    high: slipperiness >= 0.7 ? 1 : 0,
    moderate: slipperiness >= 0.4 && slipperiness < 0.7 ? 1 : 0,
    slight: slipperiness < 0.4 ? 1 : 0,
  }
}

const getPredictedStateFlags = (currentLight, etaSeconds) => {
  let red = 0
  let yellow = 0

  if (currentLight === 'Green') {
    yellow = etaSeconds > 30 ? 1 : 0
  } else if (currentLight === 'Yellow') {
    red = 1
  } else if (currentLight === 'Red') {
    red = etaSeconds % 20 < 5 ? 0 : 1
  }

  return { red, yellow }
}

const getTimeOfDay = (hour) => {
  if (hour >= 5 && hour < 12) return { morning: 1, evening: 0, night: 0 }
  if (hour >= 17 && hour < 21) return { morning: 0, evening: 1, night: 0 }
  if (hour >= 21 || hour < 5) return { morning: 0, evening: 0, night: 1 }
  return { morning: 0, evening: 0, night: 0 }
}

export const buildFeatureVector = (input = {}) => {
  const timestamp = input.timestamp ? new Date(input.timestamp) : new Date()
  const dayOfWeek = timestamp.getDay()
  const hour = timestamp.getHours()
  const minute = timestamp.getMinutes()
  const hourSin = Math.sin((2 * Math.PI * hour) / 24)
  const hourCos = Math.cos((2 * Math.PI * hour) / 24)
  const minuteSin = Math.sin((2 * Math.PI * minute) / 60)
  const minuteCos = Math.cos((2 * Math.PI * minute) / 60)

  const latitude = clampNumber(input.latitude ?? input.position?.[0])
  const longitude = clampNumber(input.longitude ?? input.position?.[1])

  const vehicleCount = clampNumber(input.vehicle_count, 0)
  const pedestrianCount = clampNumber(input.pedestrian_count, 0)
  const totalTraffic = vehicleCount + pedestrianCount

  const eta = Math.max(0, clampNumber(input.eta_to_light_s, 0))
  const distance = Math.max(0, clampNumber(input.distance_to_next_light_m, 0))
  const leadSpeed = clampNumber(input.lead_vehicle_speed_kmh, 0)
  const speedLimit = Math.max(1, clampNumber(input.speed_limit_kmh, 1))
  const currentSpeed = clampNumber(input.current_speed_kmh, leadSpeed || speedLimit)
  const acceleration = clampNumber(input.acceleration, 0)

  const headwaySeconds =
    currentSpeed > 0 ? (distance / (currentSpeed * (1000 / 3600))) : (eta || 0)

  const visibilityMeters = mapWeatherVisibility(
    input.weather_condition,
    input.visibility_meters
  )
  const roadSlipperiness = clampNumber(
    input.road_slipperiness ?? mapRoadConditionToSlipperiness(input.road_condition),
    0.2
  )

  const morningRush = hour >= 7 && hour < 10 ? 1 : 0
  const eveningRush = hour >= 16 && hour < 19 ? 1 : 0
  const weekend = dayOfWeek === 0 || dayOfWeek === 6 ? 1 : 0

  const lightMap = { Green: 0, Yellow: 1, Red: 2 }
  const currentLight = input.current_light || 'Green'
  const lightCode = lightMap[currentLight] ?? 0
  const isGreen = currentLight === 'Green' ? 1 : 0
  const isRed = currentLight === 'Red' ? 1 : 0

  const requiredSpeedMs = eta > 0 ? distance / eta : (currentSpeed * 1000) / 3600
  const timingUrgency = eta <= 25 ? 1 : 0
  const etaSquared = eta * eta
  const distanceSquared = distance * distance
  const etaDistanceInteraction = eta * distance

  const trafficDensity = totalTraffic / Math.max(distance || 1, 1)
  const rushTraffic = totalTraffic * (morningRush || eveningRush ? 1 : 0)
  const speedRatio = currentSpeed / speedLimit
  const speedDeficit = Math.max(0, speedLimit - currentSpeed)

  const etaRolling3 = getRollingAverage(etaHistory, eta, 3)
  const etaRolling5 = getRollingAverage(etaHistory, eta, 5)

  const hourTraffic = totalTraffic * (hour / 24)
  const rushUrgency = (morningRush || eveningRush ? 1 : 0) * (timingUrgency ? 1 : 0)

  const weatherFlags = getWeatherFlags(input.weather_condition)
  const timeOfDayFlags = getTimeOfDay(hour)
  const visibilityFlags = getVisibilityFlags(visibilityMeters)
  const slipperinessFlags = getSlipperinessFlags(roadSlipperiness)
  const predictedStateFlags = getPredictedStateFlags(currentLight, eta)

  const congestionLevelModerate = totalTraffic >= 8 && totalTraffic <= 20 ? 1 : 0

  const features = [
    vehicleCount,
    pedestrianCount,
    dayOfWeek,
    latitude,
    longitude,
    acceleration,
    visibilityMeters,
    roadSlipperiness,
    distance,
    eta,
    leadSpeed,
    headwaySeconds,
    hour,
    minute,
    hourSin,
    hourCos,
    minuteSin,
    minuteCos,
    morningRush,
    eveningRush,
    weekend,
    lightCode,
    isGreen,
    isRed,
    requiredSpeedMs,
    timingUrgency,
    etaSquared,
    distanceSquared,
    etaDistanceInteraction,
    totalTraffic,
    trafficDensity,
    rushTraffic,
    speedRatio,
    speedDeficit,
    etaRolling3,
    etaRolling5,
    hourTraffic,
    rushUrgency,
    currentLight === 'Red' ? 1 : 0,
    currentLight === 'Yellow' ? 1 : 0,
    weatherFlags.rainy,
    weatherFlags.snowy,
    weatherFlags.sunny,
    timeOfDayFlags.evening,
    timeOfDayFlags.morning,
    timeOfDayFlags.night,
    visibilityFlags.low,
    visibilityFlags.moderate,
    slipperinessFlags.high,
    slipperinessFlags.moderate,
    slipperinessFlags.slight,
    predictedStateFlags.red,
    predictedStateFlags.yellow,
    congestionLevelModerate,
  ]

  if (features.length !== FEATURE_NAMES.length) {
    console.warn('Feature vector length mismatch', features.length, FEATURE_NAMES.length)
  }

  return features.map((value) => (Number.isFinite(value) ? value : 0))
}

export const getFeatureNames = () => [...FEATURE_NAMES]

