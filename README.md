# GreenGo - Smart Traffic Light Detection App

GreenGo is a modern web application that helps drivers navigate routes more efficiently by predicting traffic light changes and providing optimal speed recommendations. The app integrates with a machine learning API to provide real-time traffic light predictions based on various factors including weather, road conditions, and traffic density.

## Features

- 🚦 **Smart Traffic Prediction**: Real-time detection and countdown for upcoming traffic lights
- 🚗 **Optimal Speed Guidance**: AI-powered speed recommendations to catch green lights
- 🗺️ **Interactive Map**: Google Maps-like interface with route simulation
- 📊 **Real-time Updates**: Live tracking of vehicle position and traffic light status
- 📱 **Modern UI**: Beautiful, responsive design with smooth animations
- 📜 **Journey History**: Track and reuse previous routes
- ⚙️ **Settings Management**: Customize your experience

## Tech Stack

- **React 18** - Modern UI framework
- **React Router** - Navigation and routing
- **Leaflet** - Interactive maps
- **Vite** - Fast build tool
- **Axios** - API communication
- **Lucide React** - Modern icons

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn

### Installation

1. Clone the repository or navigate to the project directory
2. Install dependencies:

```bash
npm install
```

### Running the Development Server

```bash
npm run dev
```

The app will open at `http://localhost:3000`

### Building for Production

```bash
npm run build
```

The production build will be in the `dist` folder.

## Project Structure

```
GreenGo-project/
├── src/
│   ├── components/          # Reusable components
│   │   └── NavigationBar.jsx
│   ├── pages/              # Page components
│   │   ├── LandingPage.jsx
│   │   ├── DestinationPage.jsx
│   │   ├── DrivePage.jsx
│   │   ├── HistoryPage.jsx
│   │   └── SettingsPage.jsx
│   ├── services/           # API services
│   │   └── api.js
│   ├── App.jsx             # Main app component
│   ├── main.jsx            # Entry point
│   └── index.css           # Global styles
├── index.html
├── package.json
├── vite.config.js
└── README.md
```

## API Integration

The app integrates with the GreenGo API hosted at:
```
https://greengo-api-915779460150.us-east1.run.app
```

### API Endpoints

- `POST /predict` - Get traffic light predictions and speed recommendations
- `GET /health` - Check API health status

### API Request Format

```json
{
  "timestamp": "2025-01-15T08:30:00",
  "current_light": "Green",
  "eta_to_light_s": 45.5,
  "distance_to_next_light_m": 200.0,
  "vehicle_count": 3,
  "pedestrian_count": 2,
  "lead_vehicle_speed_kmh": 45.0,
  "speed_limit_kmh": 50.0
}
```

## Usage

1. **Landing Page**: Start by clicking "Start Smart Drive"
2. **Destination Input**: Enter your current location and destination
3. **Drive Mode**: View the route on the map and start your journey
4. **Real-time Updates**: Monitor traffic light countdowns and speed recommendations
5. **History**: Review past journeys in the History page
6. **Settings**: Customize app preferences

## Features in Detail

### Traffic Light Detection
- Detects upcoming traffic lights along your route
- Shows countdown timer for light changes
- Displays current light status (Green/Yellow/Red)

### Speed Recommendations
- AI-powered speed suggestions based on:
  - Traffic light timing
  - Weather conditions
  - Road status
  - Vehicle and pedestrian density
  - Speed limits

### Route Simulation
- Interactive map with route visualization
- Animated vehicle movement
- Traffic light markers along the route
- Real-time position updates

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Contributing

This is a competition project. For questions or issues, please contact the development team.

## License

This project is proprietary and confidential.

## Acknowledgments

- OpenStreetMap for map tiles
- Leaflet for map functionality
- The GreenGo API team for the ML model

"# GreenGo" 
