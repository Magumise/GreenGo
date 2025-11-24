# GreenGo - Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies
```bash
npm install
```

### Step 2: Start Development Server
```bash
npm run dev
```

### Step 3: Open in Browser
The app will automatically open at `http://localhost:3000`

## 📱 Using the App

1. **Landing Page**: Click "Start Smart Drive" to begin
2. **Enter Destination**: 
   - Your current location is pre-filled
   - Enter your destination or select from popular places
   - Click "Start Smart Drive"
3. **Drive Mode**:
   - View your route on the interactive map
   - Click "Start Journey" to begin simulation
   - Watch real-time traffic light predictions
   - Follow speed recommendations

## 🔧 API Configuration

The app is configured to use the GreenGo API at:
```
https://greengo-api-915779460150.us-east1.run.app
```

If you need to change the API endpoint, edit `src/services/api.js`

## 🎨 Features

- ✅ Modern, responsive UI
- ✅ Real-time traffic light detection
- ✅ Speed recommendations from ML model
- ✅ Interactive map with route simulation
- ✅ Journey history tracking
- ✅ Settings management

## 🐛 Troubleshooting

**Map not loading?**
- Check your internet connection (maps load from OpenStreetMap)
- Ensure Leaflet CSS is loading properly

**API errors?**
- Verify the API endpoint is accessible
- Check browser console for detailed error messages
- The app will use fallback predictions if API is unavailable

**Build errors?**
- Delete `node_modules` and run `npm install` again
- Ensure you're using Node.js v16 or higher

## 📦 Building for Production

```bash
npm run build
```

The production files will be in the `dist` folder.

## 🎯 Next Steps

- Customize colors and branding in `src/index.css`
- Add more destinations in `src/pages/DestinationPage.jsx`
- Enhance route simulation in `src/pages/DrivePage.jsx`

