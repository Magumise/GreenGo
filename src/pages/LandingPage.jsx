import React from 'react'
import { useNavigate } from 'react-router-dom'
import { Zap, TrendingUp, Leaf, ArrowRight } from 'lucide-react'
import './LandingPage.css'

const LandingPage = () => {
  const navigate = useNavigate()

  const handleStart = () => {
    navigate('/destination')
  }

  return (
    <div className="landing-page-new" style={{ minHeight: '100vh', background: '#f9fafb' }}>
      <div className="landing-hero">
        <div className="hero-content">
          <div className="logo-badge">
            <Leaf className="logo-icon" size={32} />
            <span className="logo-text">GreenGo</span>
          </div>
          <h1 className="hero-title">
            Never Miss a Green Light
            <span className="title-highlight"> Again</span>
          </h1>
          <p className="hero-subtitle">
            AI-powered traffic prediction that helps you drive smarter, 
            save fuel, and reduce your carbon footprint
          </p>
          <button className="cta-button" onClick={handleStart}>
            Start Your Journey
            <ArrowRight size={20} />
          </button>
        </div>
        <div className="hero-visual">
          <div className="visual-card">
            <div className="traffic-light-visual">
              <div className="light green active"></div>
              <div className="light yellow"></div>
              <div className="light red"></div>
            </div>
            <div className="countdown-visual">12s</div>
          </div>
        </div>
      </div>

      <div className="features-section">
        <div className="feature-card">
          <div className="feature-icon-wrapper blue">
            <Zap size={24} />
          </div>
          <h3>Real-Time Predictions</h3>
          <p>Get instant traffic light predictions powered by advanced AI</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon-wrapper green">
            <TrendingUp size={24} />
          </div>
          <h3>Optimal Speed</h3>
          <p>Receive personalized speed recommendations to catch every green</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon-wrapper purple">
            <Leaf size={24} />
          </div>
          <h3>Eco-Friendly</h3>
          <p>Reduce emissions and fuel consumption with smart routing</p>
        </div>
      </div>

      <div className="stats-section">
        <div className="stat-item">
          <div className="stat-number">18%</div>
          <div className="stat-label">Fuel Savings</div>
        </div>
        <div className="stat-item">
          <div className="stat-number">95%</div>
          <div className="stat-label">Accuracy</div>
        </div>
        <div className="stat-item">
          <div className="stat-number">2.5k+</div>
          <div className="stat-label">Happy Drivers</div>
        </div>
      </div>
    </div>
  )
}

export default LandingPage
