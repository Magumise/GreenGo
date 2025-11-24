import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { Car, History, Settings } from 'lucide-react'
import './NavigationBar.css'

const NavigationBar = () => {
  const location = useLocation()

  const isActive = (path) => location.pathname === path

  return (
    <nav className="navigation-bar">
      <Link 
        to="/drive" 
        className={`nav-item ${isActive('/drive') ? 'active' : ''}`}
      >
        <Car size={24} />
        <span>Go-Green</span>
      </Link>
      <Link 
        to="/history" 
        className={`nav-item ${isActive('/history') ? 'active' : ''}`}
      >
        <History size={24} />
        <span>History</span>
      </Link>
      <Link 
        to="/settings" 
        className={`nav-item ${isActive('/settings') ? 'active' : ''}`}
      >
        <Settings size={24} />
        <span>Settings</span>
      </Link>
    </nav>
  )
}

export default NavigationBar

