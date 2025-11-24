import React, { useState } from 'react'
import { User, Bell, Shield, HelpCircle, LogOut, Moon, Sun } from 'lucide-react'
import NavigationBar from '../components/NavigationBar'
import './SettingsPage.css'

const SettingsPage = () => {
  const [darkMode, setDarkMode] = useState(false)
  const [notifications, setNotifications] = useState(true)
  const [userName, setUserName] = useState('Driver')
  const [userEmail, setUserEmail] = useState('driver@greengo.app')

  const handleLogout = () => {
    // In a real app, this would handle logout
    alert('Logout functionality would be implemented here')
  }

  return (
    <div className="settings-page">
      <div className="settings-header">
        <h1>GreenGo</h1>
        <p className="header-subtitle">Settings</p>
      </div>

      <div className="settings-content">
        <div className="profile-section">
          <div className="profile-card">
            <div className="profile-avatar">
              <User size={32} />
            </div>
            <div className="profile-info">
              <h2>{userName}</h2>
              <p>{userEmail}</p>
            </div>
          </div>
        </div>

        <div className="settings-section">
          <h3 className="section-title">Account</h3>
          <div className="settings-list">
            <div className="setting-item">
              <div className="setting-icon">
                <User size={20} />
              </div>
              <div className="setting-content">
                <span className="setting-label">Edit Profile</span>
                <span className="setting-description">Update your personal information</span>
              </div>
            </div>
            <div className="setting-item">
              <div className="setting-icon">
                <Shield size={20} />
              </div>
              <div className="setting-content">
                <span className="setting-label">Privacy & Security</span>
                <span className="setting-description">Manage your privacy settings</span>
              </div>
            </div>
          </div>
        </div>

        <div className="settings-section">
          <h3 className="section-title">Preferences</h3>
          <div className="settings-list">
            <div className="setting-item toggle-item">
              <div className="setting-icon">
                <Bell size={20} />
              </div>
              <div className="setting-content">
                <span className="setting-label">Notifications</span>
                <span className="setting-description">Receive alerts and updates</span>
              </div>
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={notifications}
                  onChange={(e) => setNotifications(e.target.checked)}
                />
                <span className="toggle-slider"></span>
              </label>
            </div>
            <div className="setting-item toggle-item">
              <div className="setting-icon">
                {darkMode ? <Moon size={20} /> : <Sun size={20} />}
              </div>
              <div className="setting-content">
                <span className="setting-label">Dark Mode</span>
                <span className="setting-description">Switch to dark theme</span>
              </div>
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={darkMode}
                  onChange={(e) => setDarkMode(e.target.checked)}
                />
                <span className="toggle-slider"></span>
              </label>
            </div>
          </div>
        </div>

        <div className="settings-section">
          <h3 className="section-title">Support</h3>
          <div className="settings-list">
            <div className="setting-item">
              <div className="setting-icon">
                <HelpCircle size={20} />
              </div>
              <div className="setting-content">
                <span className="setting-label">Help & Support</span>
                <span className="setting-description">Get help with GreenGo</span>
              </div>
            </div>
            <div className="setting-item">
              <div className="setting-icon">
                <Shield size={20} />
              </div>
              <div className="setting-content">
                <span className="setting-label">Terms & Privacy</span>
                <span className="setting-description">Read our terms and privacy policy</span>
              </div>
            </div>
          </div>
        </div>

        <div className="settings-section">
          <button className="logout-button" onClick={handleLogout}>
            <LogOut size={20} />
            Log Out
          </button>
        </div>

        <div className="app-version">
          <p>GreenGo v1.0.0</p>
          <p className="version-subtitle">Drive smarter, catch more greens</p>
        </div>
      </div>

      <NavigationBar />
    </div>
  )
}

export default SettingsPage

