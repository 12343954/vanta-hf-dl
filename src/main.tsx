import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

window.requestAnimationFrame(() => {
  document.body.classList.add('app-ready')
  window.setTimeout(() => {
    document.getElementById('vanta-splash')?.remove()
  }, 420)
})
