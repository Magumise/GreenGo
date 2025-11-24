import express from 'express'
import fetch from 'node-fetch'

const TARGET_API = process.env.TARGET_API_URL || 'https://greengo-api-915779460150.us-east1.run.app'
const PORT = process.env.PORT || 8080

const app = express()
app.use(express.json())

const addCorsHeaders = (res) => {
  res.setHeader('Access-Control-Allow-Origin', process.env.ALLOWED_ORIGIN || '*')
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type,Authorization')
}

app.options('*', (req, res) => {
  addCorsHeaders(res)
  res.sendStatus(204)
})

app.get('/health', (req, res) => {
  addCorsHeaders(res)
  res.json({ status: 'ok', target: TARGET_API })
})

app.post('/predict', async (req, res) => {
  addCorsHeaders(res)

  try {
    const response = await fetch(`${TARGET_API}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(req.body),
    })

    const text = await response.text()
    res.status(response.status).type(response.headers.get('content-type') || 'application/json').send(text)
  } catch (error) {
    console.error('Proxy error:', error)
    res.status(502).json({ message: 'Proxy error', details: error.message })
  }
})

app.listen(PORT, () => {
  console.log(`GreenGo proxy listening on port ${PORT}, forwarding to ${TARGET_API}`)
})

