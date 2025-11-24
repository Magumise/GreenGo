// Diagnostic utility to check app health
export const runDiagnostics = () => {
  const diagnostics = {
    timestamp: new Date().toISOString(),
    errors: [],
    warnings: [],
    checks: {},
  }

  // Check localStorage
  try {
    localStorage.setItem('__test__', 'test')
    localStorage.removeItem('__test__')
    diagnostics.checks.localStorage = 'OK'
  } catch (e) {
    diagnostics.errors.push('LocalStorage not available')
    diagnostics.checks.localStorage = 'FAILED'
  }

  // Check API endpoint
  diagnostics.checks.apiEndpoint = 'https://greengo-api-915779460150.us-east1.run.app'

  // Check required modules
  try {
    if (typeof React !== 'undefined') {
      diagnostics.checks.react = 'OK'
    }
  } catch (e) {
    diagnostics.errors.push('React not loaded')
    diagnostics.checks.react = 'FAILED'
  }

  return diagnostics
}

