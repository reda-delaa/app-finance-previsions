// MCP UX Evaluation using web-eval-agent for Dash UI
// Requires: uv installed

import { spawn } from 'node:child_process'
import { writeFile, mkdir } from 'node:fs/promises'
import { join } from 'node:path'
import { config } from 'dotenv'

// Load environment variables from .env file
config()

let Client, StdioClientTransport
try {
  ;({ Client } = await import('@modelcontextprotocol/sdk/client/index.js'))
  ;({ StdioClientTransport } = await import('@modelcontextprotocol/sdk/client/stdio.js'))
} catch (e) {
  console.error(JSON.stringify({ ok: false, error: 'mcp-sdk-missing', hint: 'npm i -D @modelcontextprotocol/sdk' }))
  process.exit(1)
}

const DASH_BASE = (process.env.DASH_BASE || `http://localhost:${process.env.AF_DASH_PORT||'8050'}`).replace(/\/$/, '')
const PAGES = [
  { path: '/', name: 'Root (redirect to Dashboard)' },
  { path: '/dashboard', name: 'Dashboard' },
  { path: '/signals', name: 'Signals' },
  { path: '/portfolio', name: 'Portfolio' },
  { path: '/agents', name: 'Agents Status' },
  { path: '/observability', name: 'Observability' },
  { path: '/regimes', name: 'Regimes' },
  { path: '/risk', name: 'Risk' },
  { path: '/recession', name: 'Recession' },
]

async function evaluatePage(client, url, pageName, task) {
  try {
    const res = await client.callTool({ name: 'web_eval_agent', arguments: { url, task: task(pageName), headless_browser: true } })
    const text = ((res?.content || []).find(c => c.type === 'text') || {}).text || ''
    const image = (res?.content || []).find(c => c.type === 'image')
    return { status: 'ok', evaluation: text, screenshot: image ? image : null }
  } catch (e) {
    return { status: 'error', evaluation: String(e) }
  }
}

async function main() {
  const server = spawn('uvx', ['--refresh-package', 'webEvalAgent', '--from', 'git+https://github.com/Operative-Sh/web-eval-agent.git', 'webEvalAgent'], {
    stdio: ['pipe','pipe','inherit'],
    env: { ...process.env, OPERATIVE_API_KEY: process.env.OPERATIVE_API_KEY }
  })
  server.on('error', err => { console.error(JSON.stringify({ ok:false, error:'spawn-web-eval-agent-failed', detail: String(err) })); process.exit(1) })

  // Wait a moment for the server to start
  await new Promise(resolve => setTimeout(resolve, 3000))

  if (!server.stdin || !server.stdout) {
    console.error(JSON.stringify({ ok:false, error: 'mcp-server-no-streams', hint: 'MCP server may not have started or git repo unavailable' }))
    process.exit(1)
  }

  const transport = new StdioClientTransport(server.stdin, server.stdout)
  const client = new Client({ name: 'dash-ux-eval-mcp', version: '0.1.0' }, { capabilities: {} })
  try {
    await client.connect(transport)
  } catch (connectErr) {
    console.error(JSON.stringify({ ok:false, error: 'mcp-connect-failed', hint: 'Check OPERATIVE_API_KEY and network', detail: String(connectErr) }))
    process.exit(1)
  }

  const results = []
  const outDir = 'artifacts/smoke/dash_eval'
  await mkdir(outDir, { recursive: true })

  const taskGenerator = (pageName) => `
Evaluate the user experience of the financial dashboard ${pageName} page.
Check for any errors or missing elements.
For macro pages (Regimes/Risk/Recession), ensure Plotly charts are rendered with multi-series data (bars by horizon).
For all pages, verify badges and tables are displayed.
If charts are missing, note that data might not be loaded.
Assess overall usability and report any issues.
`

  for (const { path, name } of PAGES) {
    const url = `${DASH_BASE}${path}`
    console.error(`Evaluating ${name} at ${url}`)
    const result = await evaluatePage(client, url, name, taskGenerator)
    results.push({ page: name, path, url, ...result })

    // Save screenshot if available
    if (result.screenshot && result.screenshot.data) {
      const b64 = result.screenshot.data
      if (b64 && typeof b64 === 'string') {
        try {
          const buf = Buffer.from(b64, 'base64')
          const fname = (name ? name.replace(/[\s/]/g, '_').toLowerCase() : 'page') + '.png'
          await writeFile(join(outDir, fname), buf)
        } catch (e) {
          console.error(`Screenshot save error for ${name}:`, e.message)
        }
      }
    }
  }

  await client.close()
  server.kill('SIGTERM')

  const report = { ok:true, base: DASH_BASE, timestamp: new Date().toISOString(), results }
  const repPath = `data/reports/dash_ux_eval_report.json`
  try {
    await writeFile(repPath, JSON.stringify(report, null, 2), 'utf-8')
    console.log(JSON.stringify({ ok:true, message: 'UX evaluation complete', report: repPath, screenshots: outDir }))
  } catch (writeErr) {
    console.error(JSON.stringify({ ok: false, error: `Failed to write report: ${writeErr.message}`, reportPath: repPath }))
  }
}

main().catch(e => { console.error(JSON.stringify({ ok:false, error:String(e) })); process.exit(1) })
