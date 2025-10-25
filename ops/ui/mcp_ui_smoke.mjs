// Minimal MCP Playwright UI smoke runner (best‑effort).
// Requires: npm i -D @modelcontextprotocol/sdk
// Runs @playwright/mcp via npx and calls tools to navigate + screenshot.

import { spawn } from 'node:child_process'
import { writeFile, mkdir } from 'node:fs/promises'
import { join } from 'node:path'

let Client, StdioClientTransport
try {
  // Lazy import to allow environments without the SDK to still run the repo
  ;({ Client } = await import('@modelcontextprotocol/sdk/client/index.js'))
  ;({ StdioClientTransport } = await import('@modelcontextprotocol/sdk/client/stdio.js'))
} catch (e) {
  console.error(JSON.stringify({ ok: false, error: 'mcp-sdk-missing', hint: 'npm i -D @modelcontextprotocol/sdk' }))
  process.exit(1)
}

const UI_BASE = (process.env.UI_BASE || 'http://localhost:8501').replace(/\/$/, '')
const PAGES = [
  'Dashboard',
  'Agents_Status',
  'LLM_Scoreboard',
  'Earnings',
  'Risk',
  'Recession',
  'Signals',
  'Portfolio',
  'Alerts',
]

function findTool(tools, names) {
  const lower = tools.map(t => ({ name: t.name, desc: (t.description||'') }))
  for (const n of names) {
    const hit = lower.find(t => t.name.toLowerCase().includes(n))
    if (hit) return hit.name
  }
  return null
}

async function main() {
  // Start @playwright/mcp server via npx
  const server = spawn('npx', ['@playwright/mcp'], { stdio: ['pipe','pipe','inherit'] })
  server.on('error', err => { console.error(JSON.stringify({ ok:false, error:'spawn-failed', detail: String(err) })); process.exit(1) })

  const transport = new StdioClientTransport(server.stdin, server.stdout)
  const client = new Client({ name: 'ui-smoke-mcp', version: '0.1.0' }, { capabilities: {} })
  await client.connect(transport)

  const { tools } = await client.listTools()
  const toolNames = tools.map(t => t.name)

  const navTool = findTool(tools, ['goto','navigate','page_goto'])
  const shotTool = findTool(tools, ['screenshot','page_screenshot','screencap'])
  if (!navTool) {
    console.error(JSON.stringify({ ok:false, error:'no-nav-tool', tools: toolNames }))
    process.exit(2)
  }
  if (!shotTool) {
    console.error(JSON.stringify({ ok:false, warn:'no-screenshot-tool', tools: toolNames }))
  }

  const outDir = 'artifacts/smoke/ui_mcp'
  await mkdir(outDir, { recursive: true })
  const results = []

  for (const p of PAGES) {
    const url = `${UI_BASE}/${p}`
    let status = 'ok'
    const errors = []
    try {
      await client.callTool({ name: navTool, arguments: { url } })
      if (shotTool) {
        const res = await client.callTool({ name: shotTool, arguments: {} })
        const item = (res?.content || [])[0]
        if (item && item.type === 'image') {
          const b64 = item.data || item.base64 || ''
          const buf = Buffer.from(b64, 'base64')
          await writeFile(join(outDir, `${p}.png`), buf)
        }
      }
    } catch (e) {
      status = 'error'
      errors.push(String(e))
    }
    results.push({ page: p, url, status, errors })
  }

  const repPath = join('data','reports',`dt=${new Date().toISOString().slice(0,10).replace(/-/g,'')}`,'ui_smoke_mcp_report.json')
  await mkdir(repPath.substring(0, repPath.lastIndexOf('/')), { recursive: true })
  await writeFile(repPath, JSON.stringify({ ok:true, tools: toolNames, results }, null, 2), 'utf-8')

  await client.close()
  server.kill('SIGTERM')
  console.log(JSON.stringify({ ok:true, pages: results.length, report: repPath }))
}

main().catch(e => { console.error(JSON.stringify({ ok:false, error:String(e) })); process.exit(1) })

