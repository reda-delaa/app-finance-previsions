// MCP Playwright smoke for Dash UI
// Requires: npm i -D @modelcontextprotocol/sdk

import { spawn } from 'node:child_process'
import { writeFile, mkdir } from 'node:fs/promises'
import { join } from 'node:path'

let Client, StdioClientTransport
try {
  ;({ Client } = await import('@modelcontextprotocol/sdk/client/index.js'))
  ;({ StdioClientTransport } = await import('@modelcontextprotocol/sdk/client/stdio.js'))
} catch (e) {
  console.error(JSON.stringify({ ok: false, error: 'mcp-sdk-missing', hint: 'npm i -D @modelcontextprotocol/sdk' }))
  process.exit(1)
}

const DASH_BASE = (process.env.DASH_BASE || `http://localhost:${process.env.AF_DASH_PORT||'8050'}`).replace(/\/$/, '')
const PATHS = ['/', '/dashboard', '/signals', '/portfolio', '/observability']

function findTool(tools, names) {
  const lower = tools.map(t => ({ name: t.name, desc: (t.description||'') }))
  for (const n of names) {
    const hit = lower.find(t => t.name.toLowerCase().includes(n))
    if (hit) return hit.name
  }
  return null
}

async function main() {
  const server = spawn('npx', ['@playwright/mcp'], { stdio: ['pipe','pipe','inherit'] })
  server.on('error', err => { console.error(JSON.stringify({ ok:false, error:'spawn-failed', detail: String(err) })); process.exit(1) })

  const transport = new StdioClientTransport(server.stdin, server.stdout)
  const client = new Client({ name: 'dash-smoke-mcp', version: '0.1.0' }, { capabilities: {} })
  await client.connect(transport)

  const { tools } = await client.listTools()
  const toolNames = tools.map(t => t.name)
  const navTool = findTool(tools, ['goto','navigate','page_goto'])
  const shotTool = findTool(tools, ['screenshot','page_screenshot','screencap'])
  if (!navTool) {
    console.error(JSON.stringify({ ok:false, error:'no-nav-tool', tools: toolNames }))
    process.exit(2)
  }

  const outDir = 'artifacts/smoke/dash_mcp'
  await mkdir(outDir, { recursive: true })
  const results = []

  for (const path of PATHS) {
    const url = `${DASH_BASE}${path}`
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
          const fname = (path === '/' ? 'root' : path.replace(/^\//,'')) + '.png'
          await writeFile(join(outDir, fname), buf)
        }
      }
    } catch (e) {
      status = 'error'
      errors.push(String(e))
    }
    results.push({ path, url, status, errors })
  }

  await client.close()
  server.kill('SIGTERM')
  const report = { ok:true, base: DASH_BASE, results }
  const repPath = join('data','reports',`dt=${new Date().toISOString().slice(0,10).replace(/-/g,'')}`,'dash_smoke_mcp_report.json')
  await mkdir(repPath.substring(0, repPath.lastIndexOf('/')), { recursive: true })
  await writeFile(repPath, JSON.stringify(report, null, 2), 'utf-8')
  console.log(JSON.stringify({ ok:true, report: repPath }))
}

main().catch(e => { console.error(JSON.stringify({ ok:false, error:String(e) })); process.exit(1) })

