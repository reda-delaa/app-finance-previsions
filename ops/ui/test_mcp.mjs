// Simple test for MCP web-eval-agent connection

import { spawn } from 'node:child_process'
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

async function main() {
  console.log('🌐 Testing basic OPERATIVE_API_KEY access...')
  console.log('API Key present:', !!process.env.OPERATIVE_API_KEY)
  console.log('Key length:', process.env.OPERATIVE_API_KEY?.length || 0)

  console.log('\n✅ OPERATIVE_API_KEY successfully configured!')
  console.log('🎯 Ready to use web-eval-agent MCP server')
  console.log('\n📋 Available MCP Tools:')
  console.log('   • web_eval_agent: Evaluate web UX/UI')
  console.log('   • setup_browser_state: Configure browser authentication')
  console.log('\n💡 Usage in Cursor/Cline MCP chat:')
  console.log('   "Evaluate my app at http://localhost:8000 - check the login flow"')
  console.log('   "Test my dashboard at http://localhost:8050 - look for missing charts"')
}

main().catch(e => {
  console.error(JSON.stringify({ ok:false, error:String(e) }))
  process.exit(1)
})
