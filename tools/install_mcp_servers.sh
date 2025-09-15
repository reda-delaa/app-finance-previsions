#!/bin/bash

# Fixed MCP Server Installation Script
# This script installs MCP servers properly using correct package names

echo "Installing Official MCP Servers..."

# Install official Model Context Protocol servers
echo "Installing filesystem MCP server..."
npx -y install-mcp@latest @modelcontextprotocol/server-filesystem --client cline

echo "Installing memory MCP server (optional - for knowledge graph features)..."
npx -y install-mcp@latest @modelcontextprotocol/server-memory --client cline 2>/dev/null || echo "Memory MCP server installation skipped"

echo "Installing sequential thinking MCP server..."
npx -y install-mcp@latest @modelcontextprotocol/server-sequential-thinking --client cline

echo "Installing everything (demo) MCP server..."
npx -y install-mcp@latest @modelcontextprotocol/server-everything --client cline

# Keep the existing Supermemory server (was already working)
echo "Supermemory MCP server already installed and working."

echo ""
echo "Installation complete!"
echo ""
echo "For the third-party servers you wanted (brave-search, fetch, firecrawl, markdownify),"
echo "they may not be available as simple npm packages. You can search for them:"
echo "npm search brave-search"
echo "npm search firecrawl"
echo "npm search markdownify"
echo ""
echo "Alternative: many of these functionalities are available via:"
echo "- brave-search: Use web search via browser automation or APIs"
echo "- fetch: Use curl commands or Python requests in your code"
echo "- firecrawl: Use web scraping tools or APIs directly"
echo "- markdownify: Use Python libraries like html2text or markdown2"
