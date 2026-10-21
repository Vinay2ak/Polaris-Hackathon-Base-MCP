"""One MCP tool, ordinary JSON HTTP responses, no frontend required."""
import contextlib
import os
from pathlib import Path

import anyio
import uvicorn
from mcp import types
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from search import Catalog, INPUT_SCHEMA, OUTPUT_SCHEMA

server = Server('polaris-product-search', version='1.0.0')
catalog = None


@server.list_tools()
async def list_tools():
    return [types.Tool(name='search_products',
        description='Search the product catalog with filters, ranking, and optional complementary recommendations.',
        inputSchema=INPUT_SCHEMA, outputSchema=OUTPUT_SCHEMA)]


@server.call_tool()
async def call_tool(name, arguments):
    if name != 'search_products':
        raise ValueError('Unknown tool')
    # Keep synchronous baseline search off the HTTP event loop.
    return await anyio.to_thread.run_sync(catalog.search, arguments)


allowed_hosts_env = os.getenv('MCP_ALLOWED_HOSTS', 'localhost:*,127.0.0.1:*,[::1]:*,delay-fanciness-mutate.ngrok-free.dev,delay-fanciness-mutate.ngrok-free.dev:*,dummy-mcp.01m5erq7wn81nf6mhfpqkjb0ka.apps.springwinter.dev,dummy-mcp.01m5erq7wn81nf6mhfpqkjb0ka.apps.springwinter.dev:*')
allowed_hosts = [h.strip() for h in allowed_hosts_env.split(',') if h.strip()]
enable_dns = '*' not in allowed_hosts

security = TransportSecuritySettings(
    enable_dns_rebinding_protection=enable_dns,
    allowed_hosts=allowed_hosts,
    allowed_origins=[o.strip() for o in os.getenv('MCP_ALLOWED_ORIGINS', '').split(',') if o.strip()],
)
sessions = StreamableHTTPSessionManager(server, json_response=True, stateless=True, security_settings=security)


@contextlib.asynccontextmanager
async def lifespan(app):
    global catalog
    path = os.getenv('CATALOG_PATH', str(Path(__file__).parent / 'data/products.json'))
    catalog = await anyio.to_thread.run_sync(Catalog.load, path)
    async with sessions.run():
        yield


class MCPApplication:
    async def __call__(self, scope, receive, send):
        await sessions.handle_request(scope, receive, send)


async def health(request):
    return JSONResponse({'status': 'ok', 'products': len(catalog.products)})


app = Starlette(lifespan=lifespan, routes=[
    Route('/mcp', endpoint=MCPApplication(), methods=['POST', 'GET', 'DELETE']),
    Route('/healthz', endpoint=health),
])

if __name__ == '__main__':
    uvicorn.run(app, host=os.getenv('HOST', '0.0.0.0'), port=int(os.getenv('PORT', '8000')))
