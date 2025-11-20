import os

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPServerParams, StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, McpToolset
from mcp import StdioServerParameters

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


root_agent = Agent(
    model="gemini-2.5-pro",
    name="github_agent",
    instruction="Help users get information from GitHub",
    tools=[
        McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command='npx',
                    args=[
                        '-y',
                        'mcp-remote',
                        'https://www.kaggle.com/mcp'
                    ],
                ),
                timeout=30,
            )
        )
    ],
)