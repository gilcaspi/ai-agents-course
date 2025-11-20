import os
import base64
from tempfile import gettempdir

from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.tools import McpToolset, FunctionTool
from google.adk.tools.mcp_tool import StdioConnectionParams
from mcp import StdioServerParameters
from google.genai import types


def save_image_to_file(image_base64: str, filename: str = "mcp_tiny_image.png") -> dict:
    try:
        image_bytes = base64.b64decode(image_base64)
        save_path = os.path.join(gettempdir(), filename)

        with open(save_path, "wb") as f:
            f.write(image_bytes)

        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Image saved successfully to: {save_path}",
                }
            ]
        }
    except Exception as e:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Failed to save image: {e}",
                }
            ]
        }


save_image_tool = FunctionTool(
    save_image_to_file,
)

mcp_image_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="mcp-server-everything",
            args=[],
        ),
        timeout=30,
    ),
    tool_filter=["getTinyImage"],
)

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

root_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="image_agent",
    instruction=(
        "You have two tools:\n"
        "- `getTinyImage` from `mcp_image_server` (returns a fixed tiny PNG image).\n"
        "- `save_image_tool` (saves a base64 PNG image to disk and returns the file path).\n\n"
        "When the user asks to see the MCP tiny image, always follow this exact algorithm:\n"
        "1) Call the MCP tool `getTinyImage` exactly once.\n"
        "2) From its `function_response.response.content`, find the item with `type: \"image\"`.\n"
        "3) Take its `data` field (the base64 string) and call the tool `save_image_tool` with:\n"
        "   - `image_base64`: that base64 string.\n"
        "   - `filename`: keep the default unless the user explicitly asks for another name.\n"
        "4) Use the text returned by `save_image_tool` as your final answer to the user.\n"
        "5) Your final message must be plain text only, not another tool call."
    ),
    tools=[mcp_image_server, save_image_tool],
)
