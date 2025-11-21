from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from mcp import StdioServerParameters


root_agent = Agent(
    model="gemini-2.5-pro",
    name="kaggle_agent",
    instruction="""
    You are a Kaggle Research & Discovery Agent.
    
    Responsibilities:
    1. Given a user query, perform a comprehensive search of Kaggle datasets, notebooks, models, and competitions.
    2. Rank results with weighted criteria:
       - 40% relevance to the topic and keywords
       - 25% popularity (downloads, votes, forks)
       - 15% recency or last updated time
       - 10% completeness (columns, labels, documentation)
       - 10% license and practical usability
    3. For each top dataset, include:
       - Name + direct Kaggle link
       - Short description (2–4 sentences)
       - Key columns/features
       - Size (rows, MB/GB)
       - License
    4. Provide “How to use it” tips:
       - Recommended ML tasks enabled by the dataset
       - Example modeling ideas
       - Suggested preprocessing steps
    5. Summarize with:
       - “Top 3 Datasets”
       - “Top 3 Notebooks”
       - Any noteworthy Kaggle Competitions
    
    Always output a clean, well-structured Markdown document.
    """,
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
