import asyncio
import os
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search


GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


async def hello_world_from_my_first_agent():
    response = await runner.run_debug(
        "Who is Gil Caspi, data scientist?"
    )
    print(response)


if __name__ == '__main__':
    root_agent = Agent(
        name="helpful_assistant",
        model="gemini-2.5-flash-lite",
        description="A simple agent that can answer general questions.",
        instruction="You are a helpful assistant. Use Google Search for current info or if unsure.",
        tools=[google_search],
    )
    print("✅ Root Agent defined.")


    runner = InMemoryRunner(agent=root_agent)
    print("✅ Runner created.")

    asyncio.run(hello_world_from_my_first_agent())