import asyncio
import os

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.sessions import DatabaseSessionService
from google.adk.runners import Runner
from google.adk.apps import App
from google.genai import types


# -------------------- Load API key --------------------
load_dotenv()
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")


# -------------------- Constants --------------------
APP_NAME = "StatefulApp"
USER_ID = "default"          # Real-world: this is per-user
MODEL_NAME = "gemini-2.5-flash-lite"


retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)


# -------------------- Helper: run a session --------------------
async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
):

    print(f"\n### Session: {session_name}")

    # Get or create session
    try:
        session = await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_name
        )

    if not user_queries:
        print("No queries!")
        return

    # Always list
    if isinstance(user_queries, str):
        user_queries = [user_queries]

    for query in user_queries:
        print(f"\nUser > {query}")

        # ADK message
        new_msg = types.Content(
            role="user",
            parts=[types.Part(text=query)],
        )

        # Stream events
        async for event in runner_instance.run_async(
            user_id=USER_ID,
            session_id=session.id,
            new_message=new_msg,
        ):
            if (
                event.content
                and event.content.parts
                and event.content.parts[0].text
                and event.content.parts[0].text != "None"
            ):
                print(f"{MODEL_NAME} > {event.content.parts[0].text}")


# -------------------- Create Agent --------------------
root_agent = LlmAgent(
    name="text_chat_bot",
    description="A text chatbot with persistent memory",
    model=Gemini(
        model=MODEL_NAME,
        retry_options=retry_config,
    ),
)


# -------------------- Wrap into App (REQUIRED BY ADK) --------------------
app = App(
    name=APP_NAME,
    root_agent=root_agent,
)

apps = [app]

# -------------------- Session DB --------------------
session_service = DatabaseSessionService(
    db_url="sqlite+aiosqlite:///my_agent_data.db"
)

# -------------------- Create runner --------------------
runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

print("✅ Stateful ADK Agent initialized!")
print(f"   Application: {APP_NAME}")
print(f"   Using DB: my_agent_data.db")
print(f"   User: {USER_ID}")


# -------------------- Examples --------------------
async def teach_and_test(runner):
    await run_session(
        runner,
        [
            "Hi, I am Sam! What is the capital of United States?",
            "Hello! What is my name?",
        ],
        "stateful-agentic-session",
    )


async def only_test(runner):
    await run_session(
        runner,
        "What is my name?",
        "stateful-agentic-session",
    )


if __name__ == "__main__":
    # 1st run only
    # asyncio.run(teach_and_test(runner))

    asyncio.run(only_test(runner))
