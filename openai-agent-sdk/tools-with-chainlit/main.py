from dotenv import load_dotenv
import os
import chainlit as cl
from typing import List, cast
from agents import AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, Agent, function_tool,Runner

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

@cl.set_starters
async def set_starts()->List[cl.Starter]:
    return [
        cl.Starter(
            label="Greetings",
            message="Hello! What can you help me with today?"
        ),
        cl.Starter(
            label="Weather",
            message="Find the weather in Karachi.",
        ),
    ]

@function_tool
def get_weather(location: str)-> str:
    """
    Fetch the weather for a given location, returning a short description.
    """
    return f"The weather in {location} is 22 degrees C."

@cl.on_chat_start
async def start():
    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    model = OpenAIChatCompletionsModel(
        model= "gemini-2.0-flash",
        openai_client=external_client
    )

    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )

    cl.user_session.set("chat_history", [])
    cl.user_session.set("config", config)

    agent = Agent(name="Assistant", instructions="You are a helpful assistant", model=model)
    agent.tools.append(get_weather)
    cl.user_session.set("agent", agent)

    msg = cl.Message(content="Welcome to the Panaversity AI Assistant! How can I help you today?")
    await msg.send()

@cl.on_message
async def main(message: cl.Message):
    """Process incoming messages and generate responses."""
    msg = cl.Message(content="Thinking...")
    await msg.send() 

    agent : Agent = cast(Agent, cl.user_session.get("agent"))
    config : RunConfig = cast(RunConfig, cl.user_session.get("config"))

    history = cl.user_session.get("chat_history") or []
    history.append({"role": "user", "content": message.content})

    try:
        print("\n[CALLING_AGENT_WITH_CONTEXT]\n", history, "\n")
        result = Runner.run_sync(
            agent,
            history,
            run_config=config
        )

        response_content = result.final_output
        msg.content = response_content
        await msg.update()

        history.append({"role": "developer", "content": response_content})
        cl.user_session.set("chat_history", history)

        print(f"User: {message.content}")
        print(f"Assistant: {response_content}")

    except Exception as e:
        msg.content = f"Error: {str(e)}"
        await msg.update()
        print(f"Error: {str(e)}")