import os
from agents import AsyncOpenAI,Agent, OpenAIChatCompletionsModel, RunConfig, Runner, ModelProvider
from dotenv import load_dotenv
from weather_tool import get_weather
import asyncio
from typing import cast
import chainlit as cl

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

@cl.on_chat_start
async def chat_start():
    external_client = AsyncOpenAI(
        api_key = GEMINI_API_KEY,
        base_url= "https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client= external_client,
    )

    config = RunConfig(
        model=model,
        model_provider= cast(ModelProvider, external_client),
        tracing_disabled=True,
    )

    weather_assistant = Agent(
        name="Weather Assistant",
        instructions="""You are a weather assistant that can provide current weather information.
    
    When asked about weather, use the get_weather tool to fetch accurate data.
    If the user doesn't specify a country code and there might be ambiguity,
    ask for clarification (e.g., Paris, France vs. Paris, Texas).
    
    Provide friendly commentary along with the weather data, such as clothing suggestions
    or activity recommendations based on the conditions.
    """,
    tools=[get_weather],
    )
    cl.user_session.set("agent", weather_assistant)
    cl.user_session.set("config", config)
    msg = cl.Message(content="Hi! Which city weather you like to check ?")
    await msg.send()

@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="Checking Weather")
    await msg.send()
    try:
        agent = cast(Agent, cl.user_session.get("agent"))
        config = cast(RunConfig, cl.user_session.get("config"))
        result = await Runner.run(
            agent,
            message.content,
            run_config=config,
        )
        print(result.final_output)
        msg.content = result.final_output
        await msg.update()
    except Exception as e:
        return f"Error: {str(e)}"
    
if __name__ == "__main__":
    asyncio.run(main())