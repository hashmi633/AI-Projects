from dotenv import load_dotenv
import chainlit as cl
import os
from agents import AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, Agent, Runner
from typing import cast


load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

@cl.on_chat_start
async def start():
    external_client = AsyncOpenAI(
        api_key = gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    model = OpenAIChatCompletionsModel(
        model= "gemini-2.0-flash",
        openai_client = external_client
    )
    config = RunConfig(
        model = model,
        model_provider=external_client,
        tracing_disabled=True
    )

    cl.user_session.set("config", config)

    urdu_agent = Agent(
        name="Urdu Agent",
        instructions="You only speak Urdu"
        )
    
    english_agent = Agent(
        name="English Agent",
        instructions="You only speak English"
    )

    triage_agent = Agent(
        name="Triage Agent",
        instructions="Handoff to the appropriate agent based on the language of the request",
        handoffs=[urdu_agent, english_agent]
    )
    cl.user_session.set("agent", triage_agent)
    
    msg = cl.Message(content="Hi! What would you like translated? ")
    await msg.send()

@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="Translating...")
    await msg.send()

    config = cast(RunConfig, cl.user_session.get("config"))
    agent = cast(Agent, cl.user_session.get("agent"))

    try:
        result = await Runner.run(
            agent,
            message.content,
            run_config=config,
        )
        response = result.final_output
        msg.content = response
        await msg.update()

    except Exception as e:
        msg.content = f"Error: {str(e)}"
        await msg.update()
