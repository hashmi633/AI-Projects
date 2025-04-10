from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from dotenv import load_dotenv, find_dotenv
from agents.run import RunConfig
import chainlit as cl
import os

load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")

#Step 1: Provider
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

#Step 2: model

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

# Config
config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

# Step 3: Agent
agent : Agent = Agent(
            name="Assistant",
            instructions="You are a helpful assistant",
            model=model)

result = Runner.run_sync(
    starting_agent=agent,
    input="Who is winner of last football worldcup",
    run_config=config
)

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello! I am the panaversity support agent. How can i help you ?")

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})
    result = await Runner.run(
        starting_agent=agent,
        input=history,
        run_config=config
    )
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)
    await cl.Message(content=result.final_output).send()
    