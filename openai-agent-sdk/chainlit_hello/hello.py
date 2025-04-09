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

# print(result.final_output)
@cl.on_message
async def handle_message(message: cl.Message):
    result = Runner.run_sync(
        starting_agent=agent,
        input=message.content,
        run_config=config
    )

    await cl.Message(content=result.final_output).send()