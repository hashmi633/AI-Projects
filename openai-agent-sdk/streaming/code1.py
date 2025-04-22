from dotenv import load_dotenv, find_dotenv
from agents import AsyncOpenAI, OpenAIChatCompletionsModel, Agent, Runner
from agents.run import RunConfig
from openai.types.responses import ResponseTextDeltaEvent
import os
import chainlit as cl

load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")

#Step 1: Provider
provider = AsyncOpenAI(
    api_key = gemini_api_key,
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
)

#Step 2: Model 
model = OpenAIChatCompletionsModel(
    model= "gemini-2.0-flash",
    openai_client=provider
)

#Step 3: provider and model configuration
config = RunConfig(
    model= model,
    model_provider= provider,
    tracing_disabled= True 
)

#Step 4: creation of agent
agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant",
        model=model
)

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    msg = cl.Message(content= "Hello! I am the panaversity support agent. How can i help you ?")
    await msg.send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")
    
    msg = cl.Message(content="")
    await msg.send()
    
    history.append({"role": "user", "content":message.content})

    result = Runner.run_streamed(
        starting_agent=agent,
        input = history,
        run_config = config
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)
            # print(event.data.delta, end="", flush=True)

    history.append({"role": "assistant", "content":result.final_output})
    cl.user_session.set("history", history)
    
    # await cl.Message(content=result.final_output).send()

