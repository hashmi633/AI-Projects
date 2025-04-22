from dotenv import load_dotenv
import os
import chainlit as cl 
from agents import AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, Agent, Runner
from typing import cast

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

@cl.on_chat_start
async def start():
    external_client = AsyncOpenAI(
        api_key= gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=external_client
    )
    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )

    cl.user_session.set('chat_history', [])
    cl.user_session.set("config", config)

    spanish_agent = Agent(
        name="Spanish agent",
        instructions="You translate the user's message to Spanish",
        handoff_description="An english to spanish translator",
                model= model,
        )
    
    french_agent = Agent(
        name="French agent",
        instructions="You translate the user's message to French",
        handoff_description="An english to french translator",
                model= model,
    )

    urdu_agent = Agent(
        name="Urdu Agent",
        instructions="You translate the user's message to Urdu",
        handoff_description="An english to urdu translator",
                model= model,
    )

    orchestrator_agent = Agent(
        name="orchestrator agent",
        instructions=(
            "You are a translation agent. You use the tools given to you to translate."
            "You never translate on your own, you always use the provided tools."
            "Focus only on the most recent user message for translation."
            # "If asked for multiple translations, you call the relevant tools in order."
            "Do not include or reference previous translations unless the user explicitly requests them."
            ),
        tools=[
            spanish_agent.as_tool(
                tool_name="translate_to_spanish",
                tool_description="Translate user's message to Spanish",
                
            ),
            french_agent.as_tool(
                tool_name="translate_to_french",
                tool_description="Translate the user's message to French",
                
            ),
            urdu_agent.as_tool(
                tool_name="translate_to_urdu",
                tool_description="Translate the user's message to Urdu",
                
            )
        ],
        model=model
        )
    cl.user_session.set("agent", orchestrator_agent)
    msg = cl.Message(content="Hi! What would you like translated, and to which languages? ")
    await msg.send()

@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="Translating...")
    await msg.send()

    history = cl.user_session.get("chat_history")
    latest_message = [{"role": "user", "content": message.content}]

    config = cast(RunConfig, cl.user_session.get("config"))
    agent = cast(Agent, cl.user_session.get("agent"))

    try:
        result = await Runner.run(
            agent,
            message.content,
            run_config=config
        )
        response_content = result.final_output
        msg.content = response_content
        await msg.update()

        history.append({"role": "user", "content": message.content})
        history.append({"role": "developer", "content": response_content})
        cl.user_session.set("chat_history", history)

        print(f"User: {message.content}")
        print(f"Agent: {response_content}")
    except Exception as e:
        msg.content = f"Error: {str(e)}"
        await msg.update()