import os
from agents import AsyncOpenAI,Agent, OpenAIChatCompletionsModel, RunConfig, Runner, ModelProvider
from dotenv import load_dotenv
from weather_tool import get_weather
import asyncio
from typing import cast

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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

async def main():
    try:
        result = await Runner.run(
            weather_assistant,
            "what's the weather like in newyork, USA",
            run_config=config,
        )
        print(result.final_output)
    except Exception as e:
        return f"Error: {str(e)}"
    
if __name__ == "__main__":
    asyncio.run(main())