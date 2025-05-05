import requests, os
from typing import Optional
from dataclasses import dataclass
from agents import function_tool

@dataclass
class WeatherInfo:
    temperature : float
    feels_like: float
    humidity: int
    description: str
    wind_speed: float
    pressure: int
    location_name: str
    rain_1hr: Optional[float] = None
    visibility: Optional[int] = None

@function_tool
def get_weather(lat: float, lon: float)->str:
    """Get the current weather for a specified location using OpenWeatherMap API.
    Args:
       lat: Latitude of the location (-90 to 90)
       lon: Longitude of the location (-180 to 180)    
    """
    # Get API key from environment variables
    WEATHER_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

    # Build URL with parameters
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        # print(data)

        # Extract weather data from the response
        weather_info = WeatherInfo(
            temperature=data["main"]["temp"],
            feels_like=data["main"]["feels_like"],
            description=data["weather"][0]["description"],
            humidity=data["main"]["humidity"],
            wind_speed=data["wind"]["speed"],
            pressure=data["main"]["pressure"],
            location_name=data["name"],
            rain_1hr=data.get("rain", {}).get("1h"),
            visibility=data.get("visibility"),
        )

        # Build the response string
        weather_report = f"""
        Weather in {weather_info.location_name}:
        - Temperature: {weather_info.temperature}°C (feel like {weather_info.feels_like}°C)
        - Conditions: {weather_info.description}
        - Humidity: {weather_info.humidity}%
        - Wind speed: {weather_info.wind_speed} m/s
        - Pressure: {weather_info.pressure} hPa
        """
        # print(weather_report)
        return weather_report
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {str(e)}"
