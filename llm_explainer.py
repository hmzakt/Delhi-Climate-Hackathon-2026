"""
llm_explainer.py — AI Explanation System for Pollution Spikes

Handles all LLM-related processing for generating explanations.
Separated from main.py for better code organization.
"""

import os
from typing import Optional, Dict, Any, Tuple
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# ═════════════════════════════════════════════════════════════════
# LLM CONFIGURATION
# ═════════════════════════════════════════════════════════════════

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# Ensure API key is in environment for LangChain
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


def get_llm():
    """Initialize and return Google Gemini LLM."""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not set in environment")
    
    # LangChain ChatGoogleGenerativeAI reads from GOOGLE_API_KEY env var automatically
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )


# ═════════════════════════════════════════════════════════════════
# CONTEXT PREPARATION
# ═════════════════════════════════════════════════════════════════

def prepare_pollution_context(
    station_name: str,
    datetime_ist: pd.Timestamp,
    pm25: float,
    no2: Optional[float],
    pm10: Optional[float],
    temperature: float,
    humidity: float,
    wind_speed: float,
    wind_direction: float,
    surface_pressure: Optional[float],
    precipitation: Optional[float],
    cloud_cover: Optional[float],
    fire_count: int = 0,
    frp: float = 0.0,
    max_frp: float = 0.0,
    upwind_fire: int = 0,
    is_diwali: int = 0,
    month: int = 0,
    day: int = 0,
    day_of_week: int = 0,
    season: int = 0,
    is_weekend: int = 0,
    is_night: int = 0,
    is_rush_hour: int = 0,
    stagnation_index: Optional[float] = None,
    dispersion_index: Optional[float] = None,
    label_predicted: str = "unknown"
) -> Dict[str, Any]:
    """
    Prepare a comprehensive context dictionary for AI explanation.
    Includes all relevant temporal, meteorological, and pollution data.
    """
    # Format datetime in readable format
    datetime_str = datetime_ist.strftime("%Y-%m-%d %H:%M IST")
    date_str = datetime_ist.strftime("%Y-%m-%d")
    time_str = datetime_ist.strftime("%H:%M")
    
    # Day names
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_name = day_names[day_of_week] if 0 <= day_of_week < 7 else "Unknown"
    
    # Season names
    season_names = {0: "Winter", 1: "Spring", 2: "Monsoon", 3: "Post-Monsoon"}
    season_name = season_names.get(season, "Unknown")
    
    # Wind direction description
    wind_dirs = {
        (0, 22.5): "N", (22.5, 67.5): "NE", (67.5, 112.5): "E", (112.5, 157.5): "SE",
        (157.5, 202.5): "S", (202.5, 247.5): "SW", (247.5, 292.5): "W", (292.5, 337.5): "NW",
        (337.5, 360): "N"
    }
    wind_dir_desc = next((desc for (low, high), desc in wind_dirs.items() 
                          if low <= wind_direction < high), "Unknown")
    
    return {
        "location": station_name.replace(", Delhi - DPCC", ""),
        "datetime": datetime_str,
        "date": date_str,
        "time": time_str,
        "pm25": round(pm25, 1),
        "no2": round(no2, 1) if no2 is not None else None,
        "pm10": round(pm10, 1) if pm10 is not None else None,
        "temperature": round(temperature, 1),
        "humidity": round(humidity, 1),
        "wind_speed": round(wind_speed, 1),
        "wind_direction": round(wind_direction, 0),
        "wind_direction_cardinal": wind_dir_desc,
        "surface_pressure": round(surface_pressure, 1) if surface_pressure is not None else None,
        "precipitation": round(precipitation, 2) if precipitation is not None else None,
        "cloud_cover": round(cloud_cover, 1) if cloud_cover is not None else None,
        "fire_count": fire_count,
        "frp": round(frp, 1),
        "max_frp": round(max_frp, 1),
        "upwind_fire": upwind_fire,
        "is_diwali": is_diwali,
        "hour": datetime_ist.hour,
        "month": month,
        "day": day,
        "day_of_week": day_of_week,
        "day_name": day_name,
        "season": season,
        "season_name": season_name,
        "is_weekend": is_weekend,
        "is_night": is_night,
        "is_rush_hour": is_rush_hour,
        "stagnation_index": round(stagnation_index, 1) if stagnation_index is not None else None,
        "dispersion_index": round(dispersion_index, 1) if dispersion_index is not None else None,
        "label_predicted": label_predicted.replace("_", " ").title()
    }


# ═════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ═════════════════════════════════════════════════════════════════

def get_authority_prompt(context: Dict[str, Any]) -> Tuple[str, str]:
    """Craft system + role prompt for Authorities."""
    system_prompt = """You are an environmental policy advisor with expertise in air quality management, atmospheric science, and pollution source attribution. Your role is to provide actionable insights for regulatory bodies and policymakers."""
    
    # Build comprehensive data section
    data_section = f"""Location: {context['location']}
Date: {context['date']} ({context['day_name']})
Time: {context['time']} IST (Hour: {context['hour']}:00)
Season: {context['season_name']}
Is Weekend: {'Yes' if context['is_weekend'] == 1 else 'No'}
Is Night: {'Yes' if context['is_night'] == 1 else 'No'}
Is Rush Hour: {'Yes' if context['is_rush_hour'] == 1 else 'No'}
Is Diwali Period: {'Yes' if context['is_diwali'] == 1 else 'No'}

POLLUTION MEASUREMENTS:
PM2.5: {context['pm25']} µg/m³
PM10: {context['pm10'] if context['pm10'] is not None else 'N/A'} µg/m³
NO2: {context['no2'] if context['no2'] is not None else 'N/A'} µg/m³

METEOROLOGICAL CONDITIONS:
Temperature: {context['temperature']}°C
Humidity: {context['humidity']}%
Wind Speed: {context['wind_speed']} km/h
Wind Direction: {context['wind_direction']}° ({context['wind_direction_cardinal']})
Surface Pressure: {context['surface_pressure'] if context['surface_pressure'] is not None else 'N/A'} hPa
Precipitation: {context['precipitation'] if context['precipitation'] is not None else 'N/A'} mm
Cloud Cover: {context['cloud_cover'] if context['cloud_cover'] is not None else 'N/A'}%
Stagnation Index: {context['stagnation_index'] if context['stagnation_index'] is not None else 'N/A'}
Dispersion Index: {context['dispersion_index'] if context['dispersion_index'] is not None else 'N/A'}

FIRE ACTIVITY:
Fire Count (within 100km): {context['fire_count']}
Total Fire Radiative Power: {context['frp']} MW
Max Fire Radiative Power: {context['max_frp']} MW
Upwind Fire Detected: {'Yes' if context['upwind_fire'] == 1 else 'No'}

PREDICTED SOURCE: {context['label_predicted']}"""
    
    user_prompt = f"""A pollution spike occurred with the following comprehensive data:

{data_section}

Please provide a comprehensive analysis:

1. **Most Likely Cause**: What is the primary driver of this pollution spike? Explain the evidence based on the meteorological conditions, temporal factors, and pollution measurements.

2. **Underlying Factors**: Break down the contributing factors:
   - Meteorological conditions (temperature inversion, wind patterns, humidity effects, pressure systems)
   - Emission sources (traffic patterns during rush hour, industrial activity, biomass burning, fire activity)
   - Temporal factors (time of day, day of week, season, special events like Diwali)
   - Atmospheric stability (stagnation index, dispersion conditions)

3. **Policy Recommendations**: What preventive or enforcement actions should authorities consider?
   - Immediate actions (if applicable)
   - Long-term policy measures
   - Monitoring and enforcement priorities
   - Public health advisories

Keep the response concise but thorough (300-400 words). Use clear, professional language suitable for policy briefings."""
    
    return system_prompt, user_prompt


def get_public_prompt(context: Dict[str, Any]) -> Tuple[str, str]:
    """Craft system + role prompt for Public Users."""
    system_prompt = """You are a friendly environmental educator explaining air pollution to citizens. Use simple, clear language without jargon. Be empathetic and actionable."""
    
    # Determine air quality level
    pm25_level = "Very High" if context['pm25'] > 150 else "High" if context['pm25'] > 100 else "Moderate" if context['pm25'] > 50 else "Low"
    
    user_prompt = f"""A pollution spike occurred in {context['location']} on {context['date']} at {context['time']} IST ({context['day_name']}). Here's what we know:

AIR QUALITY:
- PM2.5 Level: {context['pm25']} µg/m³ ({pm25_level})
{f"- PM10 Level: {context['pm10']} µg/m³" if context['pm10'] is not None else ""}
{f"- NO2 Level: {context['no2']} µg/m³" if context['no2'] is not None else ""}

WEATHER CONDITIONS:
- Temperature: {context['temperature']}°C
- Humidity: {context['humidity']}%
- Wind: {context['wind_speed']} km/h from {context['wind_direction_cardinal']} ({context['wind_direction']}°)
{f"- Rain: {context['precipitation']} mm" if context['precipitation'] and context['precipitation'] > 0 else ""}

CONTEXT:
- Time: {context['hour']}:00 ({'Night' if context['is_night'] == 1 else 'Day'})
- Day: {context['day_name']} ({'Weekend' if context['is_weekend'] == 1 else 'Weekday'})
- Season: {context['season_name']}
- Rush Hour: {'Yes' if context['is_rush_hour'] == 1 else 'No'}
- Diwali Period: {'Yes' if context['is_diwali'] == 1 else 'No'}

FIRE ACTIVITY:
- Fires Nearby: {context['fire_count']} {'fire(s) detected within 100km' if context['fire_count'] > 0 else 'none detected'}
{f"- Upwind Fire: Yes (smoke may be blowing toward the area)" if context['upwind_fire'] == 1 else ""}

LIKELY CAUSE: {context['label_predicted']}

Please explain:

1. **What Caused This Spike?** 
   - Was it due to fire, traffic, weather conditions, or festive celebrations?
   - Explain in simple terms what's happening and why the air quality is {pm25_level.lower()}

2. **What Can I Do Right Now?**
   - Should I avoid outdoor activities?
   - Should I use an air purifier?
   - Any other immediate precautions?
   - Who should be most careful? (children, elderly, people with breathing problems)

3. **Is This Normal?**
   - Is this typical for this time of year ({context['season_name']}) in {context['location']}?
   - How long might it last?
   - Should I be worried?

Keep it friendly, conversational, and under 250 words. Focus on practical advice that helps people protect themselves."""
    
    return system_prompt, user_prompt


# ═════════════════════════════════════════════════════════════════
# LLM CALL FUNCTION
# ═════════════════════════════════════════════════════════════════

def call_llm(system_prompt: str, user_prompt: str) -> Optional[str]:
    """
    Call Google Gemini LLM to generate explanation.
    Falls back gracefully if API is unavailable.
    """
    if not GOOGLE_API_KEY:
        print("Warning: GOOGLE_API_KEY not set. Using fallback explanation.")
        return None
    
    try:
        llm = get_llm()
        
        # Combine system and user prompts for Gemini
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Call Gemini
        response = llm.invoke(full_prompt)
        
        # Extract content from LangChain AIMessage response
        # LangChain returns an AIMessage object with .content attribute
        if hasattr(response, 'content'):
            content = response.content
            if isinstance(content, str):
                return content
            else:
                return str(content)
        elif isinstance(response, str):
            return response
        else:
            # Try to get string representation
            return str(response)
    
    except Exception as e:
        import traceback
        error_msg = f"Google Gemini API error: {e}\n{traceback.format_exc()}"
        print(error_msg)
        return None


# ═════════════════════════════════════════════════════════════════
# FALLBACK EXPLANATION
# ═════════════════════════════════════════════════════════════════

def get_fallback_explanation(context: Dict[str, Any]) -> str:
    """Generate a basic fallback explanation when LLM is unavailable."""
    return f"""Based on the data for {context['location']} at {context['datetime']}:

**Predicted Source**: {context['label_predicted']}

**Key Observations**:
- PM2.5 level: {context['pm25']} µg/m³
- Wind conditions: {context['wind_speed']} km/h
- Temperature: {context['temperature']}°C
- Humidity: {context['humidity']}%

This pollution spike appears to be primarily driven by {context['label_predicted'].lower()} conditions. {'Fire activity detected nearby may also be contributing.' if context['fire_count'] > 0 else ''}

**Recommendations**: {'Avoid outdoor activities, especially for sensitive groups.' if context['pm25'] > 100 else 'Air quality is moderate. Limit prolonged outdoor exposure.'}

*Note: AI explanation service is currently unavailable. This is a basic fallback explanation.*"""
