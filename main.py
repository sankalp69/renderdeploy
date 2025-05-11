import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import datetime
import logging

# Ensure the necessary tools are available (if any are used by the model implicitly)
# from google.generativeai import GenerativeModel # Example if needed for tool definition

# --- Configuration ---

# Configure logging (optional but helpful)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Fetch the API key from environment variables
api_key = os.getenv("GOOGLE_API_KEY")

# Flag to track if API key is configured successfully
api_configured = False

if not api_key:
    st.error("⚠️ Error: GOOGLE_API_KEY not found in environment variables. Please create a `.env` file with your key.")
    logging.error("Error: GOOGLE_API_KEY not found in environment variables.")
else:
    try:
        # Configure the generative AI client
        genai.configure(api_key=api_key)
        api_configured = True
        logging.info("Google Generative AI configured successfully.")
    except Exception as e:
        st.error(f"⚠️ Error configuring Google Generative AI: {e}")
        logging.error(f"Error configuring Google Generative AI: {e}")

# --- Helper Functions ---

def get_budget_description(budget_level):
    """Maps budget slider value to a descriptive string."""
    if budget_level == 1:
        return "Budget-Friendly"
    elif budget_level == 2:
        return "Mid-Range"
    elif budget_level == 3:
        return "Luxury"
    else:
        return "Any Budget" # Default or unexpected case

def generate_flight_suggestions(source, destination, start_date, end_date, budget_level_desc, model_name="gemini-1.5-flash"):
    """
    Generates flight suggestions using the Gemini model based on a prompt.
    Note: These are AI-generated suggestions, not real-time flight data.

    Args:
        source (str): The origin city/airport.
        destination (str): The destination city/airport.
        start_date (datetime.date): The desired departure date.
        end_date (datetime.date): The desired return date.
        budget_level_desc (str): Description of the budget level.
        model_name (str): The name of the Gemini model to use.

    Returns:
        str: The generated flight suggestions text, or an error message.
    """
    if not api_configured:
        return "API not configured for flight suggestions. Cannot generate suggestions."

    try:
        # Prompt Template for flight suggestions
        prompt = f"""
        As a travel planning AI, suggest potential flight options for a trip from {source} to {destination}.
        The desired departure date is {start_date.strftime('%Y-%m-%d')} and the return date is {end_date.strftime('%Y-%m-%d')}.
        Please provide suggestions that align with a **{budget_level_desc} budget**.

        Suggest a few possible airlines, potential layover cities (if applicable), and a general idea of what one might expect regarding flight duration or typical costs for this route and budget.
        Emphasize that these are *suggestions based on general knowledge* and that users should perform a real-time flight search for accurate prices and availability.

        Present the response clearly using Markdown.
        """

        logging.info(f"Generating {budget_level_desc} flight suggestions from {source} to {destination} using {model_name}...")

        # Initialize the model
        model = genai.GenerativeModel(model_name=model_name)

        # Set generation config (optional, adjust as needed)
        generation_config = genai.types.GenerationConfig(
            temperature=0.6, # Adjust for creativity vs. predictability
            max_output_tokens=700 # Adjust token limit
        )

        # Call the API
        response = model.generate_content(
            contents=[prompt],
            generation_config=generation_config
        )

        # Check for safety issues or empty response
        if response.parts:
            logging.info("Flight suggestions generated successfully.")
            return response.text
        else:
            logging.warning("Received an empty response or content was blocked.")
            return f"Could not generate flight suggestions. The response was empty or blocked. (Feedback: {response.prompt_feedback})"

    except Exception as e:
        logging.error(f"An error occurred during flight suggestion generation: {e}")
        return f"An error occurred: {e}"


def generate_travel_itinerary(destination, start_date, end_date, budget_level_desc, model_name="gemini-1.5-flash"):
    """
    Generates a travel itinerary using the Gemini model, considering budget.

    Args:
        destination (str): The travel destination.
        start_date (datetime.date): The start date of the trip.
        end_date (datetime.date): The end date of the trip.
        budget_level_desc (str): Description of the budget level (e.g., "Budget-Friendly").
        model_name (str): The name of the Gemini model to use.

    Returns:
        str: The generated itinerary text, or an error message.
    """
    if not api_configured:
        return "API not configured for itinerary generation. Cannot generate itinerary."

    try:
        # Calculate duration
        duration = (end_date - start_date).days + 1 # Include both start and end dates

        # Construct the prompt for the AI model, including budget
        prompt = f"""Create a detailed travel itinerary for a trip to {destination}.
        The trip starts on {start_date.strftime('%Y-%m-%d')} and ends on {end_date.strftime('%Y-%m-%d')}, lasting for {duration} days.
        Please plan the trip with a **{budget_level_desc} budget** in mind.

        Provide a day-by-day plan including:
        - Suggested activities for morning, afternoon, and evening (suitable for a {budget_level_desc} budget).
        - Recommendations for places to visit (landmarks, museums, parks, etc.) - mention cost implications if relevant to the budget.
        - Optional: Suggestions for local food or restaurants to try that fit a {budget_level_desc} budget.
        - Optional: Basic tips for getting around (e.g., public transport, walking) that are budget-conscious.

        Format the output clearly, perhaps using Markdown with headings for each day.
        Be creative and provide practical suggestions for a memorable trip.
        """

        logging.info(f"Generating {budget_level_desc} itinerary for {destination} from {start_date} to {end_date} using {model_name}...")

        # Initialize the model
        model = genai.GenerativeModel(model_name=model_name)

        # Set generation config (optional, adjust as needed)
        generation_config = genai.types.GenerationConfig(
            temperature=0.7, # Adjust for creativity vs. predictability
            max_output_tokens=2048 # Increase token limit for longer itineraries
        )

        # Call the API
        response = model.generate_content(
            contents=[prompt],
            generation_config=generation_config
        )

        # Check for safety issues or empty response
        if response.parts:
            logging.info("Itinerary generated successfully.")
            return response.text
        else:
            logging.warning("Received an empty response or content was blocked.")
            # Inspect feedback if needed: feedback = response.prompt_feedback
            return f"Could not generate itinerary. The response was empty or blocked. (Feedback: {response.prompt_feedback})"

    except Exception as e:
        logging.error(f"An error occurred during itinerary generation: {e}")
        return f"An error occurred: {e}"

def generate_recommendations(location, budget_level_desc, model_name="gemini-1.5-flash"):
     """
     Generates restaurant and hotel recommendations using the Gemini model, considering budget.

     Args:
         location (str): The location for recommendations.
         budget_level_desc (str): Description of the budget level (e.g., "Budget-Friendly").
         model_name (str): The name of the Gemini model to use.

     Returns:
         str: The generated recommendations text, or an error message.
     """
     if not api_configured:
         return "API not configured for recommendations. Cannot generate recommendations."

     try:
         # Prompt Template for recommendations, including budget
         prompt = f"""
         You are an expert Restaurant & Hotel Planner.
         Your job is to provide Restaurant & Hotel recommendations for {location}.
         Please provide recommendations specifically for a **{budget_level_desc} budget**.

         - For Restaurants: Provide Top 5 restaurants that fit a {budget_level_desc} budget, with address and a general idea of average cost or cuisine type. Include a rating if available or inferable.
         - For Hotels: Provide Top 5 hotels that fit a {budget_level_desc} budget, with address and a general idea of average cost per night or star rating. Include a rating if available or inferable.

         Return the response using Markdown for clear formatting.
         """

         logging.info(f"Generating {budget_level_desc} recommendations for {location} using {model_name}...")

         # Initialize the model
         model = genai.GenerativeModel(model_name=model_name)

         # Set generation config (optional, adjust as needed)
         generation_config = genai.types.GenerationConfig(
             temperature=0.7, # Adjust for creativity vs. predictability
             max_output_tokens=2048 # Increase token limit
         )

         # Call the API
         response = model.generate_content(
             contents=[prompt],
             generation_config=generation_config
         )

         # Check for safety issues or empty response
         if response.parts:
             logging.info("Recommendations generated successfully.")
             return response.text
         else:
             logging.warning("Received an empty response or content was blocked.")
             return f"Could not generate recommendations. The response was empty or blocked. (Feedback: {response.prompt_feedback})"

     except Exception as e:
         logging.error(f"An error occurred during recommendation generation: {e}")
         return f"An error occurred: {e}"

def get_weather_forecast(location, model_name="gemini-1.5-flash"):
     """
     Gets a weather forecast and clothing suggestions using the Gemini model based on a prompt.

     Args:
         location (str): The location for the weather forecast.
         model_name (str): The name of the Gemini model to use.

     Returns:
         str: The generated weather forecast and clothing suggestions text, or an error message.
     """
     if not api_configured:
         return "API not configured for weather forecasting. Cannot get weather forecast or clothing suggestions."

     try:
         # Prompt Template for weather forecasting and clothing suggestions
         prompt = f"""
         You are an expert weather forecaster and travel advisor. Your job is to provide a detailed weather forecast and suggest appropriate clothing to pack for a trip to {location}.
         Provide the forecast for the next 7 days, starting from today's date.
         Include details such as:
         - Daily temperature range (High/Low)
         - Precipitation (chance of rain/snow)
         - Humidity
         - Wind conditions
         - Air Quality (if available or inferable)
         - Cloud Cover

         Based on this 7-day forecast, provide a clear and concise suggestion for the type of clothing and gear someone should pack for their trip to {location} during this period. Consider layering if temperatures vary.

         Present the response clearly using Markdown, with a section for the daily forecast and a separate section for clothing suggestions.
         """

         logging.info(f"Getting weather forecast and clothing suggestions for {location} using {model_name}...")

         # Initialize the model
         model = genai.GenerativeModel(model_name=model_name)

         # Set generation config (optional, adjust as needed)
         generation_config = genai.types.GenerationConfig(
             temperature=0.4, # Lower temperature for more factual/less creative output
             max_output_tokens=1500 # Increased token limit to accommodate suggestions
         )

         # Call the API
         response = model.generate_content(
             contents=[prompt],
             generation_config=generation_config
         )

         # Check for safety issues or empty response
         if response.parts:
             logging.info("Weather forecast and clothing suggestions generated successfully.")
             return response.text
         else:
             logging.warning("Received an empty response or content was blocked.")
             return f"Could not get weather forecast and clothing suggestions. The response was empty or blocked. (Feedback: {response.prompt_feedback})"

     except Exception as e:
         logging.error(f"An error occurred during weather forecasting and clothing suggestions: {e}")
         return f"An error occurred: {e}"


# --- Streamlit UI ---

st.set_page_config(page_title="AI Ultimate Travel Planner", layout="wide")

st.title("🌍 AI Ultimate Travel Planner ✨")
st.markdown("Enter your travel details below and get your full trip plan!")

# Input fields for Source, Destination, and Dates
col1, col2 = st.columns(2)

with col1:
    source = st.text_input("🛫 Source (City/Airport):", placeholder="e.g., New York")
    destination = st.text_input("📍 Destination (City/Region):", placeholder="e.g., Paris")

with col2:
    # Default start date is today, end date is 7 days from today
    today = datetime.date.today()
    start_date = st.date_input("📅 Start Date:", value=today, min_value=today)

    default_end_date = today + datetime.timedelta(days=6)
    end_date = st.date_input("📅 End Date:", value=default_end_date, min_value=start_date)

# Budget Slider
st.markdown("---")
st.subheader("Budget Preference")
budget_level = st.slider(
    "💰 Select your budget level:",
    min_value=1,
    max_value=3,
    value=2, # Default to Mid-Range
    step=1
    # format_func=get_budget_description # Removed as per previous fix
)
budget_level_desc = get_budget_description(budget_level)
st.info(f"You selected: **{budget_level_desc}**")


# Single button to trigger all planning
st.markdown("---")
if st.button("✈️🏨🍔☁️ Plan My Trip! ☁️🍔🏨✈️", disabled=not api_configured):
    # --- Input Validation ---
    if not source:
        st.warning("Please enter a source destination for flight suggestions.")
    elif not destination:
        st.warning("Please enter a destination to plan your trip.")
    elif start_date > end_date:
        st.error("Error: End date cannot be before the start date.")
    elif not api_configured:
        st.error("API is not configured. Cannot plan your trip.")
    else:
        st.subheader(f"Planning Your Trip from {source} to {destination}...")
        st.markdown("---")

        # --- Generate and Display Flight Suggestions ---
        st.subheader("✈️ Flight Suggestions")
        st.caption("Note: These are AI-generated suggestions based on general knowledge, not real-time flight data.")
        with st.spinner(f"Getting {budget_level_desc} flight suggestions from {source} to {destination}..."):
            flight_suggestions_result = generate_flight_suggestions(source, destination, start_date, end_date, budget_level_desc)
            st.markdown("---")
            st.markdown(flight_suggestions_result)

        st.markdown("---") # Separator between sections

        # --- Generate and Display Itinerary ---
        with st.spinner(f"Generating your {budget_level_desc} itinerary for {destination}..."):
            itinerary_result = generate_travel_itinerary(destination, start_date, end_date, budget_level_desc)
            st.subheader("📝 Your Travel Itinerary")
            st.markdown(f"**Duration:** {start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}")
            st.markdown(f"**Budget:** {budget_level_desc}") # Display selected budget
            st.markdown("---")
            st.markdown(itinerary_result)

        st.markdown("---") # Separator between sections

        # --- Generate and Display Recommendations ---
        with st.spinner(f"Getting {budget_level_desc} restaurant & hotel recommendations for {destination}..."):
            recommendations_result = generate_recommendations(destination, budget_level_desc)
            st.subheader("🏨 Restaurant & Hotel Recommendations 🍔")
            st.markdown(f"**Budget:** {budget_level_desc}") # Display selected budget
            st.markdown("---")
            st.markdown(recommendations_result)

        st.markdown("---") # Separator between sections

        # --- Get and Display Weather Forecast ---
        # Weather forecast does not typically depend on budget
        with st.spinner(f"Fetching weather forecast and clothing suggestions for {destination}..."):
             weather_result = get_weather_forecast(destination)
             st.subheader("☁️ Weather Forecast and Clothing Suggestions ☀️")
             st.markdown("---")
             st.markdown(weather_result)

# Add a footer or instructions (optional)
st.markdown("---")
st.caption("Powered by Google Gemini for planning, recommendations, weather forecasting, and flight *suggestions*.") # Updated caption
st.caption("Ensure your GOOGLE_API_KEY is set in a `.env` file.")
st.caption("For real-time flight information, please use a dedicated flight search engine.")

st.markdown("Made with ❤️ by sankalp patekar")