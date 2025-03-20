import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Annotated, TypedDict

import openai
import requests
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from PIL import Image

sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv

# Now import from app.src
from app.src.embedding.model import EmbeddingModel
from app.src.rag.chain import RAGChain
from app.src.rag.document_loader import GridCodeLoader
from app.src.rag.vectorstore import VectorStore

# Load .env file from base directory
load_dotenv(Path(__file__).parent / ".env")
logger = logging.getLogger(__name__)


def get_secrets():
    """Get secrets from environment variables."""
    # Skip trying Streamlit secrets and go straight to environment variables
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "LANGCHAIN_API_KEY": os.getenv("LANGCHAIN_API_KEY"),
        "LANGCHAIN_PROJECT": os.getenv("LANGCHAIN_PROJECT", "GridGuide"),
        "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2", "true"),
    }


# Set up environment variables from secrets
secrets = get_secrets()
for key, value in secrets.items():
    if value:
        os.environ[key] = value

# Verify API keys without showing warning
if not os.getenv("OPENAI_API_KEY"):
    st.error("OpenAI API key not found. Please check your .env file.")
    st.stop()


class WeatherTool:
    def __init__(self):
        self.base_url = "https://api.weather.gov"
        self.headers = {
            "User-Agent": "(Grid Code Assistant, contact@example.com)",
            "Accept": "application/json",
        }

    def get_coordinates_from_zip(self, zipcode):
        response = requests.get(f"https://api.zippopotam.us/us/{zipcode}")
        if response.status_code == 200:
            data = response.json()
            return {
                "lat": data["places"][0]["latitude"],
                "lon": data["places"][0]["longitude"],
                "place": data["places"][0]["place name"],
                "state": data["places"][0]["state"],
            }
        return None

    def run(self, zipcode):
        coords = self.get_coordinates_from_zip(zipcode)
        if not coords:
            return {"error": "Invalid ZIP code or unable to get coordinates."}

        point_url = f"{self.base_url}/points/{coords['lat']},{coords['lon']}"
        response = requests.get(point_url, headers=self.headers)

        if response.status_code != 200:
            return {"error": "Unable to fetch weather data."}

        grid_data = response.json()
        forecast_url = grid_data["properties"]["forecast"]

        response = requests.get(forecast_url, headers=self.headers)
        if response.status_code == 200:
            forecast_data = response.json()["properties"]["periods"]
            weather_data = {
                "type": "weather",
                "location": f"{coords['place']}, {coords['state']}",
                "current": forecast_data[0],
                "forecast": forecast_data[1:4],
            }
            # Save to session state
            st.session_state.weather_data = weather_data
            return weather_data
        return {"error": "Unable to fetch forecast data."}


def initialize_rag():
    """Initialize RAG system."""
    if "rag_chain" in st.session_state:
        logger.info("Using cached RAG chain from session state")
        return st.session_state.rag_chain

    # Try multiple possible paths for the PDF
    possible_paths = [
        "Grid_Code.pdf",  # Base directory (local and Docker)
        "/app/Grid_Code.pdf",  # Docker container path
        Path(__file__).parent / "Grid_Code.pdf",  # Absolute path
    ]

    data_path = None
    for path in possible_paths:
        if isinstance(path, str):
            path = Path(path)
        logger.info(f"Checking path: {path}")
        if path.exists():
            data_path = str(path)
            logger.info(f"Found PDF at: {data_path}")
            break

    if not data_path:
        raise FileNotFoundError(
            f"PDF not found in any of these locations: {possible_paths}"
        )

    with st.spinner("Loading Grid Code documents..."):
        loader = GridCodeLoader(data_path, pages=25)
        documents = loader.load_and_split()
        logger.info(f"Loaded {len(documents)} document chunks")

    with st.spinner("Creating vector store..."):
        embedding_model = EmbeddingModel()
        vectorstore = VectorStore(embedding_model)
        vectorstore = vectorstore.create_vectorstore(documents)
        logger.info("Vector store created successfully")

    # Cache the RAG chain in session state
    rag_chain = RAGChain(vectorstore)
    st.session_state.rag_chain = rag_chain
    return rag_chain


class RAGTool:
    def __init__(self, rag_chain):
        self.rag_chain = rag_chain

    def run(self, question: str) -> str:
        """Answer questions using the Grid Code."""
        response = self.rag_chain.invoke(question)
        return response["answer"]


class AgentState(TypedDict):
    """State definition for the agent."""

    messages: Annotated[list, add_messages]


def create_agent_workflow(rag_chain, weather_tool):
    """Create an agent that can use both RAG and weather tools."""

    # Define the tools
    tools = [
        Tool(
            name="grid_code_query",
            description="Answer questions about the Grid Code and electrical regulations",
            func=lambda q: rag_chain.invoke(q)["answer"],
        ),
        Tool(
            name="get_weather",
            description="Get weather forecast for a ZIP code. Input should be a 5-digit ZIP code.",
            func=lambda z: weather_tool.run(z),
        ),
    ]

    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Create the custom prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                """You are a helpful assistant that specializes in two areas:
            1. Answering questions about electrical Grid Code regulations
            2. Providing weather information for specific locations

            For weather queries:
            - Extract the ZIP code from the question
            - Use the get_weather tool to fetch the forecast

            For Grid Code questions:
            - Use the grid_code_query tool to find relevant information
            - If the information isn't in the Grid Code, clearly state that
            - Provide specific references when possible
            """
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Create the agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )


def display_weather(weather_data):
    """Display weather information in a nice format"""
    if "error" in weather_data:
        st.error(weather_data["error"])
        return

    if weather_data.get("type") == "weather":
        # Location header
        st.header(f"Weather for {weather_data['location']}")

        # Current conditions
        current = weather_data["current"]
        st.subheader("Current Conditions")

        # Use columns for current weather layout
        col1, col2 = st.columns(2)

        with col1:
            # Temperature display with metric
            st.metric(
                "Temperature", f"{current['temperature']}°{current['temperatureUnit']}"
            )
            # Wind information
            st.info(f"💨 Wind: {current['windSpeed']} {current['windDirection']}")

        with col2:
            # Current forecast
            st.markdown(f"**🌤️ Conditions:** {current['shortForecast']}")
            st.markdown(f"**📝 Details:** {current['detailedForecast']}")

        # Extended forecast
        st.subheader("Extended Forecast")
        for period in weather_data["forecast"]:
            with st.expander(f"📅 {period['name']}"):
                st.markdown(
                    f"**🌡️ Temperature:** {period['temperature']}°{period['temperatureUnit']}"
                )
                st.markdown(
                    f"**💨 Wind:** {period['windSpeed']} {period['windDirection']}"
                )
                st.markdown(f"**🌤️ Forecast:** {period['shortForecast']}")
                st.markdown(f"**📝 Details:** {period['detailedForecast']}")


def main():
    image = Image.open("app/src/data/logo.png")
    st.image(image, use_container_width=True)

    # Initialize if not in session state
    if "app" not in st.session_state:
        rag_chain = initialize_rag()
        weather_tool = WeatherTool()
        st.session_state.app = create_agent_workflow(rag_chain, weather_tool)

    # Initialize session states if not present
    if "response" not in st.session_state:
        st.session_state.response = None

    # Initialize ALL state variables in a single place for clarity
    for key in [
        "transcription",
        "process_input",
        "audio_recorded",
        "audio_bytes",
        "clear_feedback",
    ]:
        if key not in st.session_state:
            st.session_state[key] = None

    # Create input area
    input_container = st.container()
    with input_container:
        st.write("Type your question:")

        # Create columns for input field, send button, and mic button
        col1, col2, col3 = st.columns([4, 1, 1])

        with col1:
            # Text input
            user_input = st.text_input(
                "",
                key="typed_input",
                label_visibility="collapsed",
            )

        with col2:
            # Send button for text input
            send_pressed = st.button("Send")

        with col3:
            # Audio recorder inline with input
            new_audio_bytes = audio_recorder(text="", icon_size="2x")

            # If new audio is recorded, immediately reset clear_feedback flag
            # This ensures transcription will be shown right away
            if new_audio_bytes:
                st.session_state.clear_feedback = False

    # Full width container for feedback (transcription, spinner, etc.)
    feedback_container = st.empty()  # Use st.empty() for easy clearing

    # Handle new audio recording - always process if there's new audio
    if new_audio_bytes:
        # Store audio for processing
        st.session_state.audio_bytes = new_audio_bytes
        st.session_state.audio_recorded = True

        with feedback_container.container():
            st.audio(new_audio_bytes, format="audio/wav")

            # Save audio to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(new_audio_bytes)
                tmp_path = tmp_file.name

            # Transcribe using Whisper API
            try:
                with st.spinner("Transcribing..."):
                    client = openai.OpenAI()
                    with open(tmp_path, "rb") as audio_file:
                        transcript = client.audio.transcriptions.create(
                            model="whisper-1", file=audio_file
                        )

                    # Store the transcribed text
                    transcribed_text = transcript.text
                    if transcribed_text.strip():
                        st.session_state.transcription = transcribed_text
            except Exception as e:
                st.error(f"Error transcribing audio: {e}")
    else:
        # No new audio this run
        st.session_state.audio_recorded = False

    # Only show transcription info if not marked for clearing AND we have a transcription
    if not st.session_state.clear_feedback and st.session_state.transcription:
        with feedback_container.container():
            st.info(f"Transcribed: {st.session_state.transcription}")

            # Center the button
            col1, col2, col3 = st.columns([1.5, 2, 1.5])
            with col2:
                use_transcription = st.button(
                    "Use this transcription", key="use_transcript_btn"
                )
    else:
        use_transcription = False
        # If we were clearing feedback but now we don't have new audio,
        # reset the clear flag for next time
        if st.session_state.clear_feedback and not new_audio_bytes:
            st.session_state.clear_feedback = False

    # Determine if we need to process input
    process_text = send_pressed and user_input
    process_transcription = use_transcription and st.session_state.transcription

    # Clear the feedback container if Send is clicked with text input
    if send_pressed and user_input:
        feedback_container.empty()
        st.session_state.clear_feedback = True

    # Reset process_input at the start of each interaction
    st.session_state.process_input = None

    # Set the input to process based on what was submitted
    if process_text:
        st.session_state.process_input = user_input
        # Clear transcription when sending text input
        st.session_state.transcription = None
    elif process_transcription:
        st.session_state.process_input = st.session_state.transcription
        # Clear transcription after using it
        st.session_state.transcription = None
        # Set flag to clear feedback container on next run
        st.session_state.clear_feedback = True
        # Clear the container immediately
        feedback_container.empty()

    # Process input if available
    if st.session_state.process_input:
        processing_container = st.container()
        with processing_container:
            with st.spinner("Processing your request..."):
                result = st.session_state.app.invoke(
                    {"input": st.session_state.process_input}
                )

                # Check if we have weather data in session state
                if "weather_data" in st.session_state:
                    st.session_state.response = {
                        "type": "weather",
                        "data": st.session_state.weather_data,
                    }
                    del st.session_state.weather_data
                else:
                    st.session_state.response = {
                        "type": "text",
                        "data": result["output"],
                    }

                # Clear input after processing
                st.session_state.process_input = None

    # Display response in full width container
    if st.session_state.response:
        st.markdown("---")  # Add a separator

        # Use a container for the full-width response
        response_container = st.container()
        with response_container:
            if st.session_state.response["type"] == "weather":
                display_weather(st.session_state.response["data"])
            else:
                st.write(st.session_state.response["data"])


if __name__ == "__main__":
    main()
