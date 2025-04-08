from langchain_openai import ChatOpenAI
from google.oauth2 import service_account
from googleapiclient.discovery import build
import os


def connect_model(api_key, base_url, model, temperature=0.4):
    llm = ChatOpenAI(
        model=model, api_key=api_key, base_url=base_url, temperature=temperature
    )
    return llm


def create_google_calendar_event(
    title, start_time, end_time, location="", description=""
):
    # Define the scope and get credentials from service account file
    SCOPES = ["https://www.googleapis.com/auth/calendar"]
    SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")

    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )

    service = build("calendar", "v3", credentials=credentials)

    event = {
        "summary": title,
        "location": location,
        "description": description,
        "start": {
            "dateTime": start_time,
            "timeZone": "America/New_York",
        },
        "end": {
            "dateTime": end_time,
            "timeZone": "America/New_York",
        },
    }

    created_event = (
        service.events()
        .insert(
            calendarId="a1b92fc8e136147f46e916c12e82e5ec1bbcf7771b4835e3d14d5f586a665ae0@group.calendar.google.com",
            body=event,
        )
        .execute()
    )
    return created_event
