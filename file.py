from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

response = client.models.embed_content(
    model="gemini-embedding-001",
    contents="What is Python?"
)

print(len(response.embeddings[0].values))