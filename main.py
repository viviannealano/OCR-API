from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import google.generativeai as genai
import io
import base64

app = FastAPI()

# CORS Middleware 
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://educart-capstone.vercel.app",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini API Key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


@app.get("/")
async def root():
    return {"message": "Welcome to the Gemini OCR API"}


@app.post("/api/ocr")
async def ocr(image: UploadFile = File(...)):
    # Read the image bytes
    image_bytes = await image.read()

    # Convert image bytes → Base64
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # Prepare multimodal messages
    messages = [
        {
            "role": "user",
            "parts": [
                {
                    "text": """
                    You are an OCR verification system for a university marketplace app (EduCart). 
                    Analyze the uploaded image and determine whether it is a valid identification card 
                    (student, faculty, alumni, or government-issued ID).

                    1. If the image does NOT contain an ID (selfie, paper, screenshot, object, etc.), reply EXACTLY:
                       "This is not an ID."

                    2. If the image appears to be an ID but:
                       - text is blurred,
                       - name is unreadable,
                       - quality is too low,
                       - photo is cropped or unclear,
                       reply EXACTLY:
                       "The ID is too blurry or unreadable."

                    3. If the ID is valid AND readable:
                       - Extract ONLY the person's name.
                       - Ignore middle names and middle initials.
                       - Output: First Name + Last Name only.
                       - Include multiple given names if part of the first name (e.g., “Mary Ann”).

                    VERY IMPORTANT:
                    - Output should ONLY be the extracted name or one of the EXACT error messages.
                    - Never explain.
                    - Never guess the name when unclear.
                    """
                },
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_base64
                    }
                }
            ]
        }
    ]

    # Call Gemini Model
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(messages)

    output = response.text.strip()

    # Handle exact responses
    if output == "This is not an ID.":
        return {"Error": "This is not an ID."}

    if output == "The ID is too blurry or unreadable.":
        return {"Error": "The ID is too blurry or unreadable."}

    # Otherwise, return extracted name
    return {"Name": output}
