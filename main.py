from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import google.generativeai as genai
import io
import os

app = FastAPI()

# CORS Middleware 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://educart-capstone.vercel.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

@app.get("/")
async def root():
    return {"message": "Welcome to the Gemini OCR API"}

@app.post("/api/ocr")
async def ocr(image: UploadFile = File(...)):
    # Read image bytes
    image_bytes = await image.read()

    # Prepare image for Gemini
    image_part = {
        "mime_type": image.content_type,
        "data": image_bytes
    }

    model = genai.GenerativeModel("gemini-2.5-flash")

    response = model.generate_content([
        """
        You are an OCR verification system for a university marketplace app (EduCart). 
        Analyze the uploaded image and determine whether it is a valid identification card 
        (student, faculty, alumni, or government-issued ID).

        1. If the image does NOT contain an ID, 
           reply EXACTLY:
           "This is not an ID."

        2. If the image appears to be an ID but:
           - the text is blurred,
           - the name area is unreadable,
           - the quality is too low,
           - the ID is cropped,
           - lighting makes the text unclear,
           reply EXACTLY:
           "The ID is too blurry or unreadable."

           Do NOT guess the name.

        3. If the ID is valid AND readable:
           - Extract ONLY the person's name.
           - Ignore middle names and middle initials.
           - Output EXACTLY: First Name + Last Name.
           - If the first name includes multiple given names (ex: “Mary Ann”), keep them.

        VERY IMPORTANT RULES:
        - Output ONLY the extracted name OR one of the two exact messages.
        - Never give explanations.
        - Never add extra text.
        - Never guess names when text is unclear.
        """,
        image_part  # ← THIS now correctly includes the uploaded image
    ])

    text_response = response.text.strip()

    if text_response == "This is not an ID.":
        return {"Error": "This is not an ID."}

    if text_response == "The ID is too blurry or unreadable.":
        return {"Error": "The ID is too blurry or unreadable."}

    return {"Name": text_response}

