import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import routes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

app = FastAPI(title="ComicCraft", description="AI Comic Story Creator using Gemini Models")

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(routes.router)
