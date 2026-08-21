# Glosseum Backend
An uv FastAPI application made to interact with [this frontend](https://github.com/jgaldoo/TFM-2025-26-glosseum-frontend)

📖 This application was created as part of a Master's Thesis for the "Máster en Ingeniería Informática" of the "Universidad Complutense de Madrid".

## Technologies

This project was developed in [Python]() using [uv](). This uses the following libraries amongst others:
- [Ollama]() (not directly) to run AI models locally.
- [FastAPI]() to manage the API REST endpoint.
- [Pydantic]() to declare models used to transfer data inside and outside of the application.
- [PydanticAI]() to receive structured format from AI answers
- [Google Cloud Vision]() to apply OCR and detect text from images

## How to run

To run this project you must have [Python]() and [UV installed]().

Then download this project and open it in your IDE of preference.

From the root of the project, download all dependencies:
```
uv sync
```

Ensure you fill all .env fields. Also ensure the ``OLLAMA_LOCAL_MODEL`` value corresponds to the name of an AI model you can run locally in your computer. Run this model.

Then execute the following command to initiate the endpoint
```
uv run fastapi run main.py --host 0.0.0.0 --port 8000
```

That should be it!