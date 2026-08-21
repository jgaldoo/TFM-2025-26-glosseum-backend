# Glosseum Backend
An uv FastAPI application made to interact with [this frontend](https://github.com/jgaldoo/TFM-2025-26-glosseum-frontend)

📖 This application was created as part of a Master's Thesis for the "Máster en Ingeniería Informática" of the "Universidad Complutense de Madrid".

## Technologies

This project was developed in [Python](https://www.python.org/) using [uv](https://docs.astral.sh/uv/). This uses the following libraries amongst others:
- [FastAPI](https://fastapi.tiangolo.com/) to manage the API REST endpoint.
- [Pydantic](https://pydantic.dev/docs/validation/latest/get-started/) to declare models used to transfer data inside and outside the application.
- [PydanticAI](https://pydantic.dev/docs/ai/overview/) to receive structured format from AI answers
- [Google Cloud Vision](https://docs.cloud.google.com/vision/docs/libraries?utm_source=chatgpt.com&hl=es) to apply OCR and detect text from images

It has been developed with the usage of [Ollama](https://ollama.com/) to run AI models locally.

## How to run

To run this project you must have [Python](https://www.python.org/downloads/) and [uv installed](https://docs.astral.sh/uv/#installation).

Then download this project and open it in your IDE of preference.

From the root of the project, download all dependencies:
```
uv sync
```

Ensure you fill all .env fields with valid values. This means you must provide credentials for the services of Google Cloud Vision and Gemini.

Also ensure the ``OLLAMA_LOCAL_MODEL`` value corresponds to the name of an AI model you can run locally in your computer. Run this model.
For example, for ``qwen2.5:7b-instruct``, with Ollama, you can run:
```
ollama serve
ollama run qwen2.5:7b-instruct
```

Then execute the following command to initiate the endpoint
```
uv run fastapi run main.py --host 0.0.0.0 --port 8000
```

That should be it!