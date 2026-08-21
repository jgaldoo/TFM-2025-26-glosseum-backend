from fastapi import FastAPI
from src.core.api.routes import router as api_router

app = FastAPI()

app.include_router(api_router)


def main():
    print("Hello from glosseum-backend!")


if __name__ == "__main__":
    main()
