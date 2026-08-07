
from contextlib import asynccontextmanager
from pathlib import Path
import os
import traceback

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from model_loader import ModelBundle, load_model_bundle
from inference import translate



# This is the HTTP server
# Responsible for:
#   creates FastAPI application
#   loads the model when container starts
#   Runs the warmup translation
#   Provides SageMaker's /ping endpoint
#   Provides SageMaker's /invodations endpoint
# Docker will launch this file



MODEL_DIR = Path(os.getenv("MODEL_DIR", "/opt/ml/model"))

bundle: ModelBundle | None = None
load_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bundle, load_error

    try:
        bundle = load_model_bundle(MODEL_DIR)

        # Process-level model warmup.
        translate(
            bundle=bundle,
            text="Hello.",
            max_output_length=8,
        )
        print(f"Model loaded successfully from {MODEL_DIR}", flush=True)
        
    except Exception:
        load_error = traceback.format_exc()
        print(load_error, flush=True)

    yield


app = FastAPI(lifespan=lifespan)


@app.api_route("/ping", methods=["GET", "POST"])
def ping() -> Response:
    if bundle is None:
        return Response(
            content=load_error or "Model is not loaded",
            status_code=503,
        )

    return Response(status_code=200)


@app.post("/invocations")
async def invocations(request: Request) -> JSONResponse:
    if bundle is None:
        return JSONResponse(
            content={"error": load_error or "Model unavailable"},
            status_code=503,
        )

    try:
        payload = await request.json()

        text = payload.get("text")
        max_output_length = int(
            payload.get("max_output_length", 100)
        )

        if not isinstance(text, str) or not text.strip():
            return JSONResponse(
                content={"error": "'text' must be a non-empty string"},
                status_code=400,
            )

        if len(text) > 1000:
            return JSONResponse(
                content={"error": "Input text is too long"},
                status_code=400,
            )

        if not 2 <= max_output_length <= 200:
            return JSONResponse(
                content={
                    "error": (
                        "max_output_length must be between 2 and 200"
                    )
                },
                status_code=400,
            )

        translation = translate(
            bundle=bundle,
            text=text,
            max_output_length=max_output_length,
        )

        return JSONResponse(
            content={
                "input": text,
                "translation": translation,
            }
        )

    except Exception as exception:
        return JSONResponse(
            content={
                "error": type(exception).__name__,
                "message": str(exception),
            },
            status_code=500,
        )



# async function warmModel() {
#   const statusElement = document.querySelector("#model-status");
#   const submitButton = document.querySelector("#translate-button");

#   statusElement.textContent = "Starting translation model…";
#   submitButton.disabled = true;

#   try {
#     const response = await fetch(
#       "https://api.example.com/translate",
#       {
#         method: "POST",
#         headers: {
#           "Content-Type": "application/json",
#         },
#         body: JSON.stringify({
#           text: "Hello.",
#           max_output_length: 8,
#         }),
#       }
#     );

#     if (!response.ok) {
#       throw new Error(`Warmup failed: ${response.status}`);
#     }

#     statusElement.textContent = "Model ready";
#     submitButton.disabled = false;
#   } catch (error) {
#     console.error(error);
#     statusElement.textContent =
#       "Model is still starting. Try again shortly.";
#   }
# }