# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# This file is based on https://github.com/deepset-ai/haystack
# See: https://github.com/deepset-ai/haystack/blob/main/rest_api/utils.py

import logging

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.openapi.utils import get_openapi
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from fastrag import __version__ as fastrag_version

logger = logging.getLogger(__name__)

app = None


async def http_error_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse({"errors": [exc.detail]}, status_code=exc.status_code)


def get_app() -> FastAPI:
    """
    Initializes the App object and creates the global pipelines as possible.
    """
    global app  # pylint: disable=global-statement
    if app:
        return app

    from .config import ROOT_PATH

    app = FastAPI(
        title="fasrRAG REST API",
        debug=True,
        version=fastrag_version,
        root_path=ROOT_PATH,
    )

    from .controller import qa, setup_pipeline

    router = APIRouter()
    router.include_router(setup_pipeline.router, tags=["demo"])
    router.include_router(qa.router, tags=["Q&A"])
    # router.include_router(file_upload.router, tags=["file-upload"])
    # router.include_router(document.router, tags=["document"])

    # This middleware enables allow all cross-domain requests to the API from a browser. For production
    # deployments, it could be made more restrictive.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_exception_handler(HTTPException, http_error_handler)
    app.include_router(router)

    # Simplify operation IDs so that generated API clients have simpler function
    # names (see https://fastapi.tiangolo.com/advanced/path-operation-advanced-configuration/#using-the-path-operation-function-name-as-the-operationid).
    # The operation IDs will be the same as the route names (i.e. the python method names of the endpoints)
    # Should be called only after all routes have been added.
    for route in app.routes:
        if isinstance(route, APIRoute):
            route.operation_id = route.name

    return app
