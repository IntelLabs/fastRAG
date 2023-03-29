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
# See: https://github.com/deepset-ai/haystack/blob/main/rest_api/controller/setup_pipeline.py

from fastapi import APIRouter, FastAPI

import fastrag

from ..utils import get_app

logger = fastrag.utils.init_logger(__name__)
router = APIRouter()


@router.get("/status", status_code=200)
def status():
    return {"message": "System is up"}


@router.get("/version", status_code=200)
def version():
    return {"fastRAG": fastrag.__version__}
