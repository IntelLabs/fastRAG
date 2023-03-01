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
# See: https://github.com/deepset-ai/haystack/blob/main/rest_api/application.py

import argparse
import logging

import uvicorn

import fastrag

from .utils import get_app

logger = logging.getLogger(__name__)
config = None
app = get_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="path to pipeline configuration (yaml)"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="host IP to use for service API calls"
    )
    parser.add_argument("--port", type=int, default=8000, help="service port number to use")
    args = parser.parse_args()
    app.pipeline = fastrag.load_pipeline(args.config)
    uvicorn.run(app, host=args.host, port=args.port)
