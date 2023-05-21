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
# See: https://github.com/deepset-ai/haystack/blob/main/rest_api/schema.py

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

from haystack.schema import Answer, Document
from pydantic import BaseConfig, BaseModel, Extra, Field

BaseConfig.arbitrary_types_allowed = True
BaseConfig.json_encoders = {
    np.ndarray: lambda x: x.tolist(),
    pd.DataFrame: lambda x: x.to_dict(orient="records"),
}


PrimitiveType = Union[str, int, float, bool]


class SetupParams(BaseModel):
    params: dict = None


class RequestBaseModel(BaseModel):
    class Config:
        # Forbid any extra fields in the request to avoid silent failures
        extra = Extra.forbid


class QueryRequest(RequestBaseModel):
    query: str
    params: Optional[dict] = None
    debug: Optional[bool] = False


class FilterRequest(RequestBaseModel):
    filters: Optional[
        Dict[str, Union[PrimitiveType, List[PrimitiveType], Dict[str, PrimitiveType]]]
    ] = None


class CreateLabelSerialized(RequestBaseModel):
    id: Optional[str] = None
    query: str
    document: Document
    is_correct_answer: bool
    is_correct_document: bool
    origin: Literal["user-feedback", "gold-label"]
    answer: Optional[Answer] = None
    no_answer: Optional[bool] = None
    pipeline_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    meta: Optional[dict] = None
    filters: Optional[dict] = None


class QueryResponse(BaseModel):
    query: str
    answers: Optional[List] = []
    documents: List[Document] = []
    images: Optional[Dict] = None
    relations: Optional[List] = None
    debug: Optional[Dict] = Field(None, alias="_debug")
    timings: Optional[Dict] = None
    results: Optional[List] = None
