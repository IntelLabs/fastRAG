import re
from enum import Enum
from functools import wraps
from importlib import import_module
from time import perf_counter
from typing import Any, Dict, Optional

import yaml


# adapted from https://github.com/deepset-ai/haystack/blob/594d2a10f84d13aef495c1cfbdaf4acad730c914/haystack/nodes/ranker/base.py#L68
def fastrag_timing(self, fn, attr_name):
    """Wrapper method used to time functions."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if attr_name not in self.__dict__:
            self.__dict__[attr_name] = 0
        tic = perf_counter()
        ret = fn(*args, **kwargs)
        toc = perf_counter()
        self.__dict__[attr_name] = toc - tic
        return ret

    return wrapper


def add_timing_to_pipeline(fn):
    """wrapper for Haystack Pipeline to add timing info"""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        ret = fn(*args, **kwargs)
        p = args[0]
        # print(args, kwargs)
        ret["timings"] = get_timing_from_pipeline(p)
        return ret

    return wrapper


def get_timing_from_pipeline(pipeline):
    """run print_time() in each component that has timing measurement func"""
    timings = {}
    for c in pipeline.components:
        if hasattr(pipeline.components[c], "query_time"):
            name = pipeline.components[c].__class__.__name__
            timings[name] = (
                pipeline.components[c].query_count,
                pipeline.components[c].query_time,
            )
    return timings


def missing_deps(classname: str, import_error: Exception):
    class MissingDependency:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"Failed to import '{classname}', " f"(Original error: {str(import_error)})"
            ) from import_error

        def __getattr__(self, *a, **k):
            return None

    return MissingDependency


def init_haystack_cls(component_name: str, parameters: Dict[str, Any], name: Optional[str] = None):
    return BaseComponent._create_instance(component_name, parameters, name)


def init_cls(class_name: str, parameters: dict, **kwargs):
    """Initialize a class object given name as string and initiate using parameters dict"""
    try:
        module_path, class_name = class_name.rsplit(".", 1)
        module = import_module(module_path)
        class_init = getattr(module, class_name)

    except (ImportError, AttributeError) as e:
        raise ImportError(class_name)
    try:
        p = parameters.copy()
        p.update(kwargs)
        new_cls_obj = class_init(**p)
        return new_cls_obj
    except Exception as e:
        raise e


def load_yaml(filename: str) -> dict:
    """Load a yaml from file and return as dicts"""
    with open(filename) as fp:
        content = yaml.safe_load(fp)
    return content


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(pattern, flags=re.IGNORECASE + re.UNICODE + re.MULTILINE)
    except BaseException:
        return False
    return pattern.search(text) is not None


GET_TEXT_MAP = {
    "direct": lambda x: x["text"],
    "content": lambda x: x.content,
}

GET_TITLE_MAP = {
    "direct": lambda x: x["title"],
    "content": lambda x: x.meta["title"],
}


def get_ans(text, answers):
    for ans in answers:
        has_answer = regex_match(text, ans)
        if has_answer:
            return True
    return False


def get_retrieval_gt_by_ids(contexts, expected):
    return [int(x.id in expected) for x in contexts]


def get_has_answer_data(contexts, answers, answer_type, text_get="direct", title_get="direct"):
    if answer_type == AnswerGroundType.ID:
        return get_retrieval_gt_by_ids(contexts, answers)

    passage_ground_truth = []
    for c in contexts:
        text = GET_TEXT_MAP[text_get](c)
        title = GET_TITLE_MAP[title_get](c)
        context_has_answer = get_ans(text, answers)
        title_has_answer = get_ans(title, answers)
        passage_has_answer = int(context_has_answer or title_has_answer)
        passage_ground_truth.append(passage_has_answer)

    return passage_ground_truth


def remove_html_from_text(text):
    text = re.sub("<[^<]+?>", "", text)
    text = text.replace("\n", " ")
    return text


class AnswerGroundType(Enum):
    HAS_ANSWER = 1
    ID = 2
