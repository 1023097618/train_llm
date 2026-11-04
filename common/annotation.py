import datetime
from functools import wraps
import inspect
import json
from types import UnionType
from typing import Optional, Union, get_origin,get_type_hints
from docstring_parser import parse

_PY_TO_JSON = {str: "string", int: "integer", float: "number",
               bool: "boolean", list: "array", dict: "object"}

def _one_type(py_type):
    return {"type": _PY_TO_JSON.get(py_type, "string")}

def _type_schema(py_type):
    """the JSON schema fragment of a type (without description)"""
    origin = get_origin(py_type)
    if origin is Union or origin is UnionType:
        non_none = [a for a in py_type.__args__ if a is not type(None)]
        if len(non_none) == 1:
            return _one_type(non_none[0])
        return {"anyOf": [_one_type(a) for a in non_none]}
    return _one_type(py_type)

def tool(fn):
    doc = parse(fn.__doc__ or "")
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    param_docs = {p.arg_name: p.description for p in doc.params}

    properties, required = {}, []
    for name, p in sig.parameters.items():
        properties[name] = {
            **_type_schema(hints.get(name, str)),
            "description": param_docs.get(name, ""),
        }
        if p.default is inspect.Parameter.empty:
            required.append(name)

    fn.schema = {"function": {
        "name": fn.__name__,
        "description": (doc.short_description or "").strip(),
        "parameters": {"type": "object", "properties": properties, "required": required},
    }}
    fn.schema=json.dumps(fn.schema, ensure_ascii=False)


    @wraps(fn)                       # keeps __name__, __doc__ and the rest
    def new_fn(*args, **kwargs):
        result = fn(*args, **kwargs)
        if isinstance(result, str):
            return result           # a string is used as is, json.dumps would add another layer of quotes
        try:
            return json.dumps(result, ensure_ascii=False)  # dicts, lists and scalars
        except (TypeError, ValueError):
            return str(result)      # fallback for what json cannot serialize
    
    new_fn.schema = fn.schema
    return new_fn