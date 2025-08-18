import json
from jinja2 import Environment, FileSystemLoader, select_autoescape
from core.config import settings


def _tojson(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


_env = Environment(
    loader=FileSystemLoader(settings.template_path),
    autoescape=select_autoescape([]),
    trim_blocks=True,
    lstrip_blocks=True,
)
_env.filters["tojson"] = _tojson


def render_template(name: str, **data) -> str:
    tpl = _env.get_template(name)
    return tpl.render(**data)
