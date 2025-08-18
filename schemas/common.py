from __future__ import annotations
from typing import Generic, Optional, TypeVar
from pydantic.generics import GenericModel  # v2에서도 사용

T = TypeVar("T")


class CommonResponse(GenericModel, Generic[T]):
    success: bool
    code: str
    message: str
    data: Optional[T] = None
