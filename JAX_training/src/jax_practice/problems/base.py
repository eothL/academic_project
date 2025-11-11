"""Shared problem definition infrastructure."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from types import ModuleType
from typing import Callable


TestRunner = Callable[[ModuleType], None]


@dataclass(frozen=True)
class Problem:
    slug: str
    title: str
    prompt: str
    starter_code: str
    example_solution: str
    test_runner: TestRunner

    def render_starter(self) -> str:
        header = dedent(
            f'''"""
            {self.title}

            {self.prompt.strip()}
            """
            '''
        ).strip()
        return f"{header}\n\n{self.starter_code.strip()}\n"

    def run_hidden_tests(self, module: ModuleType) -> None:
        self.test_runner(module)
