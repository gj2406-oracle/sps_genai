FROM python:3.12-slim

WORKDIR /code

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN pip install uv
RUN uv sync --frozen

RUN uv run python -m spacy download en_core_web_md

COPY app ./app

CMD ["uv", "run", "fastapi", "run", "app/main.py", "--port", "80"]
