"""FastAPI application serving the Message Intelligence results.

The application is a thin presentation layer over the validated pipeline
artifacts: every route delegates to :class:`ApiService` / the
:class:`OutputRepository` and returns a Pydantic response model. No business
logic lives in the route handlers and no raw sensitive value can appear in a
response - the models it serializes are the pipeline's sanitized ones.

Dashboard:
    GET /                       - single-page dashboard (Jinja2 + static assets)
API:
    GET /health                 - liveness
    GET /api/stats              - aggregate statistics
    GET /api/messages           - message list with search / category / sensitive filters
    GET /api/messages/{id}      - full sanitized detail for one message
    GET /api/tasks              - extracted tasks/events with filters
    GET /api/sensitive          - sanitized sensitive detections
    GET /api/demo/mandatory     - the 15 mandatory demo messages
    GET /api/validation         - the validation report
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import Settings
from app.models.api import (
    HealthResponse,
    MandatoryDemoResponse,
    MessageDetail,
    MessageListResponse,
    SensitiveListResponse,
    StatsResponse,
    TaskListResponse,
    ValidationReportResponse,
)
from app.models.classification import Category
from app.models.task_event import ItemType, Priority
from app.services.api_service import ApiService, MessageNotFoundError
from app.services.mandatory_demo import MandatoryDemoError
from app.services.output_repository import OutputRepository, OutputRepositoryError

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

APP_TITLE = "Message Intelligence"
APP_VERSION = "0.1.0"
APP_DESCRIPTION = (
    "Sanitized read API over the generated pipeline artifacts "
    "(classification, extraction, sensitive detections, validation). "
    "Responses never contain raw sensitive values."
)

DEFAULT_PAGE_LIMIT = 100
MAX_PAGE_LIMIT = 900
MAX_TASK_LIMIT = 2000


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application with services bound to ``settings``."""
    settings = settings or Settings.from_env()
    repository = OutputRepository(
        outputs_dir=settings.outputs_dir,
        mandatory_demo_ids_path=settings.mandatory_demo_ids_path,
    )
    service = ApiService(repository)

    app = FastAPI(
        title=APP_TITLE,
        version=APP_VERSION,
        description=APP_DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.settings = settings
    app.state.service = service

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    templates = Jinja2Templates(directory=TEMPLATES_DIR)

    _register_exception_handlers(app)

    # ------------------------------------------------------------- dashboard

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "dashboard.html")

    # ---------------------------------------------------------------- system

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    # ------------------------------------------------------------------ stats

    @app.get("/api/stats", response_model=StatsResponse, tags=["api"])
    def api_stats() -> StatsResponse:
        return service.stats()

    # --------------------------------------------------------------- messages

    @app.get("/api/messages", response_model=MessageListResponse, tags=["api"])
    def api_messages(
        search: str | None = Query(default=None, max_length=200),
        category: Category | None = None,
        sensitive: bool | None = None,
        limit: int = Query(default=DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
        offset: int = Query(default=0, ge=0),
    ) -> MessageListResponse:
        return service.list_messages(
            search=search,
            category=category.value if category else None,
            sensitive=sensitive,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/messages/{message_id}", response_model=MessageDetail, tags=["api"])
    def api_message(message_id: str) -> MessageDetail:
        return service.get_message(message_id)

    # ------------------------------------------------------------------ tasks

    @app.get("/api/tasks", response_model=TaskListResponse, tags=["api"])
    def api_tasks(
        item_type: ItemType | None = Query(default=None, alias="type"),
        priority: Priority | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        limit: int = Query(default=DEFAULT_PAGE_LIMIT, ge=1, le=MAX_TASK_LIMIT),
        offset: int = Query(default=0, ge=0),
    ) -> TaskListResponse:
        return service.list_tasks(
            item_type=item_type,
            priority=priority,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            offset=offset,
        )

    # --------------------------------------------------------------- sensitive

    @app.get("/api/sensitive", response_model=SensitiveListResponse, tags=["api"])
    def api_sensitive(
        limit: int = Query(default=DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
        offset: int = Query(default=0, ge=0),
    ) -> SensitiveListResponse:
        return service.list_sensitive(limit=limit, offset=offset)

    # ---------------------------------------------------------- mandatory demo

    @app.get("/api/demo/mandatory", response_model=MandatoryDemoResponse, tags=["api"])
    def api_mandatory_demo() -> MandatoryDemoResponse:
        return service.mandatory_demo()

    # -------------------------------------------------------------- validation

    @app.get("/api/validation", response_model=ValidationReportResponse, tags=["api"])
    def api_validation() -> ValidationReportResponse:
        return service.validation_report()

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    """Map domain errors to stable HTTP responses (no raw exceptions leaked)."""

    @app.exception_handler(OutputRepositoryError)
    def handle_repository_error(_: Request, exc: OutputRepositoryError) -> JSONResponse:
        logger.warning("output repository error: %s", exc)
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(MandatoryDemoError)
    def handle_mandatory_error(_: Request, exc: MandatoryDemoError) -> JSONResponse:
        logger.warning("mandatory demo error: %s", exc)
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(MessageNotFoundError)
    def handle_not_found(_: Request, exc: MessageNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    def handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. See server logs for details."},
        )


app = create_app()
