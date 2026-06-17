from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.action_cleanliness import (
    ActionCleanlinessService,
    StaffZoneVisit,
    SAMPLE_ORDER_TIME,
    SAMPLE_PAYMENT_COMPLETED_TIME,
    SAMPLE_TRAJECTORY_JSON,
    TableOccupancySample,
    YoloTableState,
    ZoneTableMapping,
    parse_action_time_value,
    parse_trajectory_json,
)
from app.analysis import AnalysisService, extract_first_video_frame, image_data_url, read_image, save_analysis_crop, write_image
from app.auth import create_access_token, hash_password, normalize_user_id, parse_access_token, validate_password, verify_password
from app.cleanliness import (
    CLEANLINESS_IMAGE_EXTENSIONS,
    DEFAULT_CLEANLINESS_PROMPT_PROFILE,
    CleanlinessService,
    normalize_cleanliness_prompt_profile,
)
from app.cleanliness_metric import build_visual_payload_from_yolo_detections, normalize_visual_metric_input
from app.config import (
    DATA_DIR,
    FRONTEND_DIR,
    MOBILE_CORS_EXTRA_ORIGINS,
    MOBILE_VIDEO_DIR,
    MOBILE_VIDEO_UPLOAD_INTERVAL_SECONDS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    POSTER_TEMPLATE_DIR,
    UPLOAD_DIR,
)
from app.database import (
    count_owner_accounts,
    fetch_cleanliness_filter_options,
    fetch_cleanliness_result_by_job_id,
    fetch_cleanliness_results,
    fetch_cleanliness_store_summary,
    fetch_filter_options,
    fetch_latest_by_roi,
    fetch_owner_account,
    fetch_results,
    init_db,
    insert_owner_account,
    insert_cleanliness_result,
    update_cleanliness_result,
)
from app.hybrid_cleanliness import (
    OBJECT_CONFIDENCE_THRESHOLD,
    build_final_cleanliness_result,
    build_hybrid_cleanliness_result,
    object_score_to_decision,
)
from app.person_masking import IMAGE_EXTENSIONS, PersonMaskService
from app.roi_store import ConfigStore
from app.schemas import (
    ActionWorkflowRequest,
    ROI,
    safe_filename_part,
)
from app.video_cleanliness import VIDEO_CLEANLINESS_EXTENSIONS, VideoCleanlinessService
from app.vision_workflow_preprocessor import (
    DynamicVideoSamplingConfig,
    build_workflow_frames_from_images,
    build_workflow_frames_from_video,
    captured_at_for_video_frame,
    sample_dynamic_video_workflow_frames,
    sample_video_workflow_frames,
    summarize_dynamic_video_samples,
)
from app.yolo_module import yolo_module


app = FastAPI(title="MVP1 Franchise Quality Monitor")

CORS_ALLOWED_ORIGINS = [
    "http://localhost",
    "https://localhost",
    "capacitor://localhost",
    "ionic://localhost",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    *MOBILE_CORS_EXTRA_ORIGINS,
]
CORS_ALLOWED_ORIGIN_REGEX = (
    r"^(?:"
    r"https?://(?:"
    r"localhost|"
    r"127(?:\.\d{1,3}){3}|"
    r"10(?:\.\d{1,3}){3}|"
    r"192\.168(?:\.\d{1,3}){2}|"
    r"172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2}"
    r")(?::\d{1,5})?|"
    r"(?:capacitor|ionic)://localhost"
    r")$"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_origin_regex=CORS_ALLOWED_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="app/templates")
config_store = ConfigStore()
analysis_service = AnalysisService()
person_mask_service = PersonMaskService()
cleanliness_service = CleanlinessService()
action_cleanliness_service = ActionCleanlinessService()
video_cleanliness_service = VideoCleanlinessService(person_mask_service=person_mask_service)
workflow_yolo_helper = yolo_module()

init_db()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")
app.mount("/mobile", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="mobile_frontend")


def save_upload(upload: UploadFile, target_dir: Path, fallback_stem: str) -> Path:
    suffix = Path(upload.filename or "").suffix.lower() or ".bin"
    target_path = target_dir / f"{fallback_stem}{suffix}"
    with target_path.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return target_path


def config_payload(config_id: str | None) -> dict[str, Any] | None:
    if not config_id:
        return None
    try:
        return config_store.load(config_id).to_dict()
    except FileNotFoundError:
        return None


def with_human_clear_ratio(record: dict[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    if "visible_ratio" in payload and "human_clear_ratio" not in payload:
        payload["human_clear_ratio"] = payload["visible_ratio"]
    return payload


def auth_payload(user_id: str) -> dict[str, str]:
    return {
        "access_token": create_access_token(user_id),
        "token_type": "bearer",
        "user_id": user_id,
    }


ACTION_WORKFLOW_DEMO_STAFF_VISITS_JSON = json.dumps(
    [
        {
            "visit_id": "visit_000341",
            "staff_id": "staff_03",
            "zone_id": "zone_B",
            "entered_at": "2026-06-03T14:44:10",
            "left_at": "2026-06-03T14:44:36",
            "dwell_seconds": 26,
            "mean_confidence": 0.76,
            "sample_count": 8,
        }
    ],
    ensure_ascii=False,
    indent=2,
)

VISUAL_PAYLOAD_SOURCE_PRESET = "preset"
VISUAL_PAYLOAD_SOURCE_JSON = "json"
VISUAL_PAYLOAD_SOURCE_YOLO = "yolo"


def current_owner_id(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="authentication required")
    token = authorization.split(" ", 1)[1].strip()
    try:
        user_id = parse_access_token(token)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="invalid or expired token") from exc
    if fetch_owner_account(user_id) is None:
        raise HTTPException(status_code=401, detail="owner account not found")
    return user_id


@app.get("/api/auth/status")
def auth_status() -> dict[str, Any]:
    return {"has_owner": count_owner_accounts() > 0}


@app.post("/api/auth/bootstrap")
async def auth_bootstrap(user_id: str = Form(...), password: str = Form(...)) -> JSONResponse:
    if count_owner_accounts() > 0:
        raise HTTPException(status_code=409, detail="owner account already exists")
    try:
        normalized_user_id = normalize_user_id(user_id)
        validate_password(password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    insert_owner_account(normalized_user_id, hash_password(password))
    return JSONResponse(auth_payload(normalized_user_id), status_code=201)


@app.post("/api/auth/login")
async def auth_login(user_id: str = Form(...), password: str = Form(...)) -> dict[str, str]:
    try:
        normalized_user_id = normalize_user_id(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    account = fetch_owner_account(normalized_user_id)
    if account is None or not verify_password(password, str(account["password_hash"])):
        raise HTTPException(status_code=401, detail="invalid user_id or password")
    return auth_payload(normalized_user_id)


@app.get("/api/auth/me")
def auth_me(owner_id: str = Depends(current_owner_id)) -> dict[str, str]:
    return {"user_id": owner_id}


@app.get("/")
def dashboard(request: Request) -> Any:
    configs = config_store.list_configs()
    stores = config_store.list_store_names()
    latest_pop = [with_human_clear_ratio(row) for row in fetch_latest_by_roi("POP")]
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "request": request,
            "title": "Dashboard",
            "configs": configs,
            "stores": stores,
            "latest_pop": latest_pop,
            "absent_pop_count": len([row for row in latest_pop if row["decision"] == "Absent"]),
        },
    )


@app.get("/stores")
def stores_page(request: Request) -> Any:
    store_summaries = config_store.list_store_summaries()
    return templates.TemplateResponse(
        request,
        "stores.html",
        {
            "request": request,
            "title": "Stores",
            "stores": store_summaries,
        },
    )


@app.get("/setup")
def setup_page(
    request: Request,
    config_id: str | None = Query(default=None),
    store_name: str | None = Query(default=None),
) -> Any:
    store_summaries = config_store.list_store_summaries()
    selected = config_payload(config_id)
    selected_store_name = selected["store_name"] if selected else (store_name or "")
    selected_store_configs = config_store.list_configs_by_store(selected_store_name) if selected_store_name else []
    return templates.TemplateResponse(
        request,
        "setup.html",
        {
            "request": request,
            "title": "ROI Setup",
            "stores": store_summaries,
            "selected_store_name": selected_store_name,
            "selected_store_configs": selected_store_configs,
            "selected_config": selected,
        },
    )


@app.post("/stores")
async def create_store(store_name: str = Form(...)) -> RedirectResponse:
    try:
        normalized = store_name.strip()
        config_store.add_store_name(normalized)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(url=f"/stores?store_name={quote_plus(normalized)}", status_code=303)


@app.post("/setup/save")
async def save_setup(
    store_name: str = Form(...),
    cctv_nickname: str = Form(...),
    rois_json: str = Form(...),
    existing_reference_path: str = Form(default=""),
    reference_image: UploadFile | None = File(default=None),
) -> RedirectResponse:
    try:
        raw_rois = json.loads(rois_json)
        rois = [ROI.from_dict(item) for item in raw_rois]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid roi payload: {exc}") from exc

    temp_path: Path | None = None
    uploaded_temp_path: Path | None = None
    if reference_image and reference_image.filename:
        suffix = Path(reference_image.filename).suffix.lower() or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(reference_image.file, temp_file)
            uploaded_temp_path = Path(temp_file.name)
        if suffix in VIDEO_CLEANLINESS_EXTENSIONS:
            extracted_frame = extract_first_video_frame(uploaded_temp_path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_frame_file:
                temp_path = Path(temp_frame_file.name)
            write_image(temp_path, extracted_frame)
        else:
            temp_path = uploaded_temp_path

    try:
        config = config_store.save_config(
            store_name=store_name,
            cctv_nickname=cctv_nickname,
            rois=rois,
            reference_image_source=temp_path,
            existing_reference_path=existing_reference_path or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)
        if uploaded_temp_path and uploaded_temp_path != temp_path and uploaded_temp_path.exists():
            uploaded_temp_path.unlink(missing_ok=True)

    return RedirectResponse(
        url=f"/setup?store_name={quote_plus(config.store_name)}&config_id={config.config_id}",
        status_code=303,
    )


@app.get("/api/configs")
def list_configs() -> list[dict[str, Any]]:
    return [config.to_dict() for config in config_store.list_configs()]


@app.get("/api/stores")
def list_stores() -> list[dict[str, int | str]]:
    return config_store.list_store_summaries()


@app.get("/api/configs/{config_id}")
def get_config(config_id: str) -> dict[str, Any]:
    try:
        return config_store.load(config_id).to_dict()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="config not found") from exc


@app.get("/api/configs/{config_id}/rois/{roi_name}")
def get_roi(config_id: str, roi_name: str) -> dict[str, Any]:
    try:
        return config_store.get_roi(config_id, roi_name).to_dict()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="config not found") from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="roi not found") from exc


@app.post("/api/mobile/roi-configs")
async def save_mobile_roi_config(
    owner_id: str = Depends(current_owner_id),
    store_name: str = Form(...),
    cctv_nickname: str = Form(...),
    rois_json: str = Form(...),
    reference_image: UploadFile = File(...),
) -> JSONResponse:
    _ = owner_id
    try:
        raw_rois = json.loads(rois_json)
        rois = [ROI.from_dict(item) for item in raw_rois]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid roi payload: {exc}") from exc
    if not rois:
        raise HTTPException(status_code=400, detail="at least one roi is required")

    temp_path: Path | None = None
    suffix = Path(reference_image.filename or "").suffix.lower() or ".png"
    if suffix not in CLEANLINESS_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="reference image must be png, jpg, jpeg, bmp, or webp")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(reference_image.file, temp_file)
        temp_path = Path(temp_file.name)

    try:
        merged_rois = list(rois)
        config_id = f"{safe_filename_part(store_name)}_{safe_filename_part(cctv_nickname)}"
        try:
            existing_config = config_store.load(config_id)
            merged_by_name = {area.name.strip(): area for area in existing_config.areas}
            for roi in rois:
                merged_by_name[roi.name.strip()] = roi
            merged_rois = list(merged_by_name.values())
        except FileNotFoundError:
            pass

        config = config_store.save_config(
            store_name=store_name,
            cctv_nickname=cctv_nickname,
            rois=merged_rois,
            reference_image_source=temp_path,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)

    return JSONResponse(config.to_dict())


@app.get("/analyze")
def analyze_page(request: Request, config_id: str | None = Query(default=None)) -> Any:
    configs = config_store.list_configs()
    poster_templates = sorted(POSTER_TEMPLATE_DIR.glob("*"))
    selected = config_payload(config_id)
    return templates.TemplateResponse(
        request,
        "analyze.html",
        {
            "request": request,
            "title": "Analysis",
            "configs": configs,
            "configs_payload": [config.to_dict() for config in configs],
            "poster_templates": [path.name for path in poster_templates],
            "selected_config": selected,
        },
    )


def data_url(path: Path) -> str:
    return "/data/" + path.relative_to(DATA_DIR).as_posix()


def workflow_video_observation_budget(candidate_cap: int) -> int:
    return max(candidate_cap * 6, 24)


def workflow_video_sampling_config(interval_seconds: float, observation_budget: int) -> DynamicVideoSamplingConfig:
    return DynamicVideoSamplingConfig(
        idle_interval_seconds=max(interval_seconds, 1.0),
        occupied_interval_seconds=max(1.0, min(interval_seconds, 5.0)),
        transition_interval_seconds=max(0.5, min(interval_seconds / 2.0, 2.0)),
        post_check_interval_seconds=max(1.0, min(interval_seconds / 2.0, 3.0)),
        max_observations=max(observation_budget, 1),
    )


def serialize_dynamic_video_candidate(
    sample: dict[str, Any],
    *,
    captured_at_start: str,
) -> dict[str, Any]:
    serialized = {
        "timestamp_sec": round(float(sample.get("offset_seconds", 0.0)), 2),
        "captured_at": captured_at_for_video_frame(
            captured_at_start=captured_at_start,
            frame_index=sample.get("frame_index"),
            fps=sample.get("fps"),
            interval_seconds=max(float(sample.get("offset_seconds", 0.0)), 1.0),
            sample_index=0,
            offset_seconds=sample.get("offset_seconds"),
        ),
        "frame_type": sample.get("frame_type"),
        "sampling_state": sample.get("sampling_state"),
        "priority": sample.get("priority"),
        "reason_codes": list(sample.get("reason_codes", [])),
        "features": dict(sample.get("features", {})),
        "episode_id": sample.get("episode_id"),
        "selected_for_review": bool(sample.get("selected_for_review", False)),
        "selection_reasons": list(sample.get("selection_reasons", [])),
    }
    crop_image = sample.get("crop_image")
    if crop_image is not None:
        serialized["preview_url"] = image_data_url(crop_image)
    return serialized


def sampler_metadata_payload(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "sampler_frame_type": sample.get("frame_type"),
        "sampler_reason_codes": list(sample.get("reason_codes", [])),
        "sampler_priority": sample.get("priority"),
        "sampler_state": sample.get("sampling_state"),
        "sampler_features": dict(sample.get("features", {})),
        "sampler_timestamp_sec": round(float(sample.get("offset_seconds", 0.0)), 2),
        "sampler_episode_id": sample.get("episode_id"),
        "sampler_selected_for_review": bool(sample.get("selected_for_review", False)),
    }


def apply_dynamic_sample_occupancy(frames: list[dict[str, Any]], samples: list[dict[str, Any]]) -> None:
    for frame, sample in zip(frames, samples):
        features = sample.get("features", {})
        if "person_present" not in features:
            continue
        frame["person_present"] = bool(features.get("person_present", False))
        frame["person_count"] = int(features.get("person_count", 0) or 0)
        frame["occupancy_source"] = "dynamic_person_relevance"
        reason = features.get("person_relevance_reason")
        if reason:
            frame["occupancy_reason_codes"] = [f"person_relevance:{reason}"]


def serialize_dynamic_video_sampling_payload(
    samples: list[dict[str, Any]],
    *,
    table_id: str,
    captured_at_start: str,
    candidate_budget: int | None = None,
) -> dict[str, Any]:
    sampling_summary = summarize_dynamic_video_samples(samples, target_count=candidate_budget)
    debug_trace = [
        serialize_dynamic_video_candidate(sample, captured_at_start=captured_at_start)
        for sample in sampling_summary["debug_trace"]
    ]
    selected_candidates = [
        serialize_dynamic_video_candidate(sample, captured_at_start=captured_at_start)
        for sample in sampling_summary["selected_samples"]
    ]
    return {
        "table_id": table_id,
        "dynamic_sampling": True,
        "candidate_count": len(selected_candidates),
        "debug_trace_count": len(debug_trace),
        "selected_candidates": selected_candidates,
        "candidates": selected_candidates,
        "candidate_summary": selected_candidates,
        "debug_trace": debug_trace,
        "events": sampling_summary["events"],
        "episodes": sampling_summary["episodes"],
    }


def dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def action_workflow_report_details(record: dict[str, Any]) -> dict[str, Any] | None:
    if record.get("mode") != "action_workflow":
        return None

    raw_action_features = record.get("action_features")
    if not raw_action_features:
        return {}

    try:
        payload = json.loads(raw_action_features)
    except (TypeError, json.JSONDecodeError):
        return {}

    result_payload = payload.get("result")
    if not isinstance(result_payload, dict):
        result_payload = {}

    applied_caps = result_payload.get("applied_caps")
    if not isinstance(applied_caps, list):
        applied_caps = []

    reason_codes = result_payload.get("reason_codes")
    if not isinstance(reason_codes, list):
        reason_codes = []

    explanation = result_payload.get("explanation")
    if not isinstance(explanation, str):
        explanation = ""

    return {
        "final_cleanliness_score": payload.get("final_cleanliness_score", result_payload.get("final_cleanliness_score")),
        "cleaning_status": result_payload.get("cleaning_status"),
        "action_score": result_payload.get("action_score"),
        "visual_score": result_payload.get("visual_score"),
        "applied_caps": applied_caps,
        "reason_codes": reason_codes,
        "explanation": explanation,
    }


def with_action_workflow_report_details(record: dict[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    payload["action_workflow_details"] = action_workflow_report_details(record)
    return payload


def form_flag_enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "on", "yes"}


def merge_report_filter_options(
    pop_options: dict[str, list[str]],
    cleanliness_options: dict[str, list[str]],
    config_options: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    config_options = config_options or {}
    options = dict(pop_options)
    for key in ("store_name", "cctv_id", "roi_name"):
        options[key] = sorted(
            {
                *pop_options.get(key, []),
                *cleanliness_options.get(key, []),
                *config_options.get(key, []),
            }
        )
    options["cleanliness_mode"] = cleanliness_options.get("mode", [])
    options["cleanliness_decision"] = cleanliness_options.get("decision", [])
    options["cleanliness_final_stage"] = cleanliness_options.get("final_stage", [])
    return options


def config_report_filter_options(configs: list[Any]) -> dict[str, list[str]]:
    options = {"store_name": set(), "cctv_id": set(), "roi_name": set()}
    for config in configs:
        options["store_name"].add(config.store_name)
        options["cctv_id"].add(config.cctv_id)
        for roi in config.areas:
            options["roi_name"].add(roi.name)
    return {key: sorted(values) for key, values in options.items()}


def config_matches_report_filters(config: Any, filters: dict[str, str | None]) -> bool:
    if filters.get("store_name") and config.store_name != filters["store_name"]:
        return False
    if filters.get("cctv_id") and config.cctv_id != filters["cctv_id"]:
        return False
    if filters.get("roi_name") and all(roi.name != filters["roi_name"] for roi in config.areas):
        return False
    return True


def merge_configured_store_summary(
    summary_rows: list[dict[str, Any]],
    configs: list[Any],
    filters: dict[str, str | None],
) -> list[dict[str, Any]]:
    if filters.get("mode") or filters.get("decision") or filters.get("final_stage"):
        return summary_rows

    rows = list(summary_rows)
    existing_store_names = {row["store_name"] for row in rows}
    for config in configs:
        if config.store_name in existing_store_names:
            continue
        if not config_matches_report_filters(config, filters):
            continue
        rows.append(
            {
                "store_name": config.store_name,
                "total_count": 0,
                "cleaned_count": 0,
                "needs_check_count": 0,
                "unknown_count": 0,
                "average_confidence": None,
                "average_score": None,
                "latest_analyzed_at": "-",
            }
        )
        existing_store_names.add(config.store_name)
    return sorted(rows, key=lambda row: (-int(row["needs_check_count"] or 0), -int(row["unknown_count"] or 0), row["store_name"]))


def resolve_roi_for_action(config: Any, table_roi_index: str | int | None, roi_name: str | None) -> tuple[ROI, int]:
    if table_roi_index not in (None, ""):
        try:
            index = int(table_roi_index)
        except (TypeError, ValueError) as exc:
            raise KeyError("invalid table roi index") from exc
        if index < 0 or index >= len(config.areas):
            raise KeyError("table roi index out of range")
        return config.areas[index], index

    if roi_name:
        for index, roi in enumerate(config.areas):
            if roi.name == roi_name:
                return roi, index

    raise KeyError("roi not found")


def workflow_observations_from_frames(
    frames: list[Any],
    *,
    customer_in_use_seconds: int,
    meal_end_seconds: int,
) -> list[TableOccupancySample]:
    sorted_frames = sorted(frames, key=lambda item: item.captured_at)
    observations: list[TableOccupancySample] = []

    for index, frame in enumerate(sorted_frames):
        observations.append(
            TableOccupancySample(
                timestamp=frame.captured_at,
                customer_present=frame.person_present,
            )
        )
        if index + 1 >= len(sorted_frames):
            continue

        next_frame = sorted_frames[index + 1]
        if frame.person_present and next_frame.captured_at - frame.captured_at >= timedelta(seconds=customer_in_use_seconds):
            observations.append(
                TableOccupancySample(
                    timestamp=frame.captured_at + timedelta(seconds=customer_in_use_seconds),
                    customer_present=True,
                )
            )
        if (not frame.person_present) and next_frame.captured_at - frame.captured_at >= timedelta(seconds=meal_end_seconds):
            observations.append(
                TableOccupancySample(
                    timestamp=frame.captured_at + timedelta(seconds=meal_end_seconds),
                    customer_present=False,
                )
            )

    return sorted(observations, key=lambda item: item.timestamp)


def workflow_visual_state_from_frame(
    frame: Any,
    *,
    table_id: str,
) -> YoloTableState:
    payload = dict(frame.payload)
    payload.setdefault("table_id", table_id)
    payload.setdefault("captured_at", frame.captured_at.isoformat(timespec="seconds"))
    return normalize_visual_metric_input(payload, table_id=table_id)


def workflow_staff_visits_from_request(payload: ActionWorkflowRequest) -> list[StaffZoneVisit]:
    visits: list[StaffZoneVisit] = []
    for visit in payload.staff_zone_visits:
        visits.append(
            StaffZoneVisit.from_dict(
                {
                    "visit_id": visit.visit_id,
                    "store_id": payload.store_id,
                    "staff_id": visit.staff_id,
                    "zone_id": visit.zone_id,
                    "entered_at": visit.entered_at.isoformat(timespec="seconds"),
                    "left_at": visit.left_at.isoformat(timespec="seconds"),
                    "dwell_seconds": visit.dwell_seconds,
                    "mean_confidence": visit.mean_confidence,
                    "sample_count": visit.sample_count,
                }
            )
        )
    return visits


def execute_action_cleanliness_workflow(payload: ActionWorkflowRequest) -> dict[str, Any]:
    observations = workflow_observations_from_frames(
        payload.frames,
        customer_in_use_seconds=action_cleanliness_service.customer_in_use_seconds,
        meal_end_seconds=action_cleanliness_service.meal_end_seconds,
    )
    meal_session = action_cleanliness_service.evaluate_meal_session(payload.table_id, observations)
    if meal_session is None:
        raise HTTPException(status_code=400, detail="insufficient frame timeline to create a meal session")
    if meal_session.meal_ended_at is None:
        raise HTTPException(status_code=400, detail="frame timeline did not reach MEAL_ENDED")

    zone_mapping = ZoneTableMapping.from_dict(
        {
            "store_id": payload.store_id,
            "zones": [
                {
                    "zone_id": payload.zone_id,
                    "beacon_ids": [],
                    "table_ids": [payload.table_id],
                }
            ],
        }
    )
    cleaning_task = action_cleanliness_service.create_cleaning_task(meal_session, zone_mapping)

    sorted_frames = sorted(payload.frames, key=lambda item: item.captured_at)
    after_frame = sorted_frames[-1]
    before_frame = next(
        (frame for frame in reversed(sorted_frames) if frame.captured_at <= meal_session.meal_ended_at),
        sorted_frames[0],
    )
    before_state = workflow_visual_state_from_frame(before_frame, table_id=payload.table_id)
    after_state = workflow_visual_state_from_frame(after_frame, table_id=payload.table_id)
    staff_zone_visits = workflow_staff_visits_from_request(payload)

    action_result = action_cleanliness_service.evaluate_cleaning_task(
        cleaning_task,
        before_state=before_state,
        after_state=after_state,
        staff_zone_visits=staff_zone_visits,
        meal_session=meal_session,
        as_of=after_frame.captured_at,
    )
    final_result = build_final_cleanliness_result(action_result, after_state)
    merged_reason_codes = list(dict.fromkeys([*action_result.reason_codes, *final_result.reason_codes]))

    response_payload = {
        "store_id": payload.store_id,
        "table_id": payload.table_id,
        "zone_id": payload.zone_id,
        "saved": False,
        "result_id": None,
        "meal_status": action_result.meal_status,
        "cleaning_status": action_result.cleaning_status,
        "action_score": round(action_result.action_score, 2),
        "visual_score": final_result.visual_score,
        "visual_clean_score": round(final_result.visual_clean_score, 2),
        "visual_mess_score": round(final_result.visual_mess_score, 2),
        "final_cleanliness_score": final_result.final_cleanliness_score,
        "final_grade": final_result.final_grade,
        "decision": final_result.decision,
        "applied_caps": final_result.applied_caps,
        "reason_codes": merged_reason_codes,
        "explanation": final_result.explanation,
    }
    return {
        "response": response_payload,
        "action_result": action_result,
        "final_result": final_result,
        "after_state": after_state,
        "evaluated_at": after_frame.captured_at,
    }


def run_action_cleanliness_workflow(payload: ActionWorkflowRequest) -> dict[str, Any]:
    return execute_action_cleanliness_workflow(payload)["response"]


def workflow_report_decision(cleaning_status: str) -> str:
    if cleaning_status == "CLEANED_LIKELY":
        return "cleaned_likely"
    if cleaning_status in {"CLEANING_UNVERIFIED", "MISSED_SUSPECTED", "NEED_CLEANING"}:
        return "needs_check"
    return "unknown"


def report_score_from_final_cleanliness_score(final_cleanliness_score: int) -> int:
    if final_cleanliness_score >= 90:
        return 5
    if final_cleanliness_score >= 70:
        return 4
    if final_cleanliness_score >= 50:
        return 3
    if final_cleanliness_score >= 30:
        return 2
    return 1


def parse_demo_json_field(raw_value: str | None, field_name: str, *, default: Any) -> Any:
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid {field_name}") from exc


def parse_interaction_roi_json(raw_value: str | None) -> ROI | None:
    payload = parse_demo_json_field(raw_value, "interaction_roi_json", default=None)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="invalid interaction_roi_json")
    try:
        return ROI.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="invalid interaction_roi_json") from exc


def parse_staff_zone_visits_json(raw_value: str | None) -> list[dict[str, Any]]:
    payload = parse_demo_json_field(raw_value, "staff_zone_visits_json", default=[])
    if not isinstance(payload, list):
        raise HTTPException(status_code=400, detail="invalid staff_zone_visits_json")
    return payload


def normalize_visual_payload_source(raw_value: str | None, *, visual_payloads_json: str | None = None) -> str:
    normalized = str(raw_value or "").strip().lower()
    if not normalized:
        return VISUAL_PAYLOAD_SOURCE_JSON if (visual_payloads_json or "").strip() else VISUAL_PAYLOAD_SOURCE_PRESET
    if normalized in {
        VISUAL_PAYLOAD_SOURCE_PRESET,
        VISUAL_PAYLOAD_SOURCE_JSON,
        VISUAL_PAYLOAD_SOURCE_YOLO,
    }:
        return normalized
    raise HTTPException(status_code=400, detail="invalid visual_payload_source")


def demo_visual_payload_template(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    before_payload = {
        "objects": [
            {"class": "cup", "confidence": 0.91},
            {"class": "tray", "confidence": 0.84},
        ],
        "vision_confidence": 0.84,
    }
    if name == "cleaned_likely":
        return before_payload, {
            "objects": [],
            "vision_confidence": 0.84,
        }
    if name == "cleaning_unverified":
        return before_payload, dict(before_payload)
    if name == "missed_suspected":
        return before_payload, dict(before_payload)
    if name == "high_mess_cap":
        return before_payload, {
            "yolo_mess_score": 0.80,
            "detected_objects": [
                {"class": "trash", "count": 1, "max_confidence": 0.71},
            ],
            "vision_confidence": 0.88,
        }
    raise HTTPException(status_code=400, detail="invalid preset")


def resolve_demo_visual_payloads(
    *,
    frame_count: int,
    visual_payload_source: str | None,
    visual_payloads_json: str | None,
    preset: str | None,
) -> list[dict[str, Any]]:
    resolved_source = normalize_visual_payload_source(
        visual_payload_source,
        visual_payloads_json=visual_payloads_json,
    )
    if resolved_source == VISUAL_PAYLOAD_SOURCE_JSON:
        payload = parse_demo_json_field(visual_payloads_json, "visual_payloads_json", default=None)
        if payload is None:
            raise HTTPException(status_code=400, detail="visual_payloads_json is required when visual_payload_source=json")
        if isinstance(payload, dict):
            return [dict(payload) for _ in range(frame_count)]
        if not isinstance(payload, list):
            raise HTTPException(status_code=400, detail="invalid visual_payloads_json")
        if len(payload) != frame_count:
            raise HTTPException(status_code=400, detail="visual_payloads_json length must match file count")
        if not all(isinstance(item, dict) for item in payload):
            raise HTTPException(status_code=400, detail="invalid visual_payloads_json")
        return [dict(item) for item in payload]

    if preset:
        before_payload, after_payload = demo_visual_payload_template(preset)
        payloads = [dict(before_payload) for _ in range(frame_count)]
        if payloads:
            payloads[-1] = dict(after_payload)
        return payloads

    return [{} for _ in range(frame_count)]


def build_visual_payload_from_image_path_with_yolo(
    image_path: Path,
    *,
    table_id: str,
    yolo_helper: Any,
) -> dict[str, Any]:
    try:
        detections = yolo_helper.detect_objects(image_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return build_visual_payload_from_yolo_detections(detections, table_id=table_id)


def build_yolo_visual_payloads_from_image_paths(
    image_paths: list[Path],
    *,
    table_id: str,
    yolo_helper: Any,
) -> list[dict[str, Any]]:
    return [
        build_visual_payload_from_image_path_with_yolo(path, table_id=table_id, yolo_helper=yolo_helper)
        for path in image_paths
    ]


def build_yolo_visual_payloads_from_video_samples(
    samples: list[dict[str, Any]],
    *,
    table_id: str,
    temp_dir: Path,
    yolo_helper: Any,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        frame_path = temp_dir / f"workflow_video_frame_{index:02d}.png"
        write_image(frame_path, sample["image"])
        payloads.append(
            build_visual_payload_from_image_path_with_yolo(frame_path, table_id=table_id, yolo_helper=yolo_helper)
        )
    return payloads


def build_action_workflow_response(payload: ActionWorkflowRequest) -> dict[str, Any]:
    workflow_run = execute_action_cleanliness_workflow(payload)
    response_payload = dict(workflow_run["response"])
    if payload.save_result:
        response_payload["result_id"] = save_action_cleanliness_workflow_result(payload, workflow_run)
        response_payload["saved"] = True
    response_payload["frames"] = payload.model_dump(mode="json")["frames"]
    return response_payload


def save_action_cleanliness_workflow_result(payload: ActionWorkflowRequest, workflow_run: dict[str, Any]) -> int:
    response_payload = dict(workflow_run["response"])
    action_result = workflow_run["action_result"]
    after_state = workflow_run["after_state"]
    exact_objects = [
        item.object_class if item.count == 1 else f"{item.object_class} x{item.count}"
        for item in after_state.detected_objects
    ]
    action_features = {
        "request": payload.model_dump(mode="json"),
        "result": response_payload,
        "final_cleanliness_score": response_payload["final_cleanliness_score"],
        "action_confidence": round(action_result.action_confidence, 2),
        "visual_penalties_breakdown": workflow_run["final_result"].penalties_breakdown,
        "visual_bonuses_breakdown": workflow_run["final_result"].bonuses_breakdown,
    }
    return insert_cleanliness_result(
        {
            "analyzed_at": workflow_run["evaluated_at"].isoformat(timespec="minutes"),
            "store_name": payload.store_id,
            "cctv_id": f"{payload.store_id}::{payload.zone_id}",
            "cctv_nickname": payload.zone_id,
            "roi_name": payload.table_id,
            "mode": "action_workflow",
            "decision": workflow_report_decision(action_result.cleaning_status),
            "score": report_score_from_final_cleanliness_score(response_payload["final_cleanliness_score"]),
            "confidence": action_result.action_confidence,
            "final_stage": "workflow_api",
            "summary": response_payload["explanation"],
            "source_path": "",
            "crop_path": "",
            "exact_objects": dump_json(exact_objects),
            "estimated_objects": dump_json([]),
            "findings": dump_json(response_payload["reason_codes"]),
            "action_features": dump_json(action_features),
        }
    )


def save_cleanliness_record(
    *,
    analyzed_at: datetime,
    config: Any,
    roi: ROI,
    mode: str,
    decision: str,
    score: int | None,
    confidence: float,
    final_stage: str,
    summary: str,
    source_path: Path | None = None,
    crop_path: Path | None = None,
    exact_objects: list[str] | None = None,
    estimated_objects: list[str] | None = None,
    findings: list[str] | None = None,
    action_features: dict[str, Any] | None = None,
) -> int:
    return insert_cleanliness_result(
        {
            "analyzed_at": analyzed_at.isoformat(timespec="minutes"),
            "store_name": config.store_name,
            "cctv_id": config.cctv_id,
            "cctv_nickname": config.cctv_nickname,
            "roi_name": roi.name,
            "mode": mode,
            "decision": decision,
            "score": score,
            "confidence": confidence,
            "final_stage": final_stage,
            "summary": summary,
            "source_path": str(source_path) if source_path is not None else "",
            "crop_path": str(crop_path) if crop_path is not None else "",
            "exact_objects": dump_json(exact_objects or []),
            "estimated_objects": dump_json(estimated_objects or []),
            "findings": dump_json(findings or []),
            "action_features": dump_json(action_features or {}),
        }
    )


@app.get("/cleanliness")
def cleanliness_page(request: Request, config_id: str | None = Query(default=None)) -> Any:
    configs = config_store.list_configs()
    selected = config_payload(config_id)
    return templates.TemplateResponse(
        request,
        "cleanliness.html",
        {
            "request": request,
            "title": "청결도 검사",
            "configs": configs,
            "configs_payload": [config.to_dict() for config in configs],
            "selected_config": selected,
            "selected_roi_name": "",
            "selected_prompt_profile": DEFAULT_CLEANLINESS_PROMPT_PROFILE,
            "selected_use_yolo": False,
        },
    )


@app.post("/cleanliness")
async def cleanliness_upload(
    request: Request,
    config_id: str = Form(...),
    roi_name: str = Form(...),
    prompt_profile: str = Form(default=DEFAULT_CLEANLINESS_PROMPT_PROFILE),
    use_yolo: str = Form(default="false"),
    image_file: UploadFile = File(...),
) -> Any:
    try:
        config = config_store.load(config_id)
        roi = config_store.get_roi(config_id, roi_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="config not found") from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="roi not found") from exc

    suffix = Path(image_file.filename or "").suffix.lower()
    if suffix not in CLEANLINESS_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="image file must be png, jpg, jpeg, bmp, or webp")

    analyzed_at = datetime.now()
    stem = f"{config_id}_{roi_name}_cleanliness_{analyzed_at.strftime('%Y%m%d%H%M%S')}"
    source_path = save_upload(image_file, UPLOAD_DIR, stem)
    crop_path = save_analysis_crop(read_image(source_path), roi, stem)
    selected_prompt_profile = normalize_cleanliness_prompt_profile(prompt_profile)
    use_yolo_flag = form_flag_enabled(use_yolo)

    try:
        result = cleanliness_service.inspect_image(
            source_path,
            inspected_path=crop_path,
            prompt_profile=selected_prompt_profile,
            use_yolo=use_yolo_flag,
            roi=roi,
            output_stem=stem,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        detail = str(exc)
        status_code = 503 if "OPENAI_API_KEY" in detail else 502
        raise HTTPException(status_code=status_code, detail=detail) from exc

    payload = result.to_dict()
    payload["store_name"] = config.store_name
    payload["cctv_nickname"] = config.cctv_nickname
    payload["roi_name"] = roi.name
    payload["source_url"] = data_url(result.source_path)
    payload["crop_url"] = data_url(crop_path)
    payload["llm_input_url"] = data_url(result.inspected_path)
    payload["llm_input_urls"] = [data_url(path) for path in result.llm_input_paths]
    yolo_payload = result.yolo_payload or {}
    crop_annotated_path = Path(str(yolo_payload.get("crop_annotated_path") or "")) if yolo_payload.get("crop_annotated_path") else None
    source_annotated_path = Path(str(yolo_payload.get("source_annotated_path") or "")) if yolo_payload.get("source_annotated_path") else None
    payload["crop_annotated_url"] = data_url(crop_annotated_path) if crop_annotated_path else ""
    payload["source_annotated_url"] = data_url(source_annotated_path) if source_annotated_path else ""
    payload["pipeline_label"] = "YOLO + LLM" if result.use_yolo else "LLM only"
    payload["yolo_payload_pretty"] = json.dumps(result.yolo_payload, ensure_ascii=False, indent=2) if result.yolo_payload else ""
    save_cleanliness_record(
        analyzed_at=analyzed_at,
        config=config,
        roi=roi,
        mode="object_yolo" if result.use_yolo else "object",
        decision=object_score_to_decision(result.score),
        score=result.score,
        confidence=result.confidence,
        final_stage="object_based_yolo" if result.use_yolo else "object_based",
        summary=result.summary,
        source_path=result.source_path,
        crop_path=crop_path,
        exact_objects=result.exact_objects,
        estimated_objects=result.estimated_objects,
        findings=result.findings,
        action_features={
            "use_yolo": result.use_yolo,
            "yolo_payload": result.yolo_payload or {},
        },
    )

    configs = config_store.list_configs()

    return templates.TemplateResponse(
        request,
        "cleanliness.html",
        {
            "request": request,
            "title": "청결도 검사",
            "configs": configs,
            "configs_payload": [item.to_dict() for item in configs],
            "selected_config": config.to_dict(),
            "selected_roi_name": roi.name,
            "selected_prompt_profile": selected_prompt_profile,
            "selected_use_yolo": use_yolo_flag,
            "cleanliness_result": payload,
        },
    )


@app.get("/api/mobile/upload-policy")
def mobile_upload_policy(owner_id: str = Depends(current_owner_id)) -> dict[str, Any]:
    _ = owner_id
    return {
        "upload_interval_seconds": MOBILE_VIDEO_UPLOAD_INTERVAL_SECONDS,
        "openai_configured": bool(OPENAI_API_KEY),
        "openai_model": OPENAI_MODEL,
        "video_extensions": sorted(VIDEO_CLEANLINESS_EXTENSIONS),
        "configs": [config.to_dict() for config in config_store.list_configs()],
    }


def run_mobile_video_cleanliness_job(
    *,
    record_id: int,
    job_id: str,
    analyzed_at: datetime,
    config: Any,
    roi: ROI,
    source_path: Path,
    prompt_profile: str,
    device_id: str,
    captured_at: str,
    upload_period_seconds: float | None,
    enable_person_masking: bool,
) -> None:
    try:
        result = video_cleanliness_service.inspect_video(
            config=config,
            roi=roi,
            video_path=source_path,
            prompt_profile=prompt_profile,
            device_id=device_id,
            captured_at=captured_at,
            upload_period_seconds=upload_period_seconds,
            enable_person_masking=enable_person_masking,
        )
        decision = object_score_to_decision(result.score)
        update_cleanliness_result(
            record_id,
            {
                "decision": decision,
                "score": result.score,
                "confidence": result.confidence,
                "final_stage": "video_external",
                "summary": result.summary,
                "source_path": str(result.source_path),
                "crop_path": "",
                "exact_objects": dump_json(result.exact_objects),
                "estimated_objects": dump_json(result.estimated_objects),
                "findings": dump_json(result.findings),
                "action_features": dump_json(
                    {
                        "job_id": job_id,
                        "device_id": device_id,
                        "captured_at": captured_at,
                        "upload_period_seconds": upload_period_seconds,
                        "analysis_url": result.analysis_url,
                        "person_masking_enabled": result.person_masking_enabled,
                        "person_masking_applied": result.person_masking_applied,
                        "person_count": result.person_count,
                        "masked_pixel_ratio": result.masked_pixel_ratio,
                        "person_masked_path": str(result.person_masked_path) if result.person_masked_path else "",
                    }
                ),
            },
        )
    except Exception as exc:
        update_cleanliness_result(
            record_id,
            {
                "decision": "unknown",
                "score": None,
                "confidence": 0.0,
                "final_stage": "video_failed",
                "summary": f"Video analysis failed: {exc}",
                "source_path": str(source_path),
                "crop_path": "",
                "exact_objects": dump_json([]),
                "estimated_objects": dump_json([]),
                "findings": dump_json([str(exc)]),
                "action_features": dump_json(
                    {
                        "job_id": job_id,
                        "device_id": device_id,
                        "captured_at": captured_at,
                        "upload_period_seconds": upload_period_seconds,
                        "person_masking_enabled": enable_person_masking,
                        "person_masking_applied": False,
                        "error": str(exc),
                    }
                ),
            },
        )
        print(
            json.dumps(
                {
                    "type": "mobile_video_analysis_failed",
                    "payload": {
                        "job_id": job_id,
                        "store_name": config.store_name,
                        "cctv_id": config.cctv_id,
                        "roi_name": roi.name,
                        "source_path": str(source_path),
                        "error": str(exc),
                    },
                },
                ensure_ascii=False,
            )
        )


def load_action_features(record: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(record.get("action_features") or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def mobile_job_payload(job_id: str, record: dict[str, Any]) -> dict[str, Any]:
    score = record.get("score")
    final_stage = str(record.get("final_stage") or "")
    if final_stage == "video_failed":
        status = "failed"
    elif final_stage == "queued":
        status = "queued"
    elif score is not None:
        status = "completed"
    else:
        status = "processing"

    low_score_threshold = 2
    is_low_score = isinstance(score, (int, float)) and score <= low_score_threshold
    action_features = load_action_features(record)
    return {
        "job_id": job_id,
        "status": status,
        "analyzed_at": record.get("analyzed_at"),
        "store_name": record.get("store_name"),
        "cctv_id": record.get("cctv_id"),
        "cctv_nickname": record.get("cctv_nickname"),
        "roi_name": record.get("roi_name"),
        "mode": record.get("mode"),
        "final_stage": final_stage,
        "decision": record.get("decision"),
        "score": score,
        "confidence": record.get("confidence"),
        "summary": record.get("summary"),
        "is_low_score": is_low_score,
        "low_score_threshold": low_score_threshold,
        "person_masking_enabled": action_features.get("person_masking_enabled") is True,
        "person_masking_applied": action_features.get("person_masking_applied") is True,
        "person_count": action_features.get("person_count") or 0,
        "masked_pixel_ratio": action_features.get("masked_pixel_ratio") or 0.0,
    }


@app.get("/api/mobile/jobs/{job_id}")
def mobile_cleanliness_job_result(
    job_id: str,
    owner_id: str = Depends(current_owner_id),
) -> dict[str, Any]:
    _ = owner_id
    record = fetch_cleanliness_result_by_job_id(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="job not found")
    return mobile_job_payload(job_id, record)


@app.post("/api/mobile/cleanliness-video")
async def mobile_cleanliness_video_upload(
    background_tasks: BackgroundTasks,
    owner_id: str = Depends(current_owner_id),
    config_id: str = Form(...),
    roi_name: str = Form(...),
    prompt_profile: str = Form(default=DEFAULT_CLEANLINESS_PROMPT_PROFILE),
    device_id: str = Form(default=""),
    captured_at: str = Form(default=""),
    upload_period_seconds: float | None = Form(default=None),
    enable_person_masking: bool = Form(default=False),
    video_file: UploadFile = File(...),
) -> JSONResponse:
    _ = owner_id
    try:
        config = config_store.load(config_id)
        roi = config_store.get_roi(config_id, roi_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="config not found") from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="roi not found") from exc

    suffix = Path(video_file.filename or "").suffix.lower()
    if suffix not in VIDEO_CLEANLINESS_EXTENSIONS:
        raise HTTPException(status_code=400, detail="video file must be mp4, mov, m4v, avi, webm, or mkv")
    if upload_period_seconds is not None and upload_period_seconds <= 0:
        raise HTTPException(status_code=400, detail="upload_period_seconds must be greater than 0")

    analyzed_at = datetime.now()
    device_part = safe_filename_part(device_id) if device_id else "mobile"
    roi_part = safe_filename_part(roi_name)
    stem = f"{config_id}_{roi_part}_{device_part}_video_cleanliness_{analyzed_at.strftime('%Y%m%d%H%M%S')}"
    source_path = save_upload(video_file, MOBILE_VIDEO_DIR, stem)
    job_id = uuid.uuid4().hex
    record_id = save_cleanliness_record(
        analyzed_at=analyzed_at,
        config=config,
        roi=roi,
        mode="video",
        decision="unknown",
        score=None,
        confidence=0.0,
        final_stage="queued",
        summary="Video analysis job accepted.",
        source_path=source_path,
        action_features={
            "job_id": job_id,
            "device_id": device_id,
            "captured_at": captured_at,
            "upload_period_seconds": upload_period_seconds,
            "prompt_profile": prompt_profile,
            "person_masking_enabled": enable_person_masking,
            "person_masking_applied": False,
        },
    )

    background_tasks.add_task(
        run_mobile_video_cleanliness_job,
        record_id=record_id,
        job_id=job_id,
        analyzed_at=analyzed_at,
        config=config,
        roi=roi,
        source_path=source_path,
        prompt_profile=prompt_profile,
        device_id=device_id,
        captured_at=captured_at,
        upload_period_seconds=upload_period_seconds,
        enable_person_masking=enable_person_masking,
    )
    return JSONResponse({"job_id": job_id}, status_code=202)


@app.get("/person-mask")
def person_mask_page(request: Request) -> Any:
    return templates.TemplateResponse(
        request,
        "person_mask.html",
        {
            "request": request,
            "title": "Person Masking",
        },
    )


@app.post("/person-mask")
async def person_mask_upload(
    request: Request,
    image_file: UploadFile = File(...),
) -> Any:
    suffix = Path(image_file.filename or "").suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="image file must be png, jpg, jpeg, bmp, or webp")

    stem = f"person_mask_source_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    source_path = save_upload(image_file, UPLOAD_DIR, stem)

    try:
        result = person_mask_service.mask_image_file(source_path, output_stem=source_path.stem)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload = result.to_dict()
    payload["source_url"] = data_url(result.source_path)
    payload["masked_url"] = data_url(result.masked_path)

    return templates.TemplateResponse(
        request,
        "person_mask.html",
        {
            "request": request,
            "title": "Person Masking",
            "mask_result": payload,
        },
    )


@app.get("/action-cleanliness")
def action_cleanliness_page(request: Request, config_id: str | None = Query(default=None)) -> Any:
    configs = config_store.list_configs()
    selected = config_payload(config_id)
    return templates.TemplateResponse(
        request,
        "action_cleanliness.html",
        {
            "request": request,
            "title": "Action 기반 청결도",
            "configs": configs,
            "configs_payload": [config.to_dict() for config in configs],
            "selected_config": selected,
            "selected_roi_name": "",
            "selected_table_roi_index": "",
            "order_time": SAMPLE_ORDER_TIME,
            "payment_completed_time": SAMPLE_PAYMENT_COMPLETED_TIME,
            "sample_trajectory_json": SAMPLE_TRAJECTORY_JSON,
            "trajectory_json": SAMPLE_TRAJECTORY_JSON,
            "near_margin": int(action_cleanliness_service.near_margin),
            "manual_confirmed": False,
        },
    )


@app.get("/action-workflow-demo")
def action_workflow_demo_page(request: Request) -> Any:
    return templates.TemplateResponse(
        request,
        "action_workflow_demo.html",
        {
            "request": request,
            "title": "Action Workflow Demo",
            "captured_at_start": "2026-06-03T14:10:20",
            "interval_seconds": 60,
            "max_frames": 10,
            "store_id": "store_001",
            "table_id": "T06",
            "zone_id": "zone_B",
            "save_result": False,
            "dynamic_sampling": True,
            "staff_zone_visits_json": ACTION_WORKFLOW_DEMO_STAFF_VISITS_JSON,
            "interaction_roi_json": "",
            "visual_payloads_json": "",
            "selected_visual_payload_source": VISUAL_PAYLOAD_SOURCE_PRESET,
            "selected_preset": "cleaned_likely",
        },
    )


@app.get("/frame-sampler-demo")
def frame_sampler_demo_page(request: Request) -> Any:
    configs = config_store.list_configs()
    return templates.TemplateResponse(
        request,
        "frame_sampler_demo.html",
        {
            "request": request,
            "title": "Frame Sampler Demo",
            "captured_at_start": "2026-06-03T14:10:20",
            "interval_seconds": 10,
            "max_frames": 12,
            "interaction_roi_json": "",
            "configs": configs,
            "configs_payload": [config.to_dict() for config in configs],
        },
    )


@app.post("/action-cleanliness")
async def action_cleanliness_submit(
    request: Request,
    config_id: str = Form(...),
    table_roi_index: str | None = Form(default=None),
    roi_name: str | None = Form(default=None),
    order_time: str = Form(default=SAMPLE_ORDER_TIME),
    payment_completed_time: str = Form(default=SAMPLE_PAYMENT_COMPLETED_TIME),
    trajectory_json: str = Form(...),
    near_margin: float = Form(default=80.0),
    manual_confirmed: bool = Form(default=False),
) -> Any:
    try:
        config = config_store.load(config_id)
        roi, resolved_table_roi_index = resolve_roi_for_action(config, table_roi_index, roi_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="config not found") from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="roi not found") from exc

    try:
        trajectory_points = parse_trajectory_json(trajectory_json)
        order_time_value = parse_action_time_value(order_time, "order_time")
        payment_completed_time_value = parse_action_time_value(payment_completed_time, "payment_completed_time")
        action_result = action_cleanliness_service.evaluate(
            roi,
            trajectory_points,
            near_margin=near_margin,
            manual_confirmed=manual_confirmed,
            order_time=order_time_value,
            payment_completed_time=payment_completed_time_value,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload = action_result.to_dict()
    payload["store_name"] = config.store_name
    payload["cctv_nickname"] = config.cctv_nickname
    payload["roi_name"] = roi.name
    payload["table_roi_index"] = resolved_table_roi_index
    payload["order_time"] = order_time
    payload["payment_completed_time"] = payment_completed_time
    save_cleanliness_record(
        analyzed_at=datetime.now(),
        config=config,
        roi=roi,
        mode="action",
        decision=action_result.decision,
        score=action_result.score,
        confidence=action_result.confidence,
        final_stage="action_only",
        summary=action_result.summary,
        action_features=action_result.features,
    )

    configs = config_store.list_configs()
    return templates.TemplateResponse(
        request,
        "action_cleanliness.html",
        {
            "request": request,
            "title": "Action 기반 청결도",
            "configs": configs,
            "configs_payload": [item.to_dict() for item in configs],
            "selected_config": config.to_dict(),
            "selected_roi_name": roi.name,
            "selected_table_roi_index": str(resolved_table_roi_index),
            "order_time": order_time,
            "payment_completed_time": payment_completed_time,
            "sample_trajectory_json": SAMPLE_TRAJECTORY_JSON,
            "trajectory_json": trajectory_json,
            "near_margin": near_margin,
            "manual_confirmed": manual_confirmed,
            "action_result": payload,
        },
    )


@app.post("/api/action-cleanliness/workflow")
async def action_cleanliness_workflow_api(payload: ActionWorkflowRequest) -> JSONResponse:
    return JSONResponse(build_action_workflow_response(payload))


@app.post("/api/action-cleanliness/workflow-video-candidates")
async def action_cleanliness_workflow_video_candidates(
    table_id: str = Form(...),
    captured_at_start: str = Form(...),
    interval_seconds: float = Form(default=10.0),
    max_frames: int = Form(default=10),
    interaction_roi_json: str = Form(default=""),
    video_file: UploadFile = File(...),
) -> JSONResponse:
    suffix = Path(video_file.filename or "").suffix.lower()
    if suffix not in VIDEO_CLEANLINESS_EXTENSIONS:
        raise HTTPException(status_code=400, detail="video file must be mp4, mov, m4v, avi, webm, or mkv")
    if interval_seconds <= 0:
        raise HTTPException(status_code=400, detail="interval_seconds must be greater than 0")
    if max_frames <= 0:
        raise HTTPException(status_code=400, detail="max_frames must be greater than 0")

    interaction_roi = parse_interaction_roi_json(interaction_roi_json)
    observation_budget = workflow_video_observation_budget(max_frames)
    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        video_path = save_upload(video_file, temp_dir, "workflow_video_candidates")
        samples = sample_dynamic_video_workflow_frames(
            video_path=video_path,
            max_frames=max_frames,
            observation_budget=observation_budget,
            interaction_roi=interaction_roi,
            person_mask_service=person_mask_service,
            sampling_config=workflow_video_sampling_config(interval_seconds, observation_budget),
        )

    return JSONResponse(
        serialize_dynamic_video_sampling_payload(
            samples,
            table_id=table_id,
            captured_at_start=captured_at_start,
            candidate_budget=max_frames,
        )
    )


@app.post("/api/action-cleanliness/workflow-from-images")
async def action_cleanliness_workflow_from_images(
    store_id: str = Form(...),
    table_id: str = Form(...),
    zone_id: str = Form(...),
    captured_at_start: str = Form(...),
    interval_seconds: float = Form(default=30.0),
    save_result: bool = Form(default=False),
    interaction_roi_json: str = Form(default=""),
    staff_zone_visits_json: str = Form(default=""),
    visual_payload_source: str = Form(default=""),
    visual_payloads_json: str = Form(default=""),
    preset: str = Form(default=""),
    image_files: list[UploadFile] = File(...),
) -> JSONResponse:
    if not image_files:
        raise HTTPException(status_code=400, detail="at least one image file is required")
    if interval_seconds <= 0:
        raise HTTPException(status_code=400, detail="interval_seconds must be greater than 0")

    interaction_roi = parse_interaction_roi_json(interaction_roi_json)
    staff_zone_visits = parse_staff_zone_visits_json(staff_zone_visits_json)
    resolved_visual_payload_source = normalize_visual_payload_source(
        visual_payload_source,
        visual_payloads_json=visual_payloads_json,
    )

    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        image_paths: list[Path] = []
        for index, image_file in enumerate(image_files):
            suffix = Path(image_file.filename or "").suffix.lower()
            if suffix not in IMAGE_EXTENSIONS:
                raise HTTPException(status_code=400, detail="image files must be png, jpg, jpeg, bmp, or webp")
            image_paths.append(save_upload(image_file, temp_dir, f"workflow_image_{index:02d}"))

        if resolved_visual_payload_source == VISUAL_PAYLOAD_SOURCE_YOLO:
            payloads = build_yolo_visual_payloads_from_image_paths(
                image_paths,
                table_id=table_id,
                yolo_helper=workflow_yolo_helper,
            )
        else:
            payloads = resolve_demo_visual_payloads(
                frame_count=len(image_files),
                visual_payload_source=resolved_visual_payload_source,
                visual_payloads_json=visual_payloads_json,
                preset=preset,
            )

        frames = build_workflow_frames_from_images(
            image_paths=image_paths,
            start_time=captured_at_start,
            interval_seconds=interval_seconds,
            table_id=table_id,
            interaction_roi=interaction_roi,
            payloads=payloads,
            person_mask_service=person_mask_service,
        )
    if len(frames) < 2:
        raise HTTPException(status_code=400, detail="at least two frames are required")

    payload = ActionWorkflowRequest(
        store_id=store_id,
        table_id=table_id,
        zone_id=zone_id,
        save_result=save_result,
        frames=frames,
        staff_zone_visits=staff_zone_visits,
    )
    return JSONResponse(build_action_workflow_response(payload))


@app.post("/api/action-cleanliness/workflow-from-video")
async def action_cleanliness_workflow_from_video(
    store_id: str = Form(...),
    table_id: str = Form(...),
    zone_id: str = Form(...),
    captured_at_start: str = Form(...),
    interval_seconds: float = Form(default=30.0),
    max_frames: int = Form(default=10),
    dynamic_sampling: bool = Form(default=True),
    save_result: bool = Form(default=False),
    interaction_roi_json: str = Form(default=""),
    staff_zone_visits_json: str = Form(default=""),
    visual_payload_source: str = Form(default=""),
    visual_payloads_json: str = Form(default=""),
    preset: str = Form(default=""),
    video_file: UploadFile = File(...),
) -> JSONResponse:
    suffix = Path(video_file.filename or "").suffix.lower()
    if suffix not in VIDEO_CLEANLINESS_EXTENSIONS:
        raise HTTPException(status_code=400, detail="video file must be mp4, mov, m4v, avi, webm, or mkv")
    if interval_seconds <= 0:
        raise HTTPException(status_code=400, detail="interval_seconds must be greater than 0")
    if max_frames <= 0:
        raise HTTPException(status_code=400, detail="max_frames must be greater than 0")

    interaction_roi = parse_interaction_roi_json(interaction_roi_json)
    staff_zone_visits = parse_staff_zone_visits_json(staff_zone_visits_json)
    resolved_visual_payload_source = normalize_visual_payload_source(
        visual_payload_source,
        visual_payloads_json=visual_payloads_json,
    )

    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        video_path = save_upload(video_file, temp_dir, "workflow_video")
        if dynamic_sampling:
            observation_budget = workflow_video_observation_budget(max_frames)
            samples = sample_dynamic_video_workflow_frames(
                video_path=video_path,
                max_frames=max_frames,
                observation_budget=observation_budget,
                interaction_roi=interaction_roi,
                person_mask_service=person_mask_service,
                sampling_config=workflow_video_sampling_config(interval_seconds, observation_budget),
            )
        else:
            samples = sample_video_workflow_frames(
                video_path=video_path,
                interval_seconds=interval_seconds,
                max_frames=max_frames,
            )
        if resolved_visual_payload_source == VISUAL_PAYLOAD_SOURCE_YOLO:
            payloads = build_yolo_visual_payloads_from_video_samples(
                list(samples),
                table_id=table_id,
                temp_dir=temp_dir,
                yolo_helper=workflow_yolo_helper,
            )
        else:
            payloads = resolve_demo_visual_payloads(
                frame_count=len(samples),
                visual_payload_source=resolved_visual_payload_source,
                visual_payloads_json=visual_payloads_json,
                preset=preset,
            )
        frames = build_workflow_frames_from_video(
            video_path=video_path,
            captured_at_start=captured_at_start,
            interval_seconds=interval_seconds,
            max_frames=max_frames,
            table_id=table_id,
            interaction_roi=interaction_roi,
            payload_builder=lambda _image, index, **_: {
                **payloads[index],
                **(sampler_metadata_payload(samples[index]) if dynamic_sampling else {}),
            },
            person_mask_service=person_mask_service,
            frame_extractor=lambda _path, **__: list(samples),
        )
        if dynamic_sampling:
            apply_dynamic_sample_occupancy(frames, list(samples))
    if len(frames) < 2:
        raise HTTPException(status_code=400, detail="at least two frames are required")

    payload = ActionWorkflowRequest(
        store_id=store_id,
        table_id=table_id,
        zone_id=zone_id,
        save_result=save_result,
        frames=frames,
        staff_zone_visits=staff_zone_visits,
    )
    response_payload = build_action_workflow_response(payload)
    if dynamic_sampling:
        response_payload.update(
            serialize_dynamic_video_sampling_payload(
                samples,
                table_id=table_id,
                captured_at_start=captured_at_start,
                candidate_budget=max_frames,
            )
        )
    return JSONResponse(response_payload)


@app.get("/hybrid-cleanliness")
def hybrid_cleanliness_page(request: Request, config_id: str | None = Query(default=None)) -> Any:
    configs = config_store.list_configs()
    selected = config_payload(config_id)
    return templates.TemplateResponse(
        request,
        "hybrid_cleanliness.html",
        {
            "request": request,
            "title": "Hybrid 청결도",
            "configs": configs,
            "configs_payload": [config.to_dict() for config in configs],
            "selected_config": selected,
            "selected_roi_name": "",
            "selected_prompt_profile": DEFAULT_CLEANLINESS_PROMPT_PROFILE,
            "sample_trajectory_json": SAMPLE_TRAJECTORY_JSON,
            "trajectory_json": SAMPLE_TRAJECTORY_JSON,
            "near_margin": int(action_cleanliness_service.near_margin),
            "manual_confirmed": False,
            "object_confidence_threshold": OBJECT_CONFIDENCE_THRESHOLD,
        },
    )


@app.post("/hybrid-cleanliness")
async def hybrid_cleanliness_submit(
    request: Request,
    config_id: str = Form(...),
    roi_name: str = Form(...),
    prompt_profile: str = Form(default=DEFAULT_CLEANLINESS_PROMPT_PROFILE),
    trajectory_json: str = Form(...),
    near_margin: float = Form(default=80.0),
    manual_confirmed: bool = Form(default=False),
    image_file: UploadFile = File(...),
) -> Any:
    try:
        config = config_store.load(config_id)
        roi = config_store.get_roi(config_id, roi_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="config not found") from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="roi not found") from exc

    suffix = Path(image_file.filename or "").suffix.lower()
    if suffix not in CLEANLINESS_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="image file must be png, jpg, jpeg, bmp, or webp")

    analyzed_at = datetime.now()
    stem = f"{config_id}_{roi_name}_hybrid_cleanliness_{analyzed_at.strftime('%Y%m%d%H%M%S')}"
    source_path = save_upload(image_file, UPLOAD_DIR, stem)
    crop_path = save_analysis_crop(read_image(source_path), roi, stem)
    selected_prompt_profile = normalize_cleanliness_prompt_profile(prompt_profile)

    try:
        object_result = cleanliness_service.inspect_image(
            source_path,
            inspected_path=crop_path,
            prompt_profile=selected_prompt_profile,
        )

        action_result = None
        if object_result.confidence < OBJECT_CONFIDENCE_THRESHOLD:
            trajectory_points = parse_trajectory_json(trajectory_json)
            action_result = action_cleanliness_service.evaluate(
                roi,
                trajectory_points,
                near_margin=near_margin,
                manual_confirmed=manual_confirmed,
            )

        hybrid_result = build_hybrid_cleanliness_result(
            object_result,
            action_result,
            object_confidence_threshold=OBJECT_CONFIDENCE_THRESHOLD,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        detail = str(exc)
        status_code = 503 if "OPENAI_API_KEY" in detail else 502
        raise HTTPException(status_code=status_code, detail=detail) from exc

    payload = hybrid_result.to_dict()
    payload["store_name"] = config.store_name
    payload["cctv_nickname"] = config.cctv_nickname
    payload["roi_name"] = roi.name
    payload["object_result"]["source_url"] = data_url(object_result.source_path)
    payload["object_result"]["crop_url"] = data_url(crop_path)
    payload["object_result"]["llm_input_url"] = data_url(object_result.inspected_path)
    save_cleanliness_record(
        analyzed_at=analyzed_at,
        config=config,
        roi=roi,
        mode="hybrid",
        decision=hybrid_result.final_decision,
        score=hybrid_result.action_result.score if hybrid_result.action_result is not None else object_result.score,
        confidence=hybrid_result.final_confidence,
        final_stage=hybrid_result.final_stage,
        summary=hybrid_result.summary,
        source_path=object_result.source_path,
        crop_path=crop_path,
        exact_objects=object_result.exact_objects,
        estimated_objects=object_result.estimated_objects,
        findings=object_result.findings,
        action_features=hybrid_result.action_result.features if hybrid_result.action_result is not None else {},
    )

    configs = config_store.list_configs()
    return templates.TemplateResponse(
        request,
        "hybrid_cleanliness.html",
        {
            "request": request,
            "title": "Hybrid 청결도",
            "configs": configs,
            "configs_payload": [item.to_dict() for item in configs],
            "selected_config": config.to_dict(),
            "selected_roi_name": roi.name,
            "selected_prompt_profile": selected_prompt_profile,
            "sample_trajectory_json": SAMPLE_TRAJECTORY_JSON,
            "trajectory_json": trajectory_json,
            "near_margin": near_margin,
            "manual_confirmed": manual_confirmed,
            "object_confidence_threshold": OBJECT_CONFIDENCE_THRESHOLD,
            "hybrid_result": payload,
        },
    )


@app.post("/analyze")
async def analyze_upload(
    request: Request,
    config_id: str = Form(...),
    roi_name: str = Form(...),
    sensor_brightness: float | None = Form(default=None),
    enable_sensor_match: bool = Form(default=False),
    media_file: UploadFile = File(...),
    poster_template_upload: UploadFile | None = File(default=None),
    poster_template_name: str = Form(default=""),
) -> Any:
    try:
        config = config_store.load(config_id)
        roi = config_store.get_roi(config_id, roi_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="config not found") from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="roi not found") from exc

    media_path = save_upload(media_file, UPLOAD_DIR, f"{config_id}_{roi_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}")

    if poster_template_upload and poster_template_upload.filename:
        poster_path = save_upload(
            poster_template_upload,
            POSTER_TEMPLATE_DIR,
            f"{config_id}_{roi_name}_poster_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        )
    elif poster_template_name:
        poster_path = POSTER_TEMPLATE_DIR / poster_template_name
        if not poster_path.exists():
            raise HTTPException(status_code=400, detail="poster template not found")
    else:
        raise HTTPException(status_code=400, detail="poster template is required")

    try:
        result = analysis_service.analyze_media(
            config=config,
            roi=roi,
            media_path=media_path,
            poster_template_path=poster_path,
            sensor_brightness=sensor_brightness,
            enable_sensor_match=enable_sensor_match,
        )
    except RuntimeError as exc:
        detail = str(exc)
        status_code = 503 if "OPENAI_API_KEY" in detail else 502
        raise HTTPException(status_code=status_code, detail=detail) from exc

    configs = config_store.list_configs()
    return templates.TemplateResponse(
        request,
        "analyze.html",
        {
            "request": request,
            "title": "Analysis",
            "configs": configs,
            "configs_payload": [item.to_dict() for item in configs],
            "poster_templates": [path.name for path in sorted(POSTER_TEMPLATE_DIR.glob("*"))],
            "selected_config": config.to_dict(),
            "analysis_result": result,
        },
    )


@app.post("/api/validate-image")
async def validate_image_api(
    config_id: str = Form(...),
    roi_name: str = Form(...),
    image_file: UploadFile = File(...),
) -> JSONResponse:
    try:
        config = config_store.load(config_id)
        roi = config_store.get_roi(config_id, roi_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="config not found") from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="roi not found") from exc

    image_path = save_upload(image_file, UPLOAD_DIR, f"{config_id}_{roi_name}_validate_image")
    try:
        validation = analysis_service.validator.validate_image(read_image(image_path), roi, source_path=image_path)
    except RuntimeError as exc:
        detail = str(exc)
        status_code = 503 if "OPENAI_API_KEY" in detail else 502
        raise HTTPException(status_code=status_code, detail=detail) from exc
    return JSONResponse(
        {
            "is_valid": validation.is_valid,
            "human_clear_ratio": round(validation.visible_ratio, 3),
            "occlusion_level": validation.occlusion_level,
            "summary": validation.summary,
            "reject_reason": validation.reject_reason,
        }
    )


@app.post("/api/validate-video")
async def validate_video_api(
    config_id: str = Form(...),
    roi_name: str = Form(...),
    sensor_brightness: float | None = Form(default=None),
    enable_sensor_match: bool = Form(default=False),
    video_file: UploadFile = File(...),
) -> JSONResponse:
    try:
        config = config_store.load(config_id)
        roi = config_store.get_roi(config_id, roi_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="config not found") from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="roi not found") from exc

    video_path = save_upload(video_file, UPLOAD_DIR, f"{config_id}_{roi_name}_validate_video")
    try:
        validation = analysis_service.validator.validate_video(
            video_path,
            roi,
            sensor_brightness=sensor_brightness,
            enable_sensor_match=enable_sensor_match,
        )
    except RuntimeError as exc:
        detail = str(exc)
        status_code = 503 if "OPENAI_API_KEY" in detail else 502
        raise HTTPException(status_code=status_code, detail=detail) from exc
    return JSONResponse(
        {
            "is_valid": validation.is_valid,
            "human_clear_ratio": round(validation.visible_ratio, 3),
            "occlusion_duration": round(validation.occlusion_duration, 2),
            "brightness_mismatch_duration": round(validation.brightness_mismatch_duration, 2),
            "occlusion_level": validation.occlusion_level,
            "summary": validation.summary,
            "reject_reason": validation.reject_reason,
        }
    )


@app.get("/reports")
def reports_page(
    request: Request,
    store_name: str | None = Query(default=None),
    cctv_id: str | None = Query(default=None),
    roi_name: str | None = Query(default=None),
    decision: str | None = Query(default=None),
    item_type: str | None = Query(default=None),
    cleanliness_mode: str | None = Query(default=None),
    cleanliness_decision: str | None = Query(default=None),
    cleanliness_final_stage: str | None = Query(default=None),
) -> Any:
    filters = {
        "store_name": store_name,
        "cctv_id": cctv_id,
        "roi_name": roi_name,
        "decision": decision,
        "item_type": item_type,
        "cleanliness_mode": cleanliness_mode,
        "cleanliness_decision": cleanliness_decision,
        "cleanliness_final_stage": cleanliness_final_stage,
    }
    pop_filters = {
        "store_name": store_name,
        "cctv_id": cctv_id,
        "roi_name": roi_name,
        "decision": decision,
        "item_type": item_type,
    }
    cleanliness_filters = {
        "store_name": store_name,
        "cctv_id": cctv_id,
        "roi_name": roi_name,
        "mode": cleanliness_mode,
        "decision": cleanliness_decision,
        "final_stage": cleanliness_final_stage,
    }
    records = [with_human_clear_ratio(row) for row in fetch_results(pop_filters)]
    latest_pop = [with_human_clear_ratio(row) for row in fetch_latest_by_roi(roi_name if roi_name else "POP")]
    cleanliness_records = [with_action_workflow_report_details(row) for row in fetch_cleanliness_results(cleanliness_filters)]
    configs = config_store.list_configs()
    cleanliness_store_summary = merge_configured_store_summary(
        fetch_cleanliness_store_summary(cleanliness_filters),
        configs,
        cleanliness_filters,
    )
    options = merge_report_filter_options(
        fetch_filter_options(),
        fetch_cleanliness_filter_options(),
        config_report_filter_options(configs),
    )
    return templates.TemplateResponse(
        request,
        "reports.html",
        {
            "request": request,
            "title": "Reports",
            "records": records,
            "latest_pop": latest_pop,
            "cleanliness_records": cleanliness_records,
            "cleanliness_store_summary": cleanliness_store_summary,
            "options": options,
            "filters": filters,
        },
    )


@app.get("/cleanliness-reports")
def cleanliness_reports_page() -> RedirectResponse:
    return RedirectResponse(url="/reports", status_code=303)
