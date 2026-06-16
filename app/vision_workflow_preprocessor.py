from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from app.action_cleanliness import (
    ActionCleanlinessService,
    MEAL_STATUS_IDLE,
    TableOccupancySample,
    format_datetime_value,
    parse_datetime_value,
)
from app.analysis import crop_roi as crop_image_to_roi, open_video_capture, read_image
from app.person_masking import PersonMaskService
from app.schemas import ROI


def crop_roi(image: np.ndarray, roi: ROI | None) -> np.ndarray:
    if roi is None:
        return image
    return crop_image_to_roi(image, roi)


def build_workflow_frame_from_image(
    *,
    captured_at: str | datetime,
    image_path: str | Path | None = None,
    image: np.ndarray | None = None,
    table_id: str | None = None,
    interaction_roi: ROI | None = None,
    payload: dict[str, Any] | None = None,
    person_mask_service: PersonMaskService | None = None,
) -> dict[str, Any]:
    source_image = load_workflow_image(image_path=image_path, image=image)
    occupancy_image = crop_roi(source_image, interaction_roi)
    occupancy_service = person_mask_service or PersonMaskService()
    _, _, detections = occupancy_service.apply_black_mask(occupancy_image)

    frame_payload = dict(payload or {})
    if table_id is not None:
        frame_payload.setdefault("table_id", table_id)

    reason_codes: list[str] = []
    if interaction_roi is None:
        reason_codes.append("ROI_NOT_PROVIDED")

    person_count = len(detections)
    return {
        "captured_at": normalize_captured_at(captured_at),
        "person_present": person_count > 0,
        "person_count": person_count,
        "payload": frame_payload,
        "occupancy_source": "person_masking",
        "occupancy_reason_codes": reason_codes,
    }


def build_workflow_frames_from_images(
    *,
    captured_at_values: Sequence[str | datetime] | None = None,
    image_paths: Sequence[str | Path] | None = None,
    images: Sequence[np.ndarray] | None = None,
    start_time: str | datetime | None = None,
    interval_seconds: float | None = None,
    table_id: str | None = None,
    interaction_roi: ROI | None = None,
    payloads: Sequence[dict[str, Any] | None] | None = None,
    person_mask_service: PersonMaskService | None = None,
) -> list[dict[str, Any]]:
    image_items = list(images or [])
    image_path_items = list(image_paths or [])
    if bool(image_items) == bool(image_path_items):
        raise ValueError("provide exactly one of images or image_paths")

    item_count = len(image_items) if image_items else len(image_path_items)
    frame_timestamps = resolve_captured_at_values(
        item_count=item_count,
        captured_at_values=captured_at_values,
        start_time=start_time,
        interval_seconds=interval_seconds,
    )
    frame_payloads = list(payloads or [])
    if frame_payloads and len(frame_payloads) != item_count:
        raise ValueError("payloads length must match the number of images")

    occupancy_service = person_mask_service or PersonMaskService()
    frames: list[dict[str, Any]] = []
    for index in range(item_count):
        frames.append(
            build_workflow_frame_from_image(
                captured_at=frame_timestamps[index],
                image=image_items[index] if image_items else None,
                image_path=image_path_items[index] if image_path_items else None,
                table_id=table_id,
                interaction_roi=interaction_roi,
                payload=frame_payloads[index] if frame_payloads else None,
                person_mask_service=occupancy_service,
            )
        )
    return frames


def sample_video_workflow_frames(
    *,
    video_path: str | Path,
    interval_seconds: float = 30.0,
    max_frames: int = 10,
    frame_extractor: Any | None = None,
) -> list[dict[str, Any]]:
    if max_frames <= 0:
        return []
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than 0")

    extractor = frame_extractor or extract_video_workflow_samples
    samples = extractor(
        Path(video_path),
        interval_seconds=interval_seconds,
        max_frames=max_frames,
    )
    return [
        normalize_video_frame_sample(
            {
                **sample,
                "sample_index": index,
            },
            interval_seconds=interval_seconds,
        )
        for index, sample in enumerate(samples)
    ]


@dataclass(frozen=True)
class DynamicVideoSamplingConfig:
    idle_interval_seconds: float = 10.0
    occupied_interval_seconds: float = 5.0
    transition_interval_seconds: float = 1.0
    post_check_interval_seconds: float = 2.0
    max_observations: int = 60
    change_threshold: float = 0.12
    stable_threshold: float = 0.04


def sample_dynamic_video_workflow_frames(
    *,
    video_path: str | Path,
    max_frames: int = 10,
    interaction_roi: ROI | None = None,
    person_mask_service: PersonMaskService | None = None,
    sampling_config: DynamicVideoSamplingConfig | None = None,
    frame_reader: Any | None = None,
    duration_seconds: float | None = None,
) -> list[dict[str, Any]]:
    if max_frames <= 0:
        return []

    config = sampling_config or DynamicVideoSamplingConfig()
    occupancy_service = person_mask_service or PersonMaskService()
    capture = None
    temp_copy: Path | None = None
    resolved_duration_seconds = duration_seconds
    reader = frame_reader
    fps_value: float | None = None

    if reader is None:
        capture, temp_copy = open_video_capture(Path(video_path))
        fps_candidate = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        fps_value = fps_candidate if fps_candidate > 0 else None
        if resolved_duration_seconds is None and fps_value is not None:
            frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            if frame_count > 0:
                resolved_duration_seconds = frame_count / fps_value

        def reader(offset_seconds: float) -> dict[str, Any] | None:
            assert capture is not None
            capture.set(cv2.CAP_PROP_POS_MSEC, offset_seconds * 1000.0)
            success, frame = capture.read()
            if not success:
                return None
            frame_index = None
            if fps_value is not None:
                frame_index = int(round(offset_seconds * fps_value))
            return {
                "image": frame,
                "frame_index": frame_index,
                "fps": fps_value,
                "offset_seconds": offset_seconds,
            }

    try:
        offset_seconds = 0.0
        observation_count = 0
        previous_crop: np.ndarray | None = None
        previous_person_present = False
        previous_state = "idle"
        samples: list[dict[str, Any]] = []

        while len(samples) < max_frames and observation_count < config.max_observations:
            if resolved_duration_seconds is not None and offset_seconds > resolved_duration_seconds + 1e-6:
                break

            sample = reader(offset_seconds) if reader is not None else None
            if sample is None:
                break
            normalized_sample = normalize_video_frame_sample(
                {
                    **sample,
                    "sample_index": observation_count,
                },
                interval_seconds=max(config.idle_interval_seconds, 0.1),
            )
            frame_image = normalized_sample["image"]
            crop_image = crop_roi(frame_image, interaction_roi)
            _, _, detections = occupancy_service.apply_black_mask(crop_image)
            person_count = len(detections)
            person_present = person_count > 0
            change_score = compute_frame_change_score(previous_crop, crop_image)
            frame_type, sampling_state, reason_codes = classify_dynamic_frame(
                previous_state=previous_state,
                previous_person_present=previous_person_present,
                person_present=person_present,
                change_score=change_score,
                sampling_config=config,
            )
            normalized_change = max(0.0, min(change_score, 1.0))
            priority = max(
                0.0,
                min(
                    1.0,
                    dynamic_frame_state_priority(frame_type)
                    + normalized_change * 0.35
                    + (0.15 if previous_person_present != person_present else 0.0),
                ),
            )
            samples.append(
                {
                    **normalized_sample,
                    "crop_image": crop_image,
                    "frame_type": frame_type,
                    "sampling_state": sampling_state,
                    "reason_codes": reason_codes,
                    "priority": round(priority, 3),
                    "features": {
                        "change_score": round(change_score, 4),
                        "person_present": person_present,
                        "person_count": person_count,
                    },
                }
            )
            previous_crop = crop_image
            previous_person_present = person_present
            previous_state = sampling_state
            offset_seconds += interval_for_sampling_state(sampling_state, config)
            observation_count += 1

        if len(samples) == 1:
            samples[0]["frame_type"] = "periodic_sample"
            samples[0]["sampling_state"] = "idle"
        return samples
    finally:
        if capture is not None:
            capture.release()
        if temp_copy and temp_copy.exists():
            temp_copy.unlink(missing_ok=True)


def build_workflow_frames_from_video(
    *,
    video_path: str | Path,
    captured_at_start: str | datetime,
    interval_seconds: float = 30.0,
    max_frames: int = 10,
    table_id: str | None = None,
    interaction_roi: ROI | None = None,
    payload_builder: Any | None = None,
    person_mask_service: PersonMaskService | None = None,
    frame_extractor: Any | None = None,
) -> list[dict[str, Any]]:
    samples = sample_video_workflow_frames(
        video_path=video_path,
        interval_seconds=interval_seconds,
        max_frames=max_frames,
        frame_extractor=frame_extractor,
    )
    occupancy_service = person_mask_service or PersonMaskService()
    frames: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        captured_at = captured_at_for_video_frame(
            captured_at_start=captured_at_start,
            frame_index=sample.get("frame_index"),
            fps=sample.get("fps"),
            interval_seconds=interval_seconds,
            sample_index=index,
            offset_seconds=sample.get("offset_seconds"),
        )
        payload = (
            payload_builder(
                sample["image"],
                index=index,
                captured_at=captured_at,
                frame_index=sample.get("frame_index"),
                offset_seconds=sample.get("offset_seconds"),
            )
            if payload_builder is not None
            else None
        )
        frames.append(
            build_workflow_frame_from_image(
                captured_at=captured_at,
                image=sample["image"],
                table_id=table_id,
                interaction_roi=interaction_roi,
                payload=payload,
                person_mask_service=occupancy_service,
            )
        )
    return frames


def captured_at_for_video_frame(
    *,
    captured_at_start: str | datetime,
    interval_seconds: float,
    sample_index: int,
    frame_index: int | None = None,
    fps: float | None = None,
    offset_seconds: float | None = None,
) -> str:
    start_datetime = parse_datetime_value(captured_at_start, "captured_at_start")
    resolved_offset_seconds = resolve_video_frame_offset_seconds(
        frame_index=frame_index,
        fps=fps,
        interval_seconds=interval_seconds,
        sample_index=sample_index,
        offset_seconds=offset_seconds,
    )
    return (start_datetime + timedelta(seconds=resolved_offset_seconds)).isoformat(timespec="seconds")


def build_meal_occupancy_sequence(
    frames: Sequence[dict[str, Any]],
    *,
    table_id: str = "table",
    occupancy_candidate_seconds: int = 60,
    meal_min_duration_seconds: int | None = None,
    meal_absence_seconds: int = 90,
) -> dict[str, Any]:
    sorted_frames = sorted(frames, key=lambda item: frame_timestamp(item))
    observations = [
        TableOccupancySample(
            timestamp=frame_timestamp(frame),
            customer_present=bool(frame.get("person_present", False)),
        )
        for frame in sorted_frames
    ]
    occupancy_candidate_started_at = first_occupancy_candidate_started_at(
        observations,
        occupancy_candidate_seconds=occupancy_candidate_seconds,
    )
    meal_service = ActionCleanlinessService(
        customer_in_use_seconds=occupancy_candidate_seconds,
        meal_min_duration_seconds=meal_min_duration_seconds,
        meal_end_seconds=meal_absence_seconds,
    )
    meal_session = meal_service.evaluate_meal_session(table_id, observations)
    return {
        "table_id": table_id,
        "occupancy_candidate_detected": occupancy_candidate_started_at is not None,
        "occupancy_candidate_started_at": format_datetime_value(occupancy_candidate_started_at),
        "meal_confirmed": meal_session is not None,
        "meal_status": meal_session.status if meal_session is not None else MEAL_STATUS_IDLE,
        "meal_started_at": format_datetime_value(meal_session.meal_started_at) if meal_session is not None else None,
        "meal_ended_at": format_datetime_value(meal_session.meal_ended_at) if meal_session is not None else None,
        "frame_count": len(sorted_frames),
    }


def load_workflow_image(
    *,
    image_path: str | Path | None,
    image: np.ndarray | None,
) -> np.ndarray:
    if image is not None:
        return image.copy()
    if image_path is None:
        raise ValueError("image or image_path is required")
    return read_image(Path(image_path))


def resolve_captured_at_values(
    *,
    item_count: int,
    captured_at_values: Sequence[str | datetime] | None,
    start_time: str | datetime | None,
    interval_seconds: float | None,
) -> list[str]:
    if captured_at_values is not None:
        if len(captured_at_values) != item_count:
            raise ValueError("captured_at_values length must match the number of images")
        return [normalize_captured_at(value) for value in captured_at_values]

    if start_time is None or interval_seconds is None:
        raise ValueError("captured_at_values or start_time with interval_seconds is required")

    start_datetime = parse_datetime_value(start_time, "start_time")
    return [
        normalize_captured_at(start_datetime + timedelta(seconds=index * interval_seconds))
        for index in range(item_count)
    ]


def normalize_captured_at(value: str | datetime) -> str:
    return parse_datetime_value(value, "captured_at").isoformat(timespec="seconds")


def frame_timestamp(frame: dict[str, Any]) -> datetime:
    return parse_datetime_value(frame["captured_at"], "captured_at")


def extract_video_workflow_samples(
    video_path: Path,
    *,
    interval_seconds: float,
    max_frames: int,
) -> list[dict[str, Any]]:
    capture, temp_copy = open_video_capture(video_path)
    fps_value = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    samples: list[dict[str, Any]] = []
    try:
        for sample_index in range(max_frames):
            offset_seconds = sample_index * interval_seconds
            capture.set(cv2.CAP_PROP_POS_MSEC, offset_seconds * 1000.0)
            success, frame = capture.read()
            if not success:
                break
            frame_index = None
            if fps_value > 0:
                frame_index = int(round(offset_seconds * fps_value))
            samples.append(
                {
                    "image": frame,
                    "frame_index": frame_index,
                    "fps": fps_value if fps_value > 0 else None,
                    "offset_seconds": offset_seconds,
                }
            )
    finally:
        capture.release()
        if temp_copy and temp_copy.exists():
            temp_copy.unlink(missing_ok=True)
    return samples


def normalize_video_frame_sample(
    sample: Any,
    *,
    interval_seconds: float,
) -> dict[str, Any]:
    if not isinstance(sample, dict):
        raise ValueError("video frame sample must be a dict")
    image = sample.get("image")
    if not isinstance(image, np.ndarray):
        raise ValueError("video frame sample must include an image ndarray")
    normalized_sample = dict(sample)
    normalized_sample.setdefault("offset_seconds", None)
    normalized_sample["offset_seconds"] = resolve_video_frame_offset_seconds(
        frame_index=normalized_sample.get("frame_index"),
        fps=normalized_sample.get("fps"),
        interval_seconds=interval_seconds,
        sample_index=int(normalized_sample.get("sample_index", 0)),
        offset_seconds=normalized_sample.get("offset_seconds"),
    )
    return normalized_sample


def resolve_video_frame_offset_seconds(
    *,
    frame_index: Any,
    fps: Any,
    interval_seconds: float,
    sample_index: int,
    offset_seconds: Any,
) -> float:
    if offset_seconds is not None:
        return float(offset_seconds)
    if frame_index is not None and fps is not None and float(fps) > 0:
        return float(frame_index) / float(fps)
    return float(sample_index * interval_seconds)


def compute_frame_change_score(previous_crop: np.ndarray | None, current_crop: np.ndarray) -> float:
    if previous_crop is None:
        return 0.0

    target_size = (96, 96)
    previous_gray = cv2.cvtColor(previous_crop, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_crop, cv2.COLOR_BGR2GRAY)
    previous_resized = cv2.resize(previous_gray, target_size, interpolation=cv2.INTER_AREA)
    current_resized = cv2.resize(current_gray, target_size, interpolation=cv2.INTER_AREA)
    previous_blurred = cv2.GaussianBlur(previous_resized, (5, 5), 0)
    current_blurred = cv2.GaussianBlur(current_resized, (5, 5), 0)
    diff = cv2.absdiff(previous_blurred, current_blurred)
    baseline = max(1.0, float(previous_blurred.mean()), float(current_blurred.mean()))
    return float((diff.mean() / baseline) / 2.0)


def classify_dynamic_frame(
    *,
    previous_state: str,
    previous_person_present: bool,
    person_present: bool,
    change_score: float,
    sampling_config: DynamicVideoSamplingConfig,
) -> tuple[str, str, list[str]]:
    reason_codes: list[str] = []

    if person_present:
        if previous_person_present:
            reason_codes.append("person_present")
            return "occupied_representative", "occupied", reason_codes
        if previous_state == "meal_end_candidate" and change_score >= sampling_config.change_threshold:
            reason_codes.extend(["person_reentered", "high_table_change"])
            return "cleaning_candidate", "cleaning_candidate", reason_codes
        reason_codes.append("person_present")
        return "occupied_representative", "occupied", reason_codes

    if previous_state == "cleaning_candidate" and change_score <= sampling_config.stable_threshold:
        reason_codes.append("post_check_stable")
        return "post_check", "post_check", reason_codes

    if previous_person_present:
        reason_codes.append("person_left")
        if change_score >= sampling_config.change_threshold:
            reason_codes.append("high_table_change")
        return "meal_end_candidate", "meal_end_candidate", reason_codes

    if change_score >= sampling_config.change_threshold:
        reason_codes.append("high_table_change")
        return "cleaning_before_candidate", "meal_end_candidate", reason_codes

    reason_codes.append("periodic_sample")
    return "periodic_sample", "idle", reason_codes


def interval_for_sampling_state(state: str, config: DynamicVideoSamplingConfig) -> float:
    if state == "occupied":
        return config.occupied_interval_seconds
    if state in {"meal_end_candidate", "cleaning_candidate"}:
        return config.transition_interval_seconds
    if state == "post_check":
        return config.post_check_interval_seconds
    return config.idle_interval_seconds


def dynamic_frame_state_priority(frame_type: str) -> float:
    return {
        "periodic_sample": 0.15,
        "occupied_representative": 0.35,
        "meal_end_candidate": 0.7,
        "cleaning_before_candidate": 0.78,
        "cleaning_candidate": 0.92,
        "post_check": 0.82,
    }.get(frame_type, 0.15)


def first_occupancy_candidate_started_at(
    observations: Sequence[TableOccupancySample],
    *,
    occupancy_candidate_seconds: int,
) -> datetime | None:
    presence_started_at: datetime | None = None
    for observation in observations:
        if observation.customer_present:
            if presence_started_at is None:
                presence_started_at = observation.timestamp
            present_duration = (observation.timestamp - presence_started_at).total_seconds()
            if present_duration >= occupancy_candidate_seconds:
                return presence_started_at
            continue
        presence_started_at = None
    return None
