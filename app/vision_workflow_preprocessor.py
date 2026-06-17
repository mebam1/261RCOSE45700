from __future__ import annotations

from dataclasses import dataclass, field
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
from app.person_masking import PersonDetection, PersonMaskService
from app.schemas import ROI


def crop_roi(image: np.ndarray, roi: ROI | None) -> np.ndarray:
    if roi is None:
        return image
    return crop_image_to_roi(image, roi)


def roi_to_rect(roi: ROI) -> ImageRect:
    bounds = roi.bounds
    return ImageRect(
        x=int(bounds["x"]),
        y=int(bounds["y"]),
        width=int(bounds["width"]),
        height=int(bounds["height"]),
    )


def expanded_interaction_rect(
    table_rect: ImageRect,
    image_shape: tuple[int, ...],
    *,
    expand_x: float,
    expand_y: float,
) -> ImageRect:
    image_height, image_width = image_shape[:2]
    pad_x = int(round(table_rect.width * expand_x))
    pad_y = int(round(table_rect.height * expand_y))
    left = max(0, table_rect.x - pad_x)
    top = max(0, table_rect.y - pad_y)
    right = min(image_width, table_rect.right + pad_x)
    bottom = min(image_height, table_rect.bottom + pad_y)
    return ImageRect(x=left, y=top, width=max(1, right - left), height=max(1, bottom - top))


def crop_rect(image: np.ndarray, rect: ImageRect) -> np.ndarray:
    return image[rect.y : rect.bottom, rect.x : rect.right]


def detection_to_frame_rect(detection: PersonDetection, origin: ImageRect) -> ImageRect:
    return ImageRect(
        x=origin.x + detection.x,
        y=origin.y + detection.y,
        width=detection.width,
        height=detection.height,
    )


def rect_intersection_area(first: ImageRect, second: ImageRect) -> int:
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.right, second.right)
    bottom = min(first.bottom, second.bottom)
    return max(0, right - left) * max(0, bottom - top)


def point_inside_rect(point: tuple[float, float], rect: ImageRect) -> bool:
    x, y = point
    return rect.x <= x <= rect.right and rect.y <= y <= rect.bottom


def rect_distance(first: ImageRect, second: ImageRect) -> float:
    dx = max(first.x - second.right, second.x - first.right, 0)
    dy = max(first.y - second.bottom, second.y - first.bottom, 0)
    return float((dx * dx + dy * dy) ** 0.5)


def point_to_rect_distance(point: tuple[float, float], rect: ImageRect) -> float:
    x, y = point
    dx = max(rect.x - x, x - rect.right, 0.0)
    dy = max(rect.y - y, y - rect.bottom, 0.0)
    return float((dx * dx + dy * dy) ** 0.5)


def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def person_relevance_score(
    detection_rect: ImageRect,
    *,
    detection_score: float,
    table_rect: ImageRect,
    interaction_rect: ImageRect,
    image_shape: tuple[int, ...],
    temporal_coupling_score: float = 0.0,
) -> dict[str, float]:
    center = (detection_rect.x + detection_rect.width / 2.0, detection_rect.y + detection_rect.height / 2.0)
    bottom_center = (center[0], float(detection_rect.bottom))
    table_diagonal = max(1.0, float((table_rect.width * table_rect.width + table_rect.height * table_rect.height) ** 0.5))
    halo_position_score = 0.0
    if point_inside_rect(center, interaction_rect) or point_inside_rect(bottom_center, interaction_rect):
        halo_position_score = clamp_score(1.0 - point_to_rect_distance(bottom_center, table_rect) / table_diagonal)

    overlap_score = 0.0
    if detection_rect.area > 0:
        overlap_score = rect_intersection_area(detection_rect, table_rect) / detection_rect.area

    proximity_score = clamp_score(1.0 - rect_distance(detection_rect, table_rect) / table_diagonal)
    size_score = clamp_score(detection_rect.height / max(1.0, image_shape[0] * 0.12))
    confidence_score = clamp_score(detection_score)

    score = (
        0.35 * halo_position_score
        + 0.25 * clamp_score(overlap_score)
        + 0.20 * proximity_score
        + 0.10 * size_score
        + 0.05 * confidence_score
        + 0.05 * clamp_score(temporal_coupling_score)
    )
    return {
        "score": round(clamp_score(score), 3),
        "halo_position_score": round(halo_position_score, 3),
        "overlap_score": round(clamp_score(overlap_score), 3),
        "proximity_score": round(proximity_score, 3),
        "size_score": round(size_score, 3),
        "confidence_score": round(confidence_score, 3),
        "temporal_coupling_score": round(clamp_score(temporal_coupling_score), 3),
    }


def estimate_relevant_person_signal(
    *,
    frame_image: np.ndarray,
    table_roi: ROI | None,
    person_mask_service: PersonMaskService,
    sampling_config: DynamicVideoSamplingConfig,
    temporal_coupling_score: float = 0.0,
) -> dict[str, Any]:
    if table_roi is None:
        _, _, detections = person_mask_service.apply_black_mask(frame_image)
        raw_person_count = len(detections)
        return {
            "raw_person_count": raw_person_count,
            "relevant_person_count": raw_person_count,
            "person_present": raw_person_count > 0,
            "best_relevant_person_score": 1.0 if raw_person_count else 0.0,
            "person_relevance_scores": [1.0 for _ in detections],
            "person_relevance_reason": "roi_not_provided" if raw_person_count else "no_person",
        }

    table_rect = roi_to_rect(table_roi)
    interaction_rect = expanded_interaction_rect(
        table_rect,
        frame_image.shape,
        expand_x=sampling_config.interaction_halo_expand_x,
        expand_y=sampling_config.interaction_halo_expand_y,
    )
    _, _, detections = person_mask_service.apply_black_mask(crop_rect(frame_image, interaction_rect))
    scored_detections = [
        person_relevance_score(
            detection_to_frame_rect(detection, interaction_rect),
            detection_score=detection.score,
            table_rect=table_rect,
            interaction_rect=interaction_rect,
            image_shape=frame_image.shape,
            temporal_coupling_score=temporal_coupling_score,
        )
        for detection in detections
    ]
    relevance_scores = [item["score"] for item in scored_detections]
    best_score = max(relevance_scores, default=0.0)
    relevant_count = sum(1 for score in relevance_scores if score >= sampling_config.person_relevant_threshold)

    if relevant_count:
        reason = "person_near_table"
    elif best_score >= sampling_config.person_uncertain_threshold:
        reason = "person_uncertain"
    elif detections:
        reason = "background_person_likely"
    else:
        reason = "no_person"

    return {
        "raw_person_count": len(detections),
        "relevant_person_count": relevant_count,
        "person_present": relevant_count > 0,
        "best_relevant_person_score": round(best_score, 3),
        "person_relevance_scores": relevance_scores,
        "person_relevance_reason": reason,
        "interaction_halo_bounds": {
            "x": interaction_rect.x,
            "y": interaction_rect.y,
            "width": interaction_rect.width,
            "height": interaction_rect.height,
        },
    }


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
    person_change_coupling_threshold: float = 0.08
    person_relevant_threshold: float = 0.6
    person_uncertain_threshold: float = 0.4
    interaction_halo_expand_x: float = 0.75
    interaction_halo_expand_y: float = 1.25


@dataclass(frozen=True)
class ImageRect:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)


@dataclass
class DynamicSamplingEpisodeSummary:
    episode_id: str
    start_index: int
    sample_indices: list[int] = field(default_factory=list)
    occupied_index: int | None = None
    meal_end_index: int | None = None
    cleaning_candidate_indices: list[int] = field(default_factory=list)
    post_check_index: int | None = None

    def selected_indices(self) -> list[int]:
        selected: list[int] = []
        if self.occupied_index is not None:
            selected.append(self.occupied_index)
        if self.meal_end_index is not None and self.meal_end_index not in selected:
            selected.append(self.meal_end_index)
        for index in self.cleaning_candidate_indices[:2]:
            if index not in selected:
                selected.append(index)
        if self.post_check_index is not None and self.post_check_index not in selected:
            selected.append(self.post_check_index)
        return selected


def sample_dynamic_video_workflow_frames(
    *,
    video_path: str | Path,
    max_frames: int = 10,
    observation_budget: int | None = None,
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
        resolved_observation_budget = max_frames if observation_budget is None else max(1, observation_budget)
        previous_crop: np.ndarray | None = None
        previous_person_present = False
        previous_state = "idle"
        samples: list[dict[str, Any]] = []

        while len(samples) < resolved_observation_budget and observation_count < config.max_observations:
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
            change_score = compute_frame_change_score(previous_crop, crop_image)
            person_signal = estimate_relevant_person_signal(
                frame_image=frame_image,
                table_roi=interaction_roi,
                person_mask_service=occupancy_service,
                sampling_config=config,
                temporal_coupling_score=1.0
                if change_score >= config.person_change_coupling_threshold
                else 0.0,
            )
            person_count = int(person_signal["relevant_person_count"])
            person_present = bool(person_signal["person_present"])
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
                        "raw_person_present": int(person_signal["raw_person_count"]) > 0,
                        "raw_person_count": int(person_signal["raw_person_count"]),
                        "relevant_person_count": person_count,
                        "best_relevant_person_score": person_signal["best_relevant_person_score"],
                        "person_relevance_scores": person_signal["person_relevance_scores"],
                        "person_relevance_reason": person_signal["person_relevance_reason"],
                        "interaction_halo_bounds": person_signal.get("interaction_halo_bounds"),
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


def summarize_dynamic_video_samples(
    samples: Sequence[dict[str, Any]],
    *,
    target_count: int | None = None,
) -> dict[str, Any]:
    episodes: list[DynamicSamplingEpisodeSummary] = []
    current_episode: DynamicSamplingEpisodeSummary | None = None
    episode_counter = 0

    def begin_episode(index: int) -> DynamicSamplingEpisodeSummary:
        nonlocal episode_counter
        episode_counter += 1
        episode = DynamicSamplingEpisodeSummary(
            episode_id=f"episode_{episode_counter:02d}",
            start_index=index,
        )
        episodes.append(episode)
        return episode

    for index, sample in enumerate(samples):
        frame_type = str(sample.get("frame_type") or "")
        sampling_state = str(sample.get("sampling_state") or "")
        features = sample.get("features", {})
        person_present = bool(features.get("person_present", False))

        if current_episode is None and (
            person_present
            or frame_type in {"meal_end_candidate", "cleaning_before_candidate", "cleaning_candidate", "post_check"}
            or sampling_state in {"occupied", "meal_end_candidate", "cleaning_candidate", "post_check"}
        ):
            current_episode = begin_episode(index)

        if current_episode is None:
            continue

        current_episode.sample_indices.append(index)
        if current_episode.occupied_index is None and frame_type == "occupied_representative":
            current_episode.occupied_index = index
        if frame_type in {"meal_end_candidate", "cleaning_before_candidate"}:
            current_episode.meal_end_index = choose_higher_priority_index(
                samples,
                current_episode.meal_end_index,
                index,
            )
        if frame_type == "cleaning_candidate":
            current_episode.cleaning_candidate_indices.append(index)
            current_episode.cleaning_candidate_indices = top_sample_indices(
                samples,
                current_episode.cleaning_candidate_indices,
                limit=2,
            )
        if frame_type == "post_check":
            current_episode.post_check_index = choose_higher_priority_index(
                samples,
                current_episode.post_check_index,
                index,
            )
            current_episode = None
            continue

        if (
            not person_present
            and frame_type == "periodic_sample"
            and current_episode.meal_end_index is not None
            and current_episode.post_check_index is None
        ):
            current_episode = None

    sample_count = len(samples)
    resolved_target_count = sample_count if target_count is None else max(1, min(sample_count, target_count))
    selected_indices, selection_reasons_by_index = select_high_recall_sample_indices(
        samples,
        episodes,
        target_count=resolved_target_count,
    )
    selected_index_set = set(selected_indices)
    debug_trace: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        debug_sample = dict(sample)
        debug_sample["selected_for_review"] = index in selected_index_set
        debug_sample["episode_id"] = sample_episode_id(episodes, index)
        debug_sample["selection_reasons"] = sorted(selection_reasons_by_index.get(index, set()))
        debug_trace.append(debug_sample)

    return {
        "debug_trace": debug_trace,
        "selected_samples": [debug_trace[index] for index in selected_indices],
        "events": build_dynamic_sampling_events(debug_trace, episodes),
        "episodes": build_dynamic_sampling_episodes(debug_trace, episodes, selected_index_set),
    }


def sample_episode_id(episodes: Sequence[DynamicSamplingEpisodeSummary], sample_index: int) -> str | None:
    for episode in episodes:
        if sample_index in episode.sample_indices:
            return episode.episode_id
    return None


def choose_higher_priority_index(
    samples: Sequence[dict[str, Any]],
    current_index: int | None,
    candidate_index: int,
) -> int:
    if current_index is None:
        return candidate_index
    current_priority = float(samples[current_index].get("priority", 0.0))
    candidate_priority = float(samples[candidate_index].get("priority", 0.0))
    if candidate_priority > current_priority:
        return candidate_index
    return current_index


def top_sample_indices(
    samples: Sequence[dict[str, Any]],
    indices: Sequence[int],
    *,
    limit: int,
) -> list[int]:
    ranked = sorted(
        dict.fromkeys(indices),
        key=lambda index: (
            float(samples[index].get("priority", 0.0)),
            float(samples[index].get("offset_seconds", 0.0)),
        ),
        reverse=True,
    )
    return sorted(ranked[:limit])


def select_high_recall_sample_indices(
    samples: Sequence[dict[str, Any]],
    episodes: Sequence[DynamicSamplingEpisodeSummary],
    *,
    target_count: int,
) -> tuple[list[int], dict[int, set[str]]]:
    selection_reasons_by_index: dict[int, set[str]] = {}

    def add_index(index: int | None, reason: str) -> None:
        if index is None or index < 0 or index >= len(samples):
            return
        selection_reasons_by_index.setdefault(index, set()).add(reason)

    for episode in episodes:
        add_index(episode.occupied_index, "episode_start")
        if episode.meal_end_index is not None:
            add_index(last_occupied_index_before(samples, episode, episode.meal_end_index), "pre_exit_anchor")
            add_index(episode.meal_end_index, "episode_end")
        for index in episode.cleaning_candidate_indices:
            add_index(index, "post_exit_change")
        add_index(episode.post_check_index, "post_exit_stable")

    transition_indices = [
        index
        for episode in episodes
        for index in [
            episode.meal_end_index,
            *episode.cleaning_candidate_indices,
            episode.post_check_index,
        ]
        if index is not None
    ]
    for index in transition_indices:
        add_index(index - 1, "transition_context")
        add_index(index + 1, "transition_context")

    selected_indices = sorted(selection_reasons_by_index)
    if not samples:
        return selected_indices, selection_reasons_by_index

    episode_count = len(episodes)
    minimum_recall_target = min(
        target_count,
        max(4, (episode_count * 3) if episode_count else min(len(samples), 3)),
    )
    peak_cap = min(
        max(2, episode_count * 2) if episode_count else 3,
        max(0, target_count - len(selected_indices)),
        10,
    )
    coverage_cap = min(
        max(2, episode_count + 1) if episode_count else 2,
        max(0, target_count - len(selected_indices)),
        6,
    )

    added_peak_count = 0
    for index in rank_change_peak_indices(samples):
        if len(selected_indices) >= target_count or added_peak_count >= peak_cap:
            break
        if index in selection_reasons_by_index:
            continue
        add_index(index, "change_peak")
        added_peak_count += 1
        selected_indices = sorted(selection_reasons_by_index)

    added_coverage_count = 0
    coverage_target = min(len(samples), max(target_count, coverage_cap))
    for index in evenly_spaced_indices(len(samples), coverage_target):
        if len(selected_indices) >= target_count or added_coverage_count >= coverage_cap:
            break
        if index in selection_reasons_by_index:
            continue
        add_index(index, "coverage")
        added_coverage_count += 1
        selected_indices = sorted(selection_reasons_by_index)

    if len(selected_indices) < minimum_recall_target:
        for index in rank_priority_indices(samples):
            if len(selected_indices) >= minimum_recall_target:
                break
            if index in selection_reasons_by_index:
                continue
            add_index(index, "priority_fill")
            selected_indices = sorted(selection_reasons_by_index)

    if not selected_indices:
        fallback_indices = top_sample_indices(samples, list(range(len(samples))), limit=min(target_count, len(samples)))
        for index in fallback_indices:
            add_index(index, "fallback")
        selected_indices = sorted(selection_reasons_by_index)

    return selected_indices, selection_reasons_by_index


def last_occupied_index_before(
    samples: Sequence[dict[str, Any]],
    episode: DynamicSamplingEpisodeSummary,
    boundary_index: int,
) -> int | None:
    for index in reversed(episode.sample_indices):
        if index >= boundary_index:
            continue
        features = samples[index].get("features", {})
        if bool(features.get("person_present", False)):
            return index
    return None


def rank_change_peak_indices(samples: Sequence[dict[str, Any]]) -> list[int]:
    ranked_candidates = sorted(
        (
            index
            for index in range(len(samples))
            if is_local_change_peak(samples, index)
            and sample_change_score(samples[index]) >= 0.08
        ),
        key=lambda index: (
            sample_change_score(samples[index]),
            float(samples[index].get("priority", 0.0)),
            float(samples[index].get("offset_seconds", 0.0)),
        ),
        reverse=True,
    )

    selected: list[int] = []
    for index in ranked_candidates:
        if any(abs(index - chosen_index) <= 1 for chosen_index in selected):
            continue
        selected.append(index)
    return selected


def rank_priority_indices(samples: Sequence[dict[str, Any]]) -> list[int]:
    return sorted(
        range(len(samples)),
        key=lambda index: (
            float(samples[index].get("priority", 0.0)),
            sample_change_score(samples[index]),
            float(samples[index].get("offset_seconds", 0.0)),
        ),
        reverse=True,
    )


def evenly_spaced_indices(sample_count: int, target_count: int) -> list[int]:
    if sample_count <= 0 or target_count <= 0:
        return []
    if target_count >= sample_count:
        return list(range(sample_count))
    indices: list[int] = []
    for slot in range(target_count):
        index = int(round(slot * (sample_count - 1) / max(1, target_count - 1)))
        if not indices or index != indices[-1]:
            indices.append(index)
    return indices


def is_local_change_peak(samples: Sequence[dict[str, Any]], index: int) -> bool:
    current = sample_change_score(samples[index])
    previous = sample_change_score(samples[index - 1]) if index > 0 else -1.0
    following = sample_change_score(samples[index + 1]) if index + 1 < len(samples) else -1.0
    return current >= previous and current >= following


def sample_change_score(sample: dict[str, Any]) -> float:
    features = sample.get("features", {})
    return float(features.get("change_score", 0.0))


def build_dynamic_sampling_events(
    samples: Sequence[dict[str, Any]],
    episodes: Sequence[DynamicSamplingEpisodeSummary],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for episode in episodes:
        event_indices = episode.selected_indices()
        for index in event_indices:
            sample = samples[index]
            if sample.get("frame_type") == "occupied_representative" and episode.meal_end_index is None:
                continue
            events.append(
                {
                    "episode_id": episode.episode_id,
                    "frame_type": sample.get("frame_type"),
                    "sampling_state": sample.get("sampling_state"),
                    "timestamp_sec": round(float(sample.get("offset_seconds", 0.0)), 2),
                    "priority": sample.get("priority"),
                    "reason_codes": list(sample.get("reason_codes", [])),
                }
            )
    return events


def build_dynamic_sampling_episodes(
    samples: Sequence[dict[str, Any]],
    episodes: Sequence[DynamicSamplingEpisodeSummary],
    selected_index_set: set[int],
) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for episode in episodes:
        occupied_sample = samples[episode.occupied_index] if episode.occupied_index is not None else None
        meal_end_sample = samples[episode.meal_end_index] if episode.meal_end_index is not None else None
        post_check_sample = samples[episode.post_check_index] if episode.post_check_index is not None else None
        cleaning_samples = [samples[index] for index in episode.cleaning_candidate_indices]
        serialized.append(
            {
                "episode_id": episode.episode_id,
                "start_timestamp_sec": round(float(samples[episode.start_index].get("offset_seconds", 0.0)), 2),
                "sample_count": len(episode.sample_indices),
                "selected_count": sum(1 for index in episode.sample_indices if index in selected_index_set),
                "occupied_at": sample_timestamp_value(occupied_sample),
                "meal_end_at": sample_timestamp_value(meal_end_sample),
                "cleaning_candidate_at": [
                    sample_timestamp_value(sample)
                    for sample in cleaning_samples
                    if sample_timestamp_value(sample) is not None
                ],
                "post_check_at": sample_timestamp_value(post_check_sample),
            }
        )
    return serialized


def sample_timestamp_value(sample: dict[str, Any] | None) -> float | None:
    if sample is None:
        return None
    return round(float(sample.get("offset_seconds", 0.0)), 2)


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
