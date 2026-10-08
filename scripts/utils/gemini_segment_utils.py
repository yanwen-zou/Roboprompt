from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PHASE_ITEM_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "required": [
        "phase_name",
        "start_frame",
        "end_frame",
    ],
    "properties": {
        "phase_name": {"type": "STRING"},
        "start_frame": {"type": "INTEGER"},
        "end_frame": {"type": "INTEGER"},
    },
}

SEGMENTATION_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "required": [
        "phases",
    ],
    "properties": {
        "phases": {
            "type": "ARRAY",
            "items": PHASE_ITEM_SCHEMA,
        },
    },
}

PHASE_TEMPLATE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "required": ["phases"],
    "properties": {
        "phases": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "required": ["phase_name"],
                "properties": {
                    "phase_name": {"type": "STRING"},
                },
            },
        },
    },
}


def build_prompt(
    task_description: str | None,
    video_meta: dict[str, Any],
    *,
    prompt_prefix: str,
    min_phases: int,
    max_phases: int,
) -> str:
    task_description = task_description or "N/A"
    duration_text = (
        f"{video_meta['duration_sec']:.3f}s" if isinstance(video_meta["duration_sec"], (float, int)) else "unknown"
    )
    lines = [
        prompt_prefix.strip(),
        f"Segment this RoboCasa episode video into {min_phases} to {max_phases} ordered sub-phases.",
        "Each phase should correspond to a visually coherent stage of the robot's behavior.",
        "Use the actual visible behavior in the video rather than guessing hidden intent.",
        f"Task description: {task_description}",
        f"Original frame count: {video_meta['original_frame_count']}",
        f"Original fps: {video_meta['original_fps']:.6f}",
        f"Uploaded frame count: {video_meta['uploaded_frame_count']}",
        f"Uploaded fps: {video_meta['uploaded_fps']:.6f}",
        f"Each uploaded frame corresponds to every {video_meta['frame_stride']} original frames.",
        f"Observed duration: {duration_text}",
        "Return valid JSON matching the schema.",
        "Only output the phases array object required by the schema.",
        "Each phase object must contain exactly these keys: phase_name, start_frame, end_frame.",
        "Do not output summary, description, confidence, timestamps, comments, or markdown.",
        "Constraints:",
        "- phases must be temporally ordered and non-overlapping",
        "- start/end frame must stay within the video duration",
        "- use concise phase names",
        "- start_frame and end_frame must use ORIGINAL video frame indices",
    ]
    return "\n".join(line for line in lines if line)


def maybe_save_debug_response(debug_dir: Path | None, episode_index: int | None, raw_text: str) -> None:
    if debug_dir is None:
        return

    debug_dir = debug_dir.expanduser().resolve()
    debug_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{episode_index:06d}" if episode_index is not None else "unknown"
    path = debug_dir / f"episode_{suffix}_raw_response.txt"
    path.write_text(raw_text, encoding="utf-8")


def maybe_save_debug_response_json(debug_dir: Path | None, episode_index: int | None, response: Any) -> None:
    if debug_dir is None:
        return

    debug_dir = debug_dir.expanduser().resolve()
    debug_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{episode_index:06d}" if episode_index is not None else "unknown"
    path = debug_dir / f"episode_{suffix}_response.json"

    if hasattr(response, "model_dump_json"):
        path.write_text(response.model_dump_json(indent=2), encoding="utf-8")
    else:
        path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")


def maybe_print_raw_response(enabled: bool, episode_index: int | None, raw_text: str, *, label: str) -> None:
    if not enabled:
        return

    suffix = f"{episode_index:06d}" if episode_index is not None else "unknown"
    preview = raw_text[:4000]
    print(f"\n===== Gemini raw output ({label}) episode_{suffix} =====")
    print(preview)
    if len(raw_text) > len(preview):
        print("\n[truncated]")
    print("===== End raw output =====\n")


def parse_json_response_text(text: str) -> dict[str, Any]:
    candidates = build_json_parse_candidates(text)
    errors: list[str] = []

    for idx, candidate in enumerate(candidates, start=1):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(f"candidate#{idx}: {exc}")
            continue

        if not isinstance(parsed, dict):
            errors.append(f"candidate#{idx}: parsed type is {type(parsed).__name__}, expected object")
            continue
        return parsed

    preview = text[:1200]
    raise RuntimeError(
        "Failed to parse Gemini JSON response. "
        f"errors={errors}. Raw response preview:\n{preview}"
    )


def build_json_parse_candidates(text: str) -> list[str]:
    stripped = text.strip()
    candidates: list[str] = []

    if stripped:
        candidates.append(stripped)

    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    for block in fenced_blocks:
        block = block.strip()
        if block:
            candidates.append(block)

    extracted = extract_outer_json_object(stripped)
    if extracted:
        candidates.append(extracted)

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def extract_outer_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        ch = text[idx]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    return None


def extract_response_payload(
    response: Any,
    *,
    debug_dir: Path | None,
    episode_index: int | None,
    print_raw_response: bool,
    allow_candidate_fallback: bool,
) -> tuple[dict[str, Any], str]:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, dict):
        return parsed, json.dumps(parsed, ensure_ascii=False)
    if parsed is not None and hasattr(parsed, "model_dump"):
        parsed_dict = parsed.model_dump()
        return parsed_dict, json.dumps(parsed_dict, ensure_ascii=False)

    text = response.text or ""
    if text.strip():
        try:
            return parse_json_response_text(text), text
        except Exception:
            maybe_save_debug_response(debug_dir, episode_index, text)
            maybe_save_debug_response_json(debug_dir, episode_index, response)
            maybe_print_raw_response(print_raw_response, episode_index, text, label="response.text")
            raise

    parts = getattr(response, "parts", None) or []
    merged_text = "\n".join(
        part_text
        for part in parts
        if (part_text := getattr(part, "text", None))
    )
    if merged_text.strip():
        try:
            return parse_json_response_text(merged_text), merged_text
        except Exception:
            maybe_save_debug_response(debug_dir, episode_index, merged_text)
            maybe_save_debug_response_json(debug_dir, episode_index, response)
            maybe_print_raw_response(print_raw_response, episode_index, merged_text, label="response.parts")
            raise

    if not allow_candidate_fallback:
        raise RuntimeError(
            "Gemini failed to produce a JSON response. "
            f"response_id={getattr(response, 'response_id', None)!r}, "
            f"response_preview={response.model_dump_json(indent=2)[:2000]}"
        )

    candidate_summaries: list[str] = []
    candidate_text_chunks: list[str] = []
    candidates = getattr(response, "candidates", None) or []
    for idx, candidate in enumerate(candidates, start=1):
        finish_reason = getattr(candidate, "finish_reason", None)
        content = getattr(candidate, "content", None)
        role = getattr(content, "role", None) if content else None
        candidate_summaries.append(f"candidate#{idx}(finish_reason={finish_reason}, role={role})")
        if content is not None:
            for part in getattr(content, "parts", None) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    candidate_text_chunks.append(part_text)

    if candidate_text_chunks:
        merged_candidate_text = "\n".join(candidate_text_chunks)
        try:
            return parse_json_response_text(merged_candidate_text), merged_candidate_text
        except Exception:
            maybe_save_debug_response(debug_dir, episode_index, merged_candidate_text)
            maybe_save_debug_response_json(debug_dir, episode_index, response)
            maybe_print_raw_response(print_raw_response, episode_index, merged_candidate_text, label="candidate.parts")
            raise

    maybe_save_debug_response_json(debug_dir, episode_index, response)
    raise RuntimeError(
        "Gemini returned no parsable JSON content. "
        f"prompt_feedback={getattr(response, 'prompt_feedback', None)!r}, "
        f"candidates={candidate_summaries}, "
        f"response_id={getattr(response, 'response_id', None)!r}, "
        f"response_preview={response.model_dump_json(indent=2)[:2000]}"
    )


def normalize_segmentation(
    segmentation: dict[str, Any],
    frame_count: int,
    frame_stride: int = 1,
    phase_name_template: list[str] | None = None,
) -> dict[str, Any]:
    phases = segmentation.get("phases") or []
    normalized_phases: list[dict[str, Any]] = []
    max_frame = max(frame_count - 1, 0)
    sampled_max_frame = max_frame // max(frame_stride, 1)

    for phase in phases:
        start_frame = int(phase.get("start_frame", 0))
        end_frame = int(phase.get("end_frame", start_frame))

        if frame_stride > 1 and end_frame <= sampled_max_frame:
            start_frame *= frame_stride
            end_frame *= frame_stride

        start_frame = min(max(start_frame, 0), max_frame)
        end_frame = min(max(end_frame, start_frame), max_frame)

        normalized_phases.append(
            {
                "phase_name": str(phase.get("phase_name", "")).strip(),
                "start_frame": start_frame,
                "end_frame": end_frame,
            }
        )

    if phase_name_template is not None:
        if len(normalized_phases) != len(phase_name_template):
            raise RuntimeError(
                "Gemini returned a phase count inconsistent with the first episode template. "
                f"expected={len(phase_name_template)} got={len(normalized_phases)} "
                f"template={phase_name_template}"
            )
        for idx, template_name in enumerate(phase_name_template):
            normalized_phases[idx]["phase_name"] = template_name

    segmentation["phases"] = normalized_phases
    for key in list(segmentation.keys()):
        if key != "phases":
            segmentation.pop(key, None)
    return segmentation


def extract_phase_name_template(segmentation: dict[str, Any]) -> list[str]:
    phases = segmentation.get("phases") or []
    template = [str(phase.get("phase_name", "")).strip() for phase in phases]
    template = [name for name in template if name]
    if not template:
        raise RuntimeError("The first episode did not produce any valid phase names to use as a template.")
    return template
