from collections import deque
import concurrent.futures
import math
import time
from typing import Any, Dict

import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy


class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: int,
        *,
        enable_rtc: bool = False,
        execution_fps: float | None = None,
        rtc_default_delay_steps: int = 4,
        rtc_latency_percentile: float = 0.95,
        rtc_max_guidance_weight: float = 10.0,
        rtc_prefix_attention_schedule: str = "exp",
    ):
        self._policy = policy
        self._action_horizon = action_horizon
        self._enable_rtc = bool(enable_rtc)
        self._execution_fps = execution_fps
        self._rtc_default_delay_steps = max(0, int(rtc_default_delay_steps))
        self._rtc_latency_percentile = float(rtc_latency_percentile)
        self._rtc_max_guidance_weight = float(rtc_max_guidance_weight)
        self._rtc_prefix_attention_schedule = str(rtc_prefix_attention_schedule)
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None
        self._latencies: deque[float] = deque(maxlen=max(1, int(action_horizon)))
        self._executor: concurrent.futures.ThreadPoolExecutor | None = (
            concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="openpi-rtc")
            if self._enable_rtc
            else None
        )
        self._pending_future: concurrent.futures.Future | None = None
        self._pending_started_at: float | None = None
        self._pending_submit_step: int = 0
        self._pending_inference_delay: int = 0

    def needs_new_inference(self) -> bool:
        if not self._enable_rtc:
            return self._last_results is None
        return self._last_results is None or self._pending_ready()

    def last_results(self) -> Dict[str, np.ndarray] | None:
        return self._last_results

    def _compute_inference_delay_steps(self) -> int:
        if not self._latencies or self._execution_fps is None or self._execution_fps <= 0:
            return self._rtc_default_delay_steps
        percentile = min(1.0, max(0.0, self._rtc_latency_percentile))
        latency = float(np.quantile(np.asarray(self._latencies, dtype=np.float32), percentile))
        return max(1, int(math.ceil(latency * float(self._execution_fps))))

    def _current_chunk_length(self) -> int:
        if self._last_results is not None:
            actions = self._last_results.get("actions")
            if isinstance(actions, np.ndarray) and actions.ndim > 0:
                return int(actions.shape[0])
        return self._action_horizon

    def _current_action_index(self) -> int:
        return max(0, min(self._cur_step - 1, self._current_chunk_length() - 1))

    def _should_start_async_refresh(self) -> bool:
        if not self._enable_rtc or self._last_results is None or self._cur_step <= 0:
            return False
        if self._pending_future is not None:
            return False
        inference_delay = self._compute_inference_delay_steps()
        action_index = self._current_action_index()
        remaining_steps = self._action_horizon - self._cur_step
        return remaining_steps <= inference_delay and action_index + inference_delay < self._current_chunk_length()

    def _with_rtc_sample_kwargs(self, sample_kwargs: dict[str, Any] | None) -> tuple[dict[str, Any] | None, int]:
        sample_kwargs = dict(sample_kwargs or {})
        if not self._should_start_async_refresh():
            return sample_kwargs or None, 0

        action_index = self._current_action_index()
        inference_delay = self._compute_inference_delay_steps()
        prefix_attention_horizon = self._current_chunk_length() - inference_delay - action_index
        if prefix_attention_horizon <= 0:
            return sample_kwargs or None, 0

        sample_kwargs["time_base"] = int(action_index)
        sample_kwargs["rtc_config"] = {
            "inference_delay": int(inference_delay),
            "prefix_attention_horizon": int(prefix_attention_horizon),
            "max_guidance_weight": float(self._rtc_max_guidance_weight),
            "prefix_attention_schedule": self._rtc_prefix_attention_schedule,
        }
        return sample_kwargs, int(inference_delay)

    def _pending_ready(self) -> bool:
        if self._pending_future is None or not self._pending_future.done():
            return False
        return self._pending_elapsed_steps() >= self._pending_inference_delay

    def _pending_elapsed_steps(self) -> int:
        return max(0, self._cur_step - self._pending_submit_step)

    def _infer_policy(self, obs: Dict, sample_kwargs: dict[str, Any] | None) -> Dict:
        start_time = time.monotonic()
        try:
            result = self._policy.infer(obs, sample_kwargs=sample_kwargs)
        finally:
            elapsed = time.monotonic() - start_time
            self._latencies.append(elapsed)
        result["client_timing"] = {
            "roundtrip_ms": elapsed * 1000,
        }
        return result

    def _start_async_refresh(self, obs: Dict, sample_kwargs: dict[str, Any] | None) -> None:
        if self._executor is None or self._pending_future is not None:
            return
        request_sample_kwargs, inference_delay = self._with_rtc_sample_kwargs(sample_kwargs)
        if inference_delay <= 0:
            return
        self._pending_started_at = time.monotonic()
        self._pending_submit_step = self._cur_step
        self._pending_inference_delay = inference_delay
        self._pending_future = self._executor.submit(self._infer_policy, obs, request_sample_kwargs)

    def _consume_pending(self, *, block: bool = False) -> bool:
        if self._pending_future is None:
            return False
        if not block and not self._pending_future.done():
            return False

        future = self._pending_future
        self._pending_future = None
        results = future.result()
        executed_since_submit = self._pending_elapsed_steps()
        start_step = max(self._pending_inference_delay, executed_since_submit)
        self._last_results = results
        self._cur_step = min(start_step, max(0, self._current_chunk_length() - 1))
        self._pending_started_at = None
        self._pending_submit_step = 0
        self._pending_inference_delay = 0
        return True

    def _load_new_chunk_sync(self, obs: Dict, sample_kwargs: dict[str, Any] | None) -> None:
        self._last_results = self._infer_policy(obs, sample_kwargs)
        self._cur_step = 0

    @override
    def infer(self, obs: Dict, *, sample_kwargs: dict[str, Any] | None = None) -> Dict:  # noqa: UP006
        if not self._enable_rtc:
            if self._last_results is None:
                self._load_new_chunk_sync(obs, sample_kwargs)
        else:
            if self._last_results is None:
                if not self._consume_pending(block=self._pending_future is not None):
                    self._load_new_chunk_sync(obs, sample_kwargs)
            elif self._pending_ready():
                self._consume_pending()
            else:
                self._start_async_refresh(obs, sample_kwargs)

        def slicer(x):
            if isinstance(x, np.ndarray):
                return x[self._cur_step, ...]
            else:
                return x

        results = tree.map_structure(slicer, self._last_results)
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results

    @override
    def reset(self) -> None:
        if self._pending_future is not None:
            if not self._pending_future.cancel():
                self._pending_future.result()
            self._pending_future = None
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0
        self._pending_started_at = None
        self._pending_submit_step = 0
        self._pending_inference_delay = 0
