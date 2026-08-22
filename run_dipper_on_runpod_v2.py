"""Run the direct curated DIPPER job through RunPod REST API v2."""

from __future__ import annotations

import argparse
import atexit
import base64
from dataclasses import dataclass
import gzip
import json
import os
from pathlib import Path
import shlex
import signal
import time
import urllib.error
import urllib.request

from run_dipper_on_runpod import (
    CONTAINER_DISK_GB,
    DEFAULT_JOB,
    DipperPodRun,
    IMAGE_NAME,
    VOLUME_DISK_GB,
    _sha256,
    _write_json_atomic,
    cost_upper_bound_usd,
    CONTROL_SERVER_START,
    control_server_b64,
    new_control_token,
)


ROOT = Path(__file__).resolve().parent
OFFICIAL_BASE_URL = "https://api.runpod.io/v2"
DEFAULT_INPUT = ROOT / "results" / "curated-dipper-inputs-v1.json"
DEFAULT_OUTPUT = ROOT / "results" / "curated-dipper-raw-v1.json"
DEFAULT_LIFECYCLE = ROOT / "results" / "curated-dipper-runpod-v2-lifecycle-v1.json"
DEFAULT_GPU_TYPES = ("NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB")


@dataclass(frozen=True)
class Config:
    api_key: str
    base_url: str


class V2HTTPError(RuntimeError):
    def __init__(self, status: int, detail: object) -> None:
        super().__init__(f"RunPod v2 HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


def resolve_config() -> Config:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise RuntimeError("RUNPOD_API_KEY is required")
    base = os.environ.get("RUNPOD_API_BASE_URL", OFFICIAL_BASE_URL).rstrip("/")
    if not base:
        raise RuntimeError("RUNPOD_API_BASE_URL cannot be empty")
    return Config(key, base)


class Client:
    def __init__(self, config: Config) -> None:
        self.config = config

    def request(self, method: str, path: str, payload=None, *, timeout: float = 60):
        body = None
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "User-Agent": "dipper-curated/1.0",
        }
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.config.base_url + "/" + path.lstrip("/"),
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
                return response.status, json.loads(raw) if raw else None
        except urllib.error.HTTPError as error:
            raw = error.read().decode("utf-8", "replace")
            try:
                value = json.loads(raw)
                detail = value.get("detail") or value.get("errors") or value.get("title")
            except json.JSONDecodeError:
                detail = {"bodyBytes": len(raw.encode())}
            raise V2HTTPError(error.code, detail) from error


def pods_from_response(value: object) -> list[dict[str, object]]:
    if not isinstance(value, dict) or not isinstance(value.get("pods"), list):
        raise ValueError("RunPod v2 list response must contain wrapped pods")
    return value["pods"]


def build_payload(
    *,
    job_source_b64: str,
    input_b64: str,
    gpu_type: str,
    control_token: str | None = None,
    control_server_source_b64: str | None = None,
) -> dict[str, object]:
    control_token = control_token or new_control_token()
    if len(control_token) < 32:
        raise ValueError("control_token must be at least 32 characters")
    command = (
        "/start.sh >/workspace/runpod-start.log 2>&1 & "
        "printf '%s' \"$DIPPER_INPUT_GZ_B64\" | base64 -d | gzip -d "
        "> /workspace/dipper-input.json; "
        "printf '%s' \"$DIPPER_JOB_GZ_B64\" | base64 -d | gzip -d "
        "> /workspace/dipper_smoke.py; "
        "printf '%s' '{\"service\":\"dipper-smoke\",\"version\":1}' "
        "> /workspace/control-ready.json; "
        + CONTROL_SERVER_START +
        "( status=0; "
        "python3 -m pip install --no-input --disable-pip-version-check "
        "'transformers==4.40.2' 'accelerate==0.30.1' 'nltk==3.8.1' "
        "'sentencepiece==0.2.0' 'protobuf==4.25.3' 'safetensors==0.4.3' "
        "'hf_transfer==0.1.8' >/workspace/dipper-install.log 2>&1 && "
        "HF_HUB_ENABLE_HF_TRANSFER=1 python3 /workspace/dipper_smoke.py remote "
        "--input /workspace/dipper-input.json "
        "--output /workspace/dipper-output.json "
        ">/workspace/dipper-job.log 2>&1 || status=$?; "
        "printf '%s' \"$status\" >/workspace/dipper-job.exit ) & "
        "exec sleep infinity"
    )
    command = "bash -lc " + shlex.quote(command)
    return {
        "args": command,
        "cloud": "SECURE",
        "disk": CONTAINER_DISK_GB,
        "env": {
            "CONTROL_SERVER_B64": control_server_source_b64 or control_server_b64(),
            "CONTROL_TOKEN": control_token,
            "DIPPER_INPUT_GZ_B64": input_b64,
            "DIPPER_JOB_GZ_B64": job_source_b64,
        },
        "gpu": {
            "count": 1,
            "id": gpu_type,
        },
        "image": IMAGE_NAME,
        "name": "dipper-curated-v2-20260821",
        "ports": ["8000/http"],
    }


class V2PodRun(DipperPodRun):
    def delete(self, reason: str) -> None:
        if not self.pod_id or self.deleted:
            return
        print(json.dumps({"event": "cleanup_started", "reason": reason}), flush=True)
        try:
            status, _ = self.client.request("DELETE", f"/pods/{self.pod_id}")
            self.deleted = status in (200, 202, 204)
            print(json.dumps({"event": "delete_response", "httpStatus": status}), flush=True)
        except V2HTTPError as error:
            if error.status == 404:
                self.deleted = True
            else:
                print(json.dumps({"event": "delete_error", "status": error.status}), flush=True)

    def verify_absent(self) -> None:
        if not self.pod_id:
            self.absent = True
            return
        for _ in range(12):
            try:
                _, value = self.client.request("GET", "/pods")
                pods = pods_from_response(value)
                self.absent = not any(pod.get("id") == self.pod_id for pod in pods)
                if self.absent:
                    return
            except Exception:
                pass
            time.sleep(5)

    def write_lifecycle(self) -> None:
        super().write_lifecycle()
        value = json.loads(self.lifecycle_path.read_text(encoding="utf-8"))
        value["apiVersion"] = "v2"
        value["methodology"] = (
            "Measure the v2 Pod lifecycle from controller start through confirmed "
            "termination. Use provider-returned hourly cost and the configured hard "
            "cost cap; a separate watchdog terminates the Pod if the controller dies."
        )
        value["sources"] = [
            {
                "title": "RunPod API v2 create Pod",
                "url": "https://docs.runpod.io/api-reference-v2/pods/create-a-pod",
            },
            {
                "title": "RunPod API v2 terminate Pod",
                "url": "https://docs.runpod.io/api-reference-v2/pods/terminate-a-pod",
            },
        ]
        _write_json_atomic(self.lifecycle_path, value)

    def run(self) -> None:
        _, value = self.client.request("GET", "/pods")
        if pods_from_response(value):
            raise RuntimeError("refusing to start while another RunPod Pod exists")
        job_b64 = base64.b64encode(gzip.compress(self.job_path.read_bytes())).decode()
        input_b64 = base64.b64encode(gzip.compress(self.input_path.read_bytes())).decode()
        created = None
        failures = []
        for gpu_type in self.gpu_type_ids:
            move_to_next_gpu = False
            for attempt in range(3):
                try:
                    status, candidate = self.client.request(
                        "POST",
                        "/pods",
                        build_payload(
                            control_token=self.control_token,
                            job_source_b64=job_b64,
                            input_b64=input_b64,
                            gpu_type=gpu_type,
                        ),
                    )
                except V2HTTPError as error:
                    if error.status in (400, 403):
                        failures.append({"gpu": gpu_type, "status": error.status})
                        move_to_next_gpu = True
                        break
                    if error.status >= 500:
                        # A lost 201 must never result in a duplicate Pod. List first and
                        # adopt an exact-name Pod if the create actually committed.
                        _, listed = self.client.request("GET", "/pods")
                        matches = [
                            pod
                            for pod in pods_from_response(listed)
                            if pod.get("name") == "dipper-curated-v2-20260821"
                        ]
                        if len(matches) == 1:
                            created = matches[0]
                            break
                        if matches:
                            raise RuntimeError("multiple same-name Pods after v2 5xx")
                        if attempt < 2:
                            time.sleep(2**attempt)
                            continue
                    raise
                if status == 201 and isinstance(candidate, dict):
                    created = candidate
                    break
            if created is not None:
                break
            if move_to_next_gpu:
                continue
        if created is None:
            raise RuntimeError(f"no v2 GPU candidate placed: {failures}")
        self.pod_id = str(created["id"])
        self.compute_per_hour = float(created["cost"])
        print(
            json.dumps(
                {
                    "computePerHourUsd": self.compute_per_hour,
                    "event": "pod_created",
                    "gpu": (created.get("gpu") or {}).get("id"),
                    "status": created.get("status"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if self.compute_per_hour > self.max_compute_per_hour_usd:
            raise RuntimeError("v2 Pod hourly rate exceeds cap")
        self.arm_watchdog()
        base = f"https://{self.pod_id}-8000.proxy.runpod.net"
        self._wait_control(base)
        artifact = self._wait_job(base)
        if len(artifact.get("cases", [])) != 10:
            raise RuntimeError("DIPPER artifact does not contain ten cases")
        print(
            json.dumps(
                {
                    "cases": 10,
                    "event": "artifact_downloaded",
                    "sha256": _sha256(self.output_path.read_bytes()),
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _run(args: argparse.Namespace) -> int:
    config = resolve_config()
    operation = V2PodRun(
        client=Client(config),
        job_path=args.job,
        input_path=args.input,
        output_path=args.output,
        lifecycle_path=args.lifecycle,
        watchdog_minutes=args.watchdog_minutes,
        hard_cost_cap_usd=args.hard_cost_cap_usd,
        cloud_type="SECURE",
        gpu_type_ids=tuple(args.gpu_types),
        max_compute_per_hour_usd=args.max_compute_per_hour_usd,
        pod_name="dipper-curated-v2-20260821",
    )

    def cleanup(reason: str) -> None:
        operation.delete(reason)

    def handle_signal(signum, _frame):
        cleanup(f"signal_{signum}")
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    atexit.register(lambda: cleanup("process_exit"))
    error = None
    try:
        operation.run()
    except BaseException as caught:
        error = caught
        operation.error = f"{type(caught).__name__}: {caught}"
    finally:
        operation.delete("job_complete")
        operation.cancel_watchdog()
        operation.verify_absent()
        operation.write_lifecycle()
        print(
            json.dumps(
                {
                    "estimatedCostUpperBoundUsd": cost_upper_bound_usd(
                        compute_per_hour=operation.compute_per_hour,
                        elapsed_seconds=time.monotonic() - operation.started_monotonic,
                        container_disk_gb=CONTAINER_DISK_GB,
                        volume_disk_gb=VOLUME_DISK_GB,
                    ),
                    "event": "cleanup_verified",
                    "podAbsent": operation.absent,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if error is not None:
        raise error
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, default=DEFAULT_JOB)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lifecycle", type=Path, default=DEFAULT_LIFECYCLE)
    parser.add_argument("--gpu-types", nargs="+", default=list(DEFAULT_GPU_TYPES))
    parser.add_argument("--watchdog-minutes", type=int, default=12)
    parser.add_argument("--hard-cost-cap-usd", type=float, default=0.35)
    parser.add_argument("--max-compute-per-hour-usd", type=float, default=1.60)
    args = parser.parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
