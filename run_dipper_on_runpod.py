"""Run the pinned DIPPER smoke job on one temporary RunPod A100 Pod.

The provider credential follows the normal RunPod environment contract.  Set
``RUNPOD_API_BASE_URL`` to route through Agent API Guard; if it is absent the
official REST endpoint is used.  A local systemd watchdog independently deletes
the Pod after 150 minutes, while the main process deletes it as soon as the job
finishes or fails.
"""

from __future__ import annotations

import argparse
import atexit
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Callable
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parent
DEFAULT_JOB = ROOT / "dipper_smoke.py"
DEFAULT_INPUT = ROOT / "results" / "dipper-smoke-inputs-v1.json"
DEFAULT_OUTPUT = ROOT / "results" / "dipper-smoke-raw-v1.json"
DEFAULT_LIFECYCLE = ROOT / "results" / "dipper-runpod-lifecycle-v1.json"
OFFICIAL_BASE_URL = "https://rest.runpod.io/v1"
GPU_TYPE_IDS = ("NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB")
IMAGE_NAME = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
CONTAINER_DISK_GB = 120
VOLUME_DISK_GB = 10
MAX_COMPUTE_PER_HOUR_USD = 1.60
WATCHDOG_MINUTES = 150
CONTROL_WAIT_SECONDS = 600
JOB_WAIT_SECONDS = 7_800
RUNNING_STORAGE_USD_PER_GB_MONTH = 0.10
HOURS_PER_MONTH = 730


@dataclass(frozen=True)
class RunPodConfig:
    api_key: str
    base_url: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_runpod_config() -> RunPodConfig:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise RuntimeError("RUNPOD_API_KEY is required")
    base = os.environ.get("RUNPOD_API_BASE_URL", OFFICIAL_BASE_URL).rstrip("/")
    if not base:
        raise RuntimeError("RUNPOD_API_BASE_URL cannot be empty")
    return RunPodConfig(api_key=key, base_url=base)


class RunPodClient:
    def __init__(
        self,
        config: RunPodConfig,
        *,
        opener: Callable[..., object] = urllib.request.urlopen,
    ) -> None:
        self.config = config
        self.opener = opener

    def request(
        self,
        method: str,
        path: str,
        payload: object | None = None,
        *,
        timeout: float = 45,
    ) -> tuple[int, object | None]:
        data = None
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        url = self.config.base_url + "/" + path.lstrip("/")
        request = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with self.opener(request, timeout=timeout) as response:  # type: ignore[attr-defined]
                raw = response.read()
                return response.status, json.loads(raw) if raw else None
        except urllib.error.HTTPError as error:
            raw = error.read().decode("utf-8", "replace")
            try:
                detail = json.loads(raw)
            except json.JSONDecodeError:
                detail = {"bodyBytes": len(raw.encode("utf-8"))}
            raise RuntimeError(
                f"RunPod API {method} {path}: HTTP {error.code}: {detail}"
            ) from error


def _start_command() -> str:
    return (
        "/start.sh >/workspace/runpod-start.log 2>&1 & "
        "printf '%s' \"$DIPPER_JOB_B64\" | base64 -d > /workspace/dipper_smoke.py; "
        "printf '%s' \"$DIPPER_INPUT_B64\" | base64 -d > /workspace/dipper-input.json; "
        "printf '%s' '{\"service\":\"dipper-smoke\",\"version\":1}' "
        "> /workspace/control-ready.json; "
        "python3 -m http.server 8000 --directory /workspace "
        ">/workspace/control-http.log 2>&1 & "
        "( status=0; "
        "python3 -m pip install --no-input --disable-pip-version-check "
        "'transformers==4.40.2' 'accelerate==0.30.1' 'nltk==3.8.1' "
        "'sentencepiece==0.2.0' 'protobuf==4.25.3' 'safetensors==0.4.3' "
        "'hf_transfer==0.1.8' "
        ">/workspace/dipper-install.log 2>&1 && "
        "HF_HUB_ENABLE_HF_TRANSFER=1 python3 /workspace/dipper_smoke.py remote "
        "--input /workspace/dipper-input.json "
        "--output /workspace/dipper-output.json "
        ">/workspace/dipper-job.log 2>&1 || status=$?; "
        "printf '%s' \"$status\" >/workspace/dipper-job.exit ) & "
        "exec sleep infinity"
    )


def build_pod_payload(
    *,
    job_source_b64: str,
    input_b64: str,
    cloud_type: str = "SECURE",
    gpu_type_ids: tuple[str, ...] = GPU_TYPE_IDS,
    pod_name: str = "dipper-synthid-smoke-20260820",
) -> dict[str, object]:
    if cloud_type not in ("SECURE", "COMMUNITY", "ALL"):
        raise ValueError("cloud_type must be SECURE, COMMUNITY, or ALL")
    return {
        "allowedCudaVersions": ["12.8"],
        "cloudType": cloud_type,
        "computeType": "GPU",
        "containerDiskInGb": CONTAINER_DISK_GB,
        "dockerEntrypoint": [],
        "dockerStartCmd": ["bash", "-lc", _start_command()],
        "env": {
            "DIPPER_INPUT_B64": input_b64,
            "DIPPER_JOB_B64": job_source_b64,
        },
        "gpuCount": 1,
        "gpuTypeIds": list(gpu_type_ids),
        "gpuTypePriority": "custom",
        "imageName": IMAGE_NAME,
        "interruptible": False,
        "name": pod_name,
        "ports": ["8000/http"],
        "supportPublicIp": True,
        "volumeInGb": VOLUME_DISK_GB,
        "volumeMountPath": "/workspace",
    }


def cost_upper_bound_usd(
    *,
    compute_per_hour: float,
    elapsed_seconds: float,
    container_disk_gb: int,
    volume_disk_gb: int,
) -> float:
    hourly_storage = (
        (container_disk_gb + volume_disk_gb)
        * RUNNING_STORAGE_USD_PER_GB_MONTH
        / HOURS_PER_MONTH
    )
    return (compute_per_hour + hourly_storage) * elapsed_seconds / 3_600


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _http_get(url: str, *, timeout: float = 20) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": "dipper-smoke/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def _lifecycle_sources() -> list[dict[str, str]]:
    return [
        {
            "title": "RunPod create Pod REST API",
            "url": "https://docs.runpod.io/api-reference/pods/POST/pods",
        },
        {
            "title": "RunPod delete Pod REST API",
            "url": "https://docs.runpod.io/api-reference/pods/DELETE/pods/{podId}",
        },
        {
            "title": "RunPod Pod and storage pricing",
            "url": "https://docs.runpod.io/pods/pricing",
        },
    ]


class DipperPodRun:
    def __init__(
        self,
        *,
        client: RunPodClient,
        job_path: Path,
        input_path: Path,
        output_path: Path,
        lifecycle_path: Path,
        watchdog_minutes: int = WATCHDOG_MINUTES,
        hard_cost_cap_usd: float | None = None,
        cloud_type: str = "SECURE",
        gpu_type_ids: tuple[str, ...] = GPU_TYPE_IDS,
        max_compute_per_hour_usd: float = MAX_COMPUTE_PER_HOUR_USD,
        pod_name: str = "dipper-synthid-smoke-20260820",
    ) -> None:
        self.client = client
        self.job_path = job_path
        self.input_path = input_path
        self.output_path = output_path
        self.lifecycle_path = lifecycle_path
        self.watchdog_minutes = watchdog_minutes
        self.hard_cost_cap_usd = hard_cost_cap_usd
        self.cloud_type = cloud_type
        self.gpu_type_ids = gpu_type_ids
        self.max_compute_per_hour_usd = max_compute_per_hour_usd
        self.pod_name = pod_name
        self.pod_id: str | None = None
        self.watchdog_unit: str | None = None
        self.deleted = False
        self.absent = False
        self.compute_per_hour = 0.0
        self.started_monotonic = time.monotonic()
        self.started_at = utc_now()
        self.error: str | None = None
        self.remote_status: str | None = None

    def delete(self, reason: str) -> None:
        if not self.pod_id or self.deleted:
            return
        print(
            json.dumps(
                {"event": "cleanup_started", "podId": self.pod_id, "reason": reason},
                sort_keys=True,
            ),
            flush=True,
        )
        try:
            status, _ = self.client.request("DELETE", f"/pods/{self.pod_id}")
            self.deleted = status in (200, 202, 204)
            print(
                json.dumps({"event": "delete_response", "httpStatus": status}),
                flush=True,
            )
        except Exception as error:
            print(
                json.dumps({"event": "delete_error", "message": str(error)}),
                flush=True,
            )

    def arm_watchdog(self) -> None:
        assert self.pod_id is not None
        unit = "runpod-delete-" + re.sub(r"[^A-Za-z0-9_-]", "-", self.pod_id)
        command = [
            "systemd-run",
            "--user",
            f"--on-active={self.watchdog_minutes}m",
            f"--unit={unit}",
            f"--setenv=RUNPOD_API_KEY={self.client.config.api_key}",
            f"--setenv=RUNPOD_API_BASE_URL={self.client.config.base_url}",
            sys.executable,
            str(Path(__file__).resolve()),
            "delete",
            "--pod-id",
            self.pod_id,
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError("could not arm RunPod watchdog: " + result.stderr.strip())
        self.watchdog_unit = unit
        print(
            json.dumps(
                {"event": "watchdog_armed", "ttlMinutes": self.watchdog_minutes},
                sort_keys=True,
            ),
            flush=True,
        )

    def cancel_watchdog(self) -> None:
        if not self.watchdog_unit:
            return
        subprocess.run(
            ["systemctl", "--user", "stop", self.watchdog_unit + ".timer"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        subprocess.run(
            ["systemctl", "--user", "reset-failed", self.watchdog_unit + ".service"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def verify_absent(self) -> None:
        if not self.pod_id:
            return
        for _ in range(12):
            try:
                _, value = self.client.request("GET", "/pods")
                pods = value if isinstance(value, list) else []
                self.absent = not any(
                    isinstance(pod, dict) and pod.get("id") == self.pod_id
                    for pod in pods
                )
                if self.absent:
                    return
            except Exception:
                pass
            time.sleep(5)

    def write_lifecycle(self) -> None:
        elapsed = time.monotonic() - self.started_monotonic
        value = {
            "computePerHourUsd": self.compute_per_hour,
            "containerDiskGb": CONTAINER_DISK_GB,
            "createdAt": self.started_at,
            "deleted": self.deleted,
            "elapsedSeconds": elapsed,
            "error": self.error,
            "estimatedCostUpperBoundUsd": cost_upper_bound_usd(
                compute_per_hour=self.compute_per_hour,
                elapsed_seconds=elapsed,
                container_disk_gb=CONTAINER_DISK_GB,
                volume_disk_gb=VOLUME_DISK_GB,
            ),
            "inputSha256": _sha256(self.input_path.read_bytes()),
            "jobSha256": _sha256(self.job_path.read_bytes()),
            "hardCostCapUsd": self.hard_cost_cap_usd,
            "cloudType": self.cloud_type,
            "gpuTypeIds": list(self.gpu_type_ids),
            "maxComputePerHourUsd": self.max_compute_per_hour_usd,
            "podName": self.pod_name,
            "methodology": (
                "Measure wall-clock time from controller start through confirmed Pod "
                "deletion. Multiply the returned RunPod compute hourly rate plus the "
                "documented running container and volume storage rates by elapsed time. "
                "This is a conservative estimate, not a provider billing receipt."
            ),
            "outputSha256": (
                _sha256(self.output_path.read_bytes()) if self.output_path.exists() else None
            ),
            "podAbsentAfterDelete": self.absent,
            "podId": self.pod_id,
            "remoteStatus": self.remote_status,
            "schemaVersion": 1,
            "sources": _lifecycle_sources(),
            "verifiedAt": utc_now(),
            "volumeDiskGb": VOLUME_DISK_GB,
            "watchdogMinutes": self.watchdog_minutes,
        }
        _write_json_atomic(self.lifecycle_path, value)

    def _wait_control(self, base: str) -> None:
        deadline = time.monotonic() + CONTROL_WAIT_SECONDS
        next_report = 0.0
        while time.monotonic() < deadline:
            try:
                status, raw = _http_get(base + "/control-ready.json", timeout=12)
                if status == 200 and json.loads(raw) == {
                    "service": "dipper-smoke",
                    "version": 1,
                }:
                    print(
                        json.dumps({"event": "control_channel_ready"}),
                        flush=True,
                    )
                    return
            except (OSError, ValueError):
                pass
            now = time.monotonic()
            if now >= next_report:
                print(
                    json.dumps(
                        {
                            "elapsedSeconds": round(now - self.started_monotonic),
                            "event": "waiting_for_control_channel",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                next_report = now + 30
            time.sleep(8)
        raise TimeoutError("DIPPER control channel was not ready within 10 minutes")

    def _download_output(self, base: str, *, require_complete: bool) -> dict[str, object]:
        status, raw = _http_get(base + "/dipper-output.json", timeout=90)
        if status != 200:
            raise RuntimeError(f"DIPPER artifact download returned HTTP {status}")
        artifact = json.loads(raw)
        if require_complete and artifact.get("status") != "complete":
            raise RuntimeError("DIPPER remote artifact is not complete")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_bytes(raw)
        self.remote_status = str(artifact.get("status"))
        return artifact

    def _wait_job(self, base: str) -> dict[str, object]:
        deadline = time.monotonic() + JOB_WAIT_SECONDS
        next_report = 0.0
        reported_lines = 0
        exit_code: int | None = None
        while time.monotonic() < deadline:
            if self.hard_cost_cap_usd is not None:
                estimated = cost_upper_bound_usd(
                    compute_per_hour=self.compute_per_hour,
                    elapsed_seconds=time.monotonic() - self.started_monotonic,
                    container_disk_gb=CONTAINER_DISK_GB,
                    volume_disk_gb=VOLUME_DISK_GB,
                )
                if estimated >= self.hard_cost_cap_usd * 0.98:
                    try:
                        self._download_output(base, require_complete=False)
                    except Exception:
                        pass
                    raise RuntimeError("DIPPER controller stopped before hard cost cap")
            try:
                status, raw = _http_get(base + "/dipper-job.exit", timeout=12)
                text = raw.decode("utf-8", "replace").strip()
                if status == 200 and text.isdigit():
                    exit_code = int(text)
                    break
            except OSError:
                pass
            now = time.monotonic()
            if now >= next_report:
                progress: list[object] = []
                try:
                    status, raw = _http_get(base + "/dipper-job.log", timeout=20)
                    if status == 200:
                        lines = raw.decode("utf-8", "replace").splitlines()
                        for line in lines[reported_lines:]:
                            try:
                                event = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if event.get("event") in ("dipper_case", "dipper_chunk"):
                                progress.append(event)
                        reported_lines = len(lines)
                except OSError:
                    pass
                print(
                    json.dumps(
                        {
                            "elapsedSeconds": round(now - self.started_monotonic),
                            "event": "job_progress",
                            "newEvents": progress,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                next_report = now + 30
            time.sleep(10)
        if exit_code is None:
            try:
                self._download_output(base, require_complete=False)
            except Exception:
                pass
            raise TimeoutError("DIPPER job exceeded the 130-minute controller deadline")
        if exit_code != 0:
            try:
                self._download_output(base, require_complete=False)
            except Exception:
                pass
            try:
                _, raw = _http_get(base + "/dipper-job.log", timeout=30)
                tail = raw.decode("utf-8", "replace")[-5_000:]
            except Exception:
                tail = "unavailable"
            raise RuntimeError(f"DIPPER job exited {exit_code}; safe log tail:\n{tail}")
        return self._download_output(base, require_complete=True)

    def run(self) -> None:
        _, existing = self.client.request("GET", "/pods")
        if not isinstance(existing, list):
            raise RuntimeError("RunPod list response was not an array")
        if existing:
            raise RuntimeError("refusing to start while another RunPod Pod exists")
        job_source = self.job_path.read_bytes()
        input_source = self.input_path.read_bytes()
        payload = build_pod_payload(
            job_source_b64=base64.b64encode(job_source).decode("ascii"),
            input_b64=base64.b64encode(input_source).decode("ascii"),
            cloud_type=self.cloud_type,
            gpu_type_ids=self.gpu_type_ids,
            pod_name=self.pod_name,
        )
        status, value = self.client.request("POST", "/pods", payload)
        if status not in (200, 201) or not isinstance(value, dict):
            raise RuntimeError("RunPod create response was invalid")
        self.pod_id = str(value["id"])
        self.compute_per_hour = float(
            value.get("costPerHr") or value.get("adjustedCostPerHr") or 999
        )
        print(
            json.dumps(
                {
                    "computePerHourUsd": self.compute_per_hour,
                    "desiredStatus": value.get("desiredStatus"),
                    "event": "pod_created",
                    "gpu": (value.get("gpu") or {}).get("displayName"),
                    "podId": self.pod_id,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if self.compute_per_hour > self.max_compute_per_hour_usd:
            raise RuntimeError(
                f"RunPod rate ${self.compute_per_hour}/h exceeds "
                f"${self.max_compute_per_hour_usd}/h cap"
            )
        self.arm_watchdog()
        base = f"https://{self.pod_id}-8000.proxy.runpod.net"
        self._wait_control(base)
        artifact = self._wait_job(base)
        if len(artifact.get("cases", [])) != 10:
            raise RuntimeError("DIPPER artifact does not contain all 10 cases")
        print(
            json.dumps(
                {
                    "cases": len(artifact["cases"]),
                    "event": "artifact_downloaded",
                    "output": str(self.output_path),
                    "sha256": _sha256(self.output_path.read_bytes()),
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _run(args: argparse.Namespace) -> int:
    config = resolve_runpod_config()
    operation = DipperPodRun(
        client=RunPodClient(config),
        job_path=args.job,
        input_path=args.input,
        output_path=args.output,
        lifecycle_path=args.lifecycle,
        watchdog_minutes=args.watchdog_minutes,
        hard_cost_cap_usd=args.hard_cost_cap_usd,
        cloud_type=args.cloud_type,
        gpu_type_ids=tuple(args.gpu_types),
        max_compute_per_hour_usd=args.max_compute_per_hour_usd,
        pod_name=args.pod_name,
    )

    def cleanup(reason: str) -> None:
        operation.delete(reason)

    def handle_signal(signum, _frame):
        cleanup(f"signal_{signum}")
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    atexit.register(lambda: cleanup("process_exit"))
    error: BaseException | None = None
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
                    "elapsedSeconds": round(
                        time.monotonic() - operation.started_monotonic, 1
                    ),
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
        if not operation.absent and operation.pod_id and error is None:
            error = RuntimeError("RunPod resource still exists after delete")
    if error is not None:
        raise error
    return 0


def _delete(args: argparse.Namespace) -> int:
    client = RunPodClient(resolve_runpod_config())
    status, _ = client.request("DELETE", f"/pods/{args.pod_id}")
    return 0 if status in (200, 202, 204) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--job", type=Path, default=DEFAULT_JOB)
    run.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--lifecycle", type=Path, default=DEFAULT_LIFECYCLE)
    run.add_argument("--watchdog-minutes", type=int, default=WATCHDOG_MINUTES)
    run.add_argument("--hard-cost-cap-usd", type=float)
    run.add_argument(
        "--cloud-type", choices=("SECURE", "COMMUNITY", "ALL"), default="SECURE"
    )
    run.add_argument("--gpu-types", nargs="+", default=list(GPU_TYPE_IDS))
    run.add_argument(
        "--max-compute-per-hour-usd", type=float, default=MAX_COMPUTE_PER_HOUR_USD
    )
    run.add_argument("--pod-name", default="dipper-synthid-smoke-20260820")
    run.set_defaults(handler=_run)
    delete = commands.add_parser("delete")
    delete.add_argument("--pod-id", required=True)
    delete.set_defaults(handler=_delete)
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
