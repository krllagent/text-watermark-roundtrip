"""Run the quality-gated 14B SynthID corpus job on one ephemeral RunPod Pod.

The RunPod credential follows the ordinary provider environment contract and
supports Agent API Guard through ``RUNPOD_API_BASE_URL``.  The controller
refuses pre-existing Pods, checks the returned hourly rate, polls a public
artifact-only control channel, and deletes the Pod in both normal and failure
paths.  A separate user-systemd watchdog deletes it if this process disappears.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import json
from pathlib import Path
import re
import signal
import subprocess
import time

from run_dipper_on_runpod import (
    DipperPodRun,
    RunPodClient,
    _http_get,
    _sha256,
    cost_upper_bound_usd,
    resolve_runpod_config,
    CONTROL_SERVER_START,
    control_server_b64,
    new_control_token,
    watchdog_command,
    watchdog_env_file,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_JOB = ROOT / "gen_quality_synthid_corpus_gpu.py"
DEFAULT_CONFIG = ROOT / "configs" / "quality-synthid-corpus-v1.json"
DEFAULT_OUTPUT = ROOT / "results" / "quality-synthid-corpus-gpu-v1.json"
DEFAULT_LIFECYCLE = ROOT / "results" / "quality-synthid-runpod-lifecycle-v1.json"

GPU_TYPE_IDS = ("NVIDIA A40", "NVIDIA RTX A6000")
IMAGE_NAME = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
CONTAINER_DISK_GB = 120
VOLUME_DISK_GB = 10
MAX_COMPUTE_PER_HOUR_USD = 1.00
HARD_COST_CAP_USD = 2.00
WATCHDOG_MINUTES = 100
CONTROL_WAIT_SECONDS = 600
JOB_WAIT_SECONDS = 4_500
SOFT_CONTROLLER_COST_STOP_USD = 1.85


def _start_command() -> str:
    return (
        "/start.sh >/workspace/runpod-start.log 2>&1 & "
        "printf '%s' \"$CORPUS_JOB_B64\" | base64 -d "
        "> /workspace/gen_quality_synthid_corpus_gpu.py; "
        "printf '%s' \"$CORPUS_CONFIG_B64\" | base64 -d "
        "> /workspace/corpus-config.json; "
        "printf '%s' '{\"service\":\"quality-synthid-corpus\",\"version\":1}' "
        "> /workspace/control-ready.json; "
        + CONTROL_SERVER_START +
        "( status=0; "
        "python3 -m pip install --no-input --disable-pip-version-check "
        "'transformers==5.15.1' 'huggingface_hub==1.28.0' "
        "'safetensors==0.8.0' 'accelerate>=1.10,<2' "
        "'sentencepiece>=0.2,<0.3' "
        ">/workspace/corpus-install.log 2>&1 && "
        "python3 /workspace/gen_quality_synthid_corpus_gpu.py "
        "--config /workspace/corpus-config.json "
        "--output /workspace/quality-synthid-corpus-v1.json "
        "--checkpoint /workspace/quality-synthid-corpus-v1.partial.json "
        ">/workspace/corpus-job.log 2>&1 || status=$?; "
        "printf '%s' \"$status\" >/workspace/corpus-job.exit ) & "
        "exec sleep infinity"
    )


def build_pod_payload(
    *,
    job_source_b64: str,
    config_b64: str,
    control_token: str | None = None,
    control_server_source_b64: str | None = None,
) -> dict[str, object]:
    control_token = control_token or new_control_token()
    if len(control_token) < 32:
        raise ValueError("control_token must be at least 32 characters")
    return {
        "allowedCudaVersions": ["12.8"],
        "cloudType": "SECURE",
        "computeType": "GPU",
        "containerDiskInGb": CONTAINER_DISK_GB,
        "dockerEntrypoint": [],
        "dockerStartCmd": ["bash", "-lc", _start_command()],
        "env": {
            "CONTROL_SERVER_B64": control_server_source_b64 or control_server_b64(),
            "CONTROL_TOKEN": control_token,
            "CORPUS_CONFIG_B64": config_b64,
            "CORPUS_JOB_B64": job_source_b64,
        },
        "gpuCount": 1,
        "gpuTypeIds": list(GPU_TYPE_IDS),
        "gpuTypePriority": "custom",
        "imageName": IMAGE_NAME,
        "interruptible": False,
        "name": "quality-synthid-corpus-14b-20260821",
        "ports": ["8000/http"],
        "supportPublicIp": True,
        "volumeInGb": VOLUME_DISK_GB,
        "volumeMountPath": "/workspace",
    }


class QualityCorpusPodRun(DipperPodRun):
    def arm_watchdog(self) -> None:
        assert self.pod_id is not None
        unit = "runpod-delete-" + re.sub(r"[^A-Za-z0-9_-]", "-", self.pod_id)
        env_file = watchdog_env_file(
            unit,
            api_key=self.client.config.api_key,
            base_url=self.client.config.base_url,
        )
        command = watchdog_command(
            unit=unit,
            minutes=WATCHDOG_MINUTES,
            env_file=env_file,
            script=Path(__file__).resolve(),
            pod_id=self.pod_id,
        )
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError("could not arm RunPod watchdog: " + result.stderr.strip())
        self.watchdog_unit = unit
        print(
            json.dumps(
                {"event": "watchdog_armed", "ttlMinutes": WATCHDOG_MINUTES},
                sort_keys=True,
            ),
            flush=True,
        )

    def _wait_control(self, base: str) -> None:
        deadline = time.monotonic() + CONTROL_WAIT_SECONDS
        next_report = 0.0
        while time.monotonic() < deadline:
            try:
                status, raw = _http_get(base + "/control-ready.json", timeout=12, token=self.control_token)
                if status == 200 and json.loads(raw) == {
                    "service": "quality-synthid-corpus",
                    "version": 1,
                }:
                    print(json.dumps({"event": "control_channel_ready"}), flush=True)
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
        raise TimeoutError("corpus control channel was not ready within 10 minutes")

    def _download_output(self, base: str, *, require_complete: bool) -> dict[str, object]:
        status, raw = _http_get(base + "/quality-synthid-corpus-v1.json", timeout=120, token=self.control_token)
        if status != 200:
            raise RuntimeError(f"corpus artifact download returned HTTP {status}")
        artifact = json.loads(raw)
        if require_complete and artifact.get("status") != "complete":
            raise RuntimeError("remote corpus artifact is not complete")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_bytes(raw)
        self.remote_status = str(artifact.get("status"))
        return artifact

    def _recover_checkpoint(self, base: str) -> None:
        try:
            status, raw = _http_get(
                base + "/quality-synthid-corpus-v1.partial.json",
                timeout=120,
                token=self.control_token,
            )
            if status == 200:
                recovery = self.output_path.with_suffix(".partial.json")
                recovery.parent.mkdir(parents=True, exist_ok=True)
                recovery.write_bytes(raw)
        except Exception:
            pass

    def _current_cost(self) -> float:
        return cost_upper_bound_usd(
            compute_per_hour=self.compute_per_hour,
            elapsed_seconds=time.monotonic() - self.started_monotonic,
            container_disk_gb=CONTAINER_DISK_GB,
            volume_disk_gb=VOLUME_DISK_GB,
        )

    def _wait_job(self, base: str) -> dict[str, object]:
        deadline = time.monotonic() + JOB_WAIT_SECONDS
        next_report = 0.0
        reported_lines = 0
        exit_code: int | None = None
        while time.monotonic() < deadline:
            if self._current_cost() >= SOFT_CONTROLLER_COST_STOP_USD:
                self._recover_checkpoint(base)
                raise RuntimeError("controller stopped job before the $2 hard cost cap")
            try:
                status, raw = _http_get(base + "/corpus-job.exit", timeout=12, token=self.control_token)
                value = raw.decode("utf-8", "replace").strip()
                if status == 200 and value.isdigit():
                    exit_code = int(value)
                    break
            except OSError:
                pass
            now = time.monotonic()
            if now >= next_report:
                progress: list[object] = []
                try:
                    status, raw = _http_get(base + "/corpus-job.log", timeout=20, token=self.control_token)
                    if status == 200:
                        lines = raw.decode("utf-8", "replace").splitlines()
                        for line in lines[reported_lines:]:
                            try:
                                event = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if event.get("event") in ("pair_attempt", "pair_accepted"):
                                progress.append(event)
                        reported_lines = len(lines)
                except OSError:
                    pass
                print(
                    json.dumps(
                        {
                            "elapsedSeconds": round(now - self.started_monotonic),
                            "estimatedCostUpperBoundUsd": round(self._current_cost(), 4),
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
            self._recover_checkpoint(base)
            raise TimeoutError("corpus job exceeded the 75-minute controller deadline")
        if exit_code != 0:
            try:
                self._download_output(base, require_complete=False)
            except Exception:
                self._recover_checkpoint(base)
            try:
                _, raw = _http_get(base + "/corpus-job.log", timeout=30, token=self.control_token)
                tail = raw.decode("utf-8", "replace")[-5_000:]
            except Exception:
                tail = "unavailable"
            raise RuntimeError(f"corpus job exited {exit_code}; safe log tail:\n{tail}")
        return self._download_output(base, require_complete=True)

    def run(self) -> None:
        _, existing = self.client.request("GET", "/pods")
        if not isinstance(existing, list):
            raise RuntimeError("RunPod list response was not an array")
        if existing:
            raise RuntimeError("refusing to start while another RunPod Pod exists")
        payload = build_pod_payload(
            control_token=self.control_token,
            job_source_b64=base64.b64encode(self.job_path.read_bytes()).decode("ascii"),
            config_b64=base64.b64encode(self.input_path.read_bytes()).decode("ascii"),
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
        if self.compute_per_hour > MAX_COMPUTE_PER_HOUR_USD:
            raise RuntimeError(
                f"RunPod rate ${self.compute_per_hour}/h exceeds "
                f"${MAX_COMPUTE_PER_HOUR_USD}/h cap"
            )
        self.arm_watchdog()
        base = f"https://{self.pod_id}-8000.proxy.runpod.net"
        self._wait_control(base)
        artifact = self._wait_job(base)
        documents = artifact.get("documents")
        if not isinstance(documents, list) or len(documents) != 10:
            raise RuntimeError("corpus artifact does not contain all 10 pairs")
        for document in documents:
            for side in ("marked", "unmarked"):
                if document[side].get("qualityIssues"):
                    raise RuntimeError("accepted corpus output has quality issues")
        print(
            json.dumps(
                {
                    "documentPairs": len(documents),
                    "event": "artifact_downloaded",
                    "output": str(self.output_path),
                    "sha256": _sha256(self.output_path.read_bytes()),
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _run(args: argparse.Namespace) -> int:
    operation = QualityCorpusPodRun(
        client=RunPodClient(resolve_runpod_config()),
        job_path=args.job,
        input_path=args.config,
        output_path=args.output,
        lifecycle_path=args.lifecycle,
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
                    "estimatedCostUpperBoundUsd": round(operation._current_cost(), 4),
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
    try:
        status, _ = client.request("DELETE", f"/pods/{args.pod_id}")
    finally:
        env_file = getattr(args, "env_file", None)
        if env_file:
            Path(env_file).unlink(missing_ok=True)
    return 0 if status in (200, 202, 204) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--job", type=Path, default=DEFAULT_JOB)
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--lifecycle", type=Path, default=DEFAULT_LIFECYCLE)
    run.set_defaults(handler=_run)
    delete = commands.add_parser("delete")
    delete.add_argument("--pod-id", required=True)
    delete.add_argument("--env-file", type=Path)
    delete.set_defaults(handler=_delete)
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
