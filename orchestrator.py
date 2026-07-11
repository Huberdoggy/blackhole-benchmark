#!/usr/bin/env python3
"""
SwarmUI render orchestrator for the black hole benchmark asset backlog.

This script stops at the staging handoff boundary defined in AGENTS.md:
it generates/copies candidate renders into ./src/assets/staging/ and does
not mark backlog items complete.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import websocket

SWARM_API_URL = "http://localhost:7801"
SWARM_OUTPUT_DIR = Path("/home/huberdoggy/projects/SwarmUI/Output/raw")
STAGING_DIR = Path("./src/assets/staging/")
BACKLOG_PATH = Path("render_backlog.md")
MODEL_PATH = Path("/home/huberdoggy/shared_os/swarm_models/diffusion_models/z_image_turbo-Q6_K.gguf")
MODEL_NAME = "z_image_turbo-Q6_K.gguf"
VAE_PATH = Path("/home/huberdoggy/shared_os/swarm_models/VAE/Flux/UltraFlux-vae.safetensors")
VAE_NAME = "Flux/UltraFlux-vae"
QWEN_ENCODER_PATH = Path("/home/huberdoggy/shared_os/swarm_models/text_encoders/Qwen3-4B-UD-Q6_K_XL.gguf")
QWEN_MODEL_NAME = "Qwen3-4B-UD-Q6_K_XL.gguf"

CFG_SCALE = 1.0
STEPS = 8
IMAGES_PER_TASK = 1
REQUEST_TIMEOUT_SECONDS = 900
WS_RECV_TIMEOUT_SECONDS = 10
WS_PROGRESS_TIMEOUT_SECONDS = 300
SMOKE_TEST_SEED = 424242
SMOKE_TEST_PROMPT = (
    "Centered relativistic black hole singularity, luminous rotational photon halo, smooth gravitational lensing arcs, "
    "dense pin-sharp starfield, blue-white hot rim with amber plasma undertones, scientific cinematic realism, crisp "
    "high-contrast 1024 square benchmark asset."
)


PROMPT_BY_COMPONENT = {
    "Deep Space Galactic Canvas": (
        "Vast deep-space starfield behind a relativistic black hole benchmark, wide cosmic canvas "
        "with dense pin-sharp stars, faint nebula dust lanes, subtle gravitational shimmer near "
        "center, cinematic high-contrast astrophotography, crisp PNG texture detail."
    ),
    "Life-like Slider Control Panel": (
        "Futuristic diagnostics control panel for a black hole raytracing benchmark, vivid "
        "glass-and-metal interface surface, illuminated sliders, gauges and telemetry traces, "
        "clean modular layout, sharp sci-fi workstation lighting, polished high-fidelity UI texture."
    ),
    "Photorealistic Relativistic Accretion Disk": (
        "Photoreal accretion disk around a black hole, face-on to slight oblique angle, "
        "incandescent plasma ring with blue-hot approaching side and deep red retreating side, "
        "turbulent thermal emission, dark central horizon, cinematic scientific realism, crisp "
        "high-energy detail."
    ),
    "Event Horizon / Gravitational Singularity Sphere": (
        "High-fidelity black hole singularity and event horizon, centered dark sphere with warped "
        "photon ring, strong gravitational lensing arcs, compressed starlight around the rim, "
        "dramatic astrophysical lighting, polished benchmark hero asset, razor-sharp contrast."
    ),
}


SIZE_BY_COMPONENT = {
    "Deep Space Galactic Canvas": (512, 512),
    "Life-like Slider Control Panel": (512, 384),
    "Photorealistic Relativistic Accretion Disk": (512, 512),
    "Event Horizon / Gravitational Singularity Sphere": (512, 512),
}


NEGATIVE_PROMPT = "blurry, low resolution, cartoon, anime, flat colors, compression artifacts"


@dataclass(frozen=True)
class RenderTask:
    task_id: str
    component: str
    filename: str
    width: int
    height: int
    prompt: str


class SwarmAPIError(RuntimeError):
    pass


def parse_backlog(path: Path) -> list[RenderTask]:
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n(?=### \[ \] Task ID:)", text)
    tasks: list[RenderTask] = []
    for block in blocks:
        if "### [ ] Task ID:" not in block:
            continue
        task_id = _extract(r"Task ID:\s*`([^`]+)`", block, "task id")
        component = _extract(r"\*\*Component Target:\*\*\s*(.+)", block, "component target")
        filename = _extract(r"\*\*Destination Filename:\*\*\s*`([^`]+)`", block, "destination filename")
        if component not in SIZE_BY_COMPONENT:
            raise ValueError(f"No render sizing rule defined for component: {component}")
        if component not in PROMPT_BY_COMPONENT:
            raise ValueError(f"No prompt rule defined for component: {component}")
        width, height = SIZE_BY_COMPONENT[component]
        tasks.append(
            RenderTask(
                task_id=task_id,
                component=component,
                filename=filename,
                width=width,
                height=height,
                prompt=PROMPT_BY_COMPONENT[component],
            )
        )
    return tasks


def _extract(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Could not parse {label} from backlog block:\n{text}")
    return match.group(1).strip()


class SwarmClient:
    def __init__(self, api_url: str, auth_token: str | None = None) -> None:
        self.api_url = api_url.rstrip("/")
        self.session_id: str | None = None
        self.auth_token = auth_token

    def get_session(self) -> str:
        data = self._post("/API/GetNewSession", {})
        session_id = data.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise SwarmAPIError(f"GetNewSession returned no session_id: {data}")
        self.session_id = session_id
        return session_id

    def generate(self, payload: dict[str, Any]) -> list[str]:
        return self.generate_ws(payload)

    def generate_http(self, payload: dict[str, Any]) -> list[str]:
        if not self.session_id:
            self.get_session()
        data = self._post("/API/GenerateText2Image", payload | {"session_id": self.session_id})
        if data.get("error_id") == "invalid_session_id":
            self.get_session()
            data = self._post("/API/GenerateText2Image", payload | {"session_id": self.session_id})
        if "error" in data:
            raise SwarmAPIError(str(data["error"]))
        images = data.get("images")
        if not isinstance(images, list) or not images:
            raise SwarmAPIError(f"GenerateText2Image returned no images: {data}")
        return [str(image) for image in images]

    def generate_ws(self, payload: dict[str, Any]) -> list[str]:
        if not self.session_id:
            self.get_session()
        data = self._generate_ws_once(payload | {"session_id": self.session_id})
        if data == ["__INVALID_SESSION__"]:
            self.get_session()
            data = self._generate_ws_once(payload | {"session_id": self.session_id})
        return data

    def _generate_ws_once(self, payload: dict[str, Any]) -> list[str]:
        socket = self._open_generation_socket()
        try:
            return self._collect_ws_images(socket, payload)
        finally:
            socket.close()

    def _open_generation_socket(self) -> websocket.WebSocket:
        ws_url = self._ws_url("/API/GenerateText2ImageWS")
        headers = []
        if self.auth_token:
            headers.append(f"Cookie: swarm_token={self.auth_token}")
        try:
            return websocket.create_connection(
                ws_url,
                timeout=WS_RECV_TIMEOUT_SECONDS,
                header=headers,
                http_proxy_host=None,
                http_proxy_port=None,
            )
        except Exception as exc:
            raise SwarmAPIError(f"Could not open SwarmUI websocket at {ws_url}: {exc}") from exc

    def _collect_ws_images(self, socket: websocket.WebSocket, payload: dict[str, Any]) -> list[str]:
        images: list[str] = []
        last_progress = time.monotonic()
        last_event: dict[str, Any] | None = None
        try:
            socket.send(json.dumps(payload))
            while True:
                if time.monotonic() - last_progress > WS_PROGRESS_TIMEOUT_SECONDS:
                    raise SwarmAPIError(f"GenerateText2ImageWS stalled waiting for progress. Last event: {last_event}")
                try:
                    raw = socket.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                if not raw:
                    break
                event = json.loads(raw)
                last_event = event
                image_ref = self._handle_ws_event(event)
                if image_ref == "__KEEP_ALIVE__":
                    continue
                last_progress = time.monotonic()
                if image_ref == "__INVALID_SESSION__":
                    return ["__INVALID_SESSION__"]
                if image_ref == "__CLOSE__":
                    break
                if image_ref:
                    images.append(image_ref)
                    return images
        except SwarmAPIError:
            raise
        except Exception as exc:
            raise SwarmAPIError(f"SwarmUI websocket generation failed: {exc}") from exc
        if not images:
            raise SwarmAPIError("GenerateText2ImageWS closed without returning an image.")
        return images

    @staticmethod
    def _handle_ws_event(event: dict[str, Any]) -> str | None:
        if event.get("keep_alive") is True:
            return "__KEEP_ALIVE__"
        if event.get("error_id") == "invalid_session_id":
            return "__INVALID_SESSION__"
        if "error" in event:
            raise SwarmAPIError(str(event["error"]))
        if "image" in event:
            image = event["image"]
            image_ref = image.get("image") if isinstance(image, dict) else image
            return str(image_ref) if image_ref else None
        if event.get("socket_intention") == "close":
            return "__CLOSE__"
        return None

    def download_or_copy(self, image_ref: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if image_ref.startswith("data:"):
            self._write_data_url(image_ref, destination)
            return
        if image_ref.startswith("http://") or image_ref.startswith("https://"):
            self._download(image_ref, destination)
            return
        source = resolve_local_output(image_ref)
        if source.exists():
            shutil.copy2(source, destination)
            return
        url = f"{self.api_url}/{image_ref.lstrip('/')}"
        self._download(url, destination)

    def _post(self, route: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.api_url}{route}"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise SwarmAPIError(f"Could not reach SwarmUI API at {url}: {exc}") from exc
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SwarmAPIError(f"SwarmUI returned non-JSON from {url}: {raw[:500]}") from exc
        if not isinstance(parsed, dict):
            raise SwarmAPIError(f"SwarmUI returned unexpected JSON from {url}: {parsed}")
        return parsed

    def _download(self, url: str, destination: Path) -> None:
        request = urllib.request.Request(url, headers=self._headers(include_content_type=False))
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                destination.write_bytes(response.read())
        except urllib.error.URLError as exc:
            raise SwarmAPIError(f"Could not download generated image {url}: {exc}") from exc

    def _headers(self, include_content_type: bool = True) -> dict[str, str]:
        headers = {"User-Agent": "physics-sandbox-orchestrator/1.0"}
        if include_content_type:
            headers["Content-Type"] = "application/json"
        if self.auth_token:
            headers["Cookie"] = f"swarm_token={self.auth_token}"
        return headers

    def _ws_url(self, route: str) -> str:
        parsed = urllib.parse.urlparse(self.api_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        netloc = parsed.netloc
        base_path = parsed.path.rstrip("/")
        return urllib.parse.urlunparse((scheme, netloc, f"{base_path}{route}", "", "", ""))

    @staticmethod
    def _write_data_url(data_url: str, destination: Path) -> None:
        _, encoded = data_url.split(",", 1)
        destination.write_bytes(base64.b64decode(encoded))


def resolve_local_output(image_ref: str) -> Path:
    path = urllib.parse.unquote(image_ref)
    if path.startswith("View/"):
        return Path("/home/huberdoggy/projects/SwarmUI/Output") / path.removeprefix("View/")
    if path.startswith("/"):
        return Path(path)
    return Path("/home/huberdoggy/projects/SwarmUI") / path


def build_payload(task: RenderTask, seed: int) -> dict[str, Any]:
    return {
        "images": IMAGES_PER_TASK,
        "prompt": task.prompt,
        "negativeprompt": NEGATIVE_PROMPT,
        "model": MODEL_NAME,
        "width": task.width,
        "height": task.height,
        "cfgscale": CFG_SCALE,
        "steps": STEPS,
        "seed": seed,
        "extra_metadata": json.dumps(
            {
                "asset_task_id": task.task_id,
                "asset_handoff_key": f"{task.component}::{task.filename}",
                "asset_component": task.component,
                "asset_destination": task.filename,
                "handoff": "physics_sandbox/src/assets/staging",
            }
        ),
    }


def build_swarm_smoke_payload(explicit_models: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "images": 1,
        "prompt": SMOKE_TEST_PROMPT,
        "negativeprompt": NEGATIVE_PROMPT,
        "model": MODEL_NAME,
        "width": 1024,
        "height": 1024,
        "cfgscale": CFG_SCALE,
        "steps": STEPS,
        "seed": SMOKE_TEST_SEED,
        "extra_metadata": json.dumps(
            {
                "asset_task_id": "BH-SWARM-SMOKE-1024",
                "asset_component": "SwarmUI 1024 nomenclature smoke test",
                "handoff": "physics_sandbox/src/assets/staging",
                "explicit_models": explicit_models,
            }
        ),
    }
    if explicit_models:
        payload.update(
            {
                "vae": VAE_NAME,
                "qwenmodel": QWEN_MODEL_NAME,
            }
        )
    return payload


def verify_environment() -> None:
    if not BACKLOG_PATH.exists():
        raise FileNotFoundError(f"Backlog not found: {BACKLOG_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")


def verify_smoke_environment() -> None:
    verify_environment()
    if not VAE_PATH.exists():
        raise FileNotFoundError(f"VAE file not found: {VAE_PATH}")
    if not QWEN_ENCODER_PATH.exists():
        raise FileNotFoundError(f"Qwen text encoder file not found: {QWEN_ENCODER_PATH}")


def run(dry_run: bool, api_url: str) -> int:
    verify_environment()
    tasks = parse_backlog(BACKLOG_PATH)
    if not tasks:
        print("No pending render tasks found.", flush=True)
        return 0

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Swarm API URL: {api_url}", flush=True)
    print(f"Model: {MODEL_NAME} ({MODEL_PATH})", flush=True)
    print(f"Staging directory: {STAGING_DIR.resolve()}", flush=True)
    print(f"Pending tasks: {len(tasks)}", flush=True)

    client = None if dry_run else SwarmClient(api_url, os.environ.get("SWARM_AUTH_TOKEN"))
    manifest: list[dict[str, Any]] = []

    for index, task in enumerate(tasks, start=1):
        seed = random.randint(1, 2_147_483_647)
        payload = build_payload(task, seed)
        destination = STAGING_DIR / task.filename
        manifest.append(
            {
                "task_id": task.task_id,
                "handoff_key": f"{task.component}::{task.filename}",
                "component": task.component,
                "filename": task.filename,
                "width": task.width,
                "height": task.height,
                "seed": seed,
                "payload": payload,
            }
        )
        print(f"[{index}/{len(tasks)}] {task.component}: {task.width}x{task.height} -> {destination}", flush=True)
        if dry_run:
            continue
        assert client is not None
        images = client.generate(payload)
        client.download_or_copy(images[0], destination)
        print(f"    staged {destination}", flush=True)
        time.sleep(0.25)

    manifest_path = STAGING_DIR / "render_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote handoff manifest: {manifest_path}", flush=True)
    if dry_run:
        print("Dry run only; no SwarmUI generation requests were sent.", flush=True)
    else:
        print("Generation complete. Pause for manual review before updating render_backlog.md.", flush=True)
    return 0


def run_swarm_smoke_test(dry_run: bool, api_url: str) -> int:
    verify_smoke_environment()
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    print("Running 1024x1024 SwarmUI nomenclature smoke test.", flush=True)
    print(f"VAE parameter: vae={VAE_NAME} ({VAE_PATH})", flush=True)
    print(f"Qwen parameter: qwenmodel={QWEN_MODEL_NAME} ({QWEN_ENCODER_PATH})", flush=True)

    client = None if dry_run else SwarmClient(api_url, os.environ.get("SWARM_AUTH_TOKEN"))
    manifest: list[dict[str, Any]] = []
    variants = [
        ("auto", build_swarm_smoke_payload(explicit_models=False)),
        ("explicit", build_swarm_smoke_payload(explicit_models=True)),
    ]

    for variant, payload in variants:
        destination = STAGING_DIR / f"swarm_smoke_{variant}_1024.png"
        manifest.append(
            {
                "variant": variant,
                "filename": destination.name,
                "payload": payload,
                "success_criteria": "Valid centered 1024x1024 black hole halo asset with sharp starfield.",
            }
        )
        print(f"[{variant}] 1024x1024 -> {destination}", flush=True)
        if dry_run:
            continue
        assert client is not None
        images = client.generate(payload)
        client.download_or_copy(images[0], destination)
        print(f"    staged {destination}", flush=True)

    manifest_path = STAGING_DIR / "swarm_smoke_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote smoke-test manifest: {manifest_path}", flush=True)
    if dry_run:
        print("Dry run only; no SwarmUI generation requests were sent.", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and stage SwarmUI assets from render_backlog.md.")
    parser.add_argument("--dry-run", action="store_true", help="Validate backlog and write payload manifest only.")
    parser.add_argument(
        "--swarm-smoke-test",
        action="store_true",
        help="Run the 1024x1024 auto-vs-explicit VAE/Qwen nomenclature smoke test.",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("SWARM_API_URL", SWARM_API_URL),
        help="SwarmUI root URL. Defaults to SWARM_API_URL env var or http://localhost:7801.",
    )
    args = parser.parse_args()
    try:
        if args.swarm_smoke_test:
            return run_swarm_smoke_test(dry_run=args.dry_run, api_url=args.api_url)
        return run(dry_run=args.dry_run, api_url=args.api_url)
    except Exception as exc:
        print(f"orchestrator.py: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
