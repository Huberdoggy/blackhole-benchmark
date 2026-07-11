#!/usr/bin/env python3
"""
Relativistic black hole raytracer and performance benchmark.

Main render path:
- pygame creates an OpenGL 4.6 core-profile window.
- ModernGL compiles and drives the fullscreen GLSL raymarcher.
- Dear PyGui runs an independent diagnostics/control viewport.
"""

from __future__ import annotations

import argparse
import math
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import dearpygui.dearpygui as dpg
import moderngl
import numpy as np
import pygame


BENCHMARK_TIERS = [
    ("720p", 1280, 720, 10.0, 1.0),
    ("1080p", 1920, 1080, 10.0, 2.5),
    ("4K", 3840, 2160, 10.0, 6.0),
]

ASSET_DIR = Path(__file__).resolve().parent / "src" / "assets" / "staging"
SPACE_ASSET = ASSET_DIR / "deep_space_canvas.png"
PANEL_ASSET = ASSET_DIR / "vivid_control_panel.png"
DISK_ASSET = ASSET_DIR / "photoreal_accretion_disk.png"
SINGULARITY_ASSET = ASSET_DIR / "high_fidel_singularity.png"


VERTEX_SHADER = """
#version 460 core

in vec2 in_pos;
out vec2 v_uv;

void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""


FRAGMENT_SHADER = """
#version 460 core

in vec2 v_uv;
out vec4 fragColor;

uniform vec2 u_resolution;
uniform float u_time;
uniform float u_mass;
uniform float u_disk_inner;
uniform float u_disk_outer;
uniform vec3 u_camera_pos;
uniform sampler2D u_space_tex;
uniform sampler2D u_disk_tex;
uniform sampler2D u_singularity_tex;

const int MAX_STEPS = 160;
const float FAR_CLIP = 90.0;
const float STEP_SIZE = 0.105;
const float G = 1.0;
const float C = 1.0;
const float PI = 3.141592653589793;

vec2 sphereUV(vec3 dir) {
    dir = normalize(dir);
    return vec2(atan(dir.z, dir.x) / (2.0 * PI) + 0.5, asin(clamp(dir.y, -1.0, 1.0)) / PI + 0.5);
}

float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

float starNoise(vec3 dir) {
    vec3 cell = floor(dir * 820.0);
    float h = hash13(cell);
    float star = smoothstep(0.9972, 1.0, h);
    float fine = hash13(cell * 1.37 + 7.1);
    return star * (0.45 + 1.85 * fine);
}

vec3 starfield(vec3 dir, float lens) {
    float stars = starNoise(normalize(dir));
    float milky = pow(max(0.0, 1.0 - abs(dir.y * 2.4 + 0.15)), 5.0);
    vec3 base = vec3(0.002, 0.004, 0.010);
    vec3 nebula = vec3(0.05, 0.08, 0.18) * milky * (0.25 + 0.75 * hash13(dir * 11.0));
    vec3 color = base + nebula + vec3(0.75, 0.82, 1.0) * stars;
    color += vec3(0.12, 0.18, 0.36) * lens;
    vec2 uv = sphereUV(dir);
    vec3 canvas = texture(u_space_tex, fract(uv + vec2(u_time * 0.0015, 0.0))).rgb;
    float canvas_luma = dot(canvas, vec3(0.299, 0.587, 0.114));
    vec3 directed = canvas * (0.33 + lens * 0.32) + color * (0.72 + canvas_luma * 0.42);
    return mix(color, directed, 0.68);
}

vec3 temperatureColor(float t) {
    vec3 red = vec3(1.35, 0.16, 0.035);
    vec3 amber = vec3(1.55, 0.62, 0.10);
    vec3 white = vec3(1.45, 1.25, 0.92);
    vec3 blue = vec3(0.42, 0.72, 1.72);
    vec3 warm = mix(red, amber, smoothstep(0.0, 0.45, t));
    vec3 hot = mix(white, blue, smoothstep(0.55, 1.0, t));
    return mix(warm, hot, smoothstep(0.35, 0.82, t));
}

mat3 lookAt(vec3 ro, vec3 target) {
    vec3 f = normalize(target - ro);
    vec3 r = normalize(cross(f, vec3(0.0, 1.0, 0.0)));
    vec3 u = cross(r, f);
    return mat3(r, u, f);
}

void main() {
    vec2 p = (gl_FragCoord.xy * 2.0 - u_resolution.xy) / u_resolution.y;
    vec3 ro = u_camera_pos;
    vec3 target = vec3(0.0, 0.0, 0.0);
    vec3 rd = normalize(lookAt(ro, target) * normalize(vec3(p, 1.75)));

    float schwarzschild = max(0.18, 0.42 * u_mass);
    vec3 pos = ro;
    vec3 disk_accum = vec3(0.0);
    float disk_alpha = 0.0;
    float min_r = 1e9;
    float lens_glow = 0.0;
    bool captured = false;

    for (int i = 0; i < MAX_STEPS; ++i) {
        vec3 prev = pos;
        float dist = length(pos);
        min_r = min(min_r, dist);

        if (dist < schwarzschild) {
            captured = true;
            break;
        }

        vec3 gravity_dir = -normalize(pos);
        float inv_cube = 1.0 / max(dist * dist * dist, 0.02);
        float bend = G * u_mass * inv_cube * STEP_SIZE * 1.9;
        rd = normalize(rd + gravity_dir * bend);

        pos += rd * STEP_SIZE;

        if ((prev.y > 0.0 && pos.y <= 0.0) || (prev.y < 0.0 && pos.y >= 0.0)) {
            float t = abs(prev.y) / max(abs(prev.y) + abs(pos.y), 0.0001);
            vec3 hit = mix(prev, pos, t);
            float radius = length(hit.xz);
            if (radius > u_disk_inner && radius < u_disk_outer && disk_alpha < 0.98) {
                float radial = clamp((radius - u_disk_inner) / max(u_disk_outer - u_disk_inner, 0.001), 0.0, 1.0);
                float temp = exp(-2.8 * radial);
                float banding = 0.65 + 0.35 * sin(radius * 28.0 - u_time * 5.0 + atan(hit.z, hit.x) * 9.0);
                vec3 tangent = normalize(vec3(-hit.z, 0.0, hit.x));
                float doppler = dot(tangent, normalize(ro - hit));
                float beam = pow(max(0.12, 1.0 + 0.72 * doppler), 2.5);
                vec3 doppler_tint = mix(vec3(1.18, 0.18, 0.04), vec3(0.42, 0.76, 1.95), smoothstep(-0.55, 0.65, doppler));
                float theta = atan(hit.z, hit.x) / (2.0 * PI) + 0.5;
                vec2 disk_uv = vec2(fract(theta + u_time * 0.018), radial);
                vec3 disk_plate = texture(u_disk_tex, disk_uv).rgb;
                float disk_detail = dot(disk_plate, vec3(0.299, 0.587, 0.114));
                float edge = smoothstep(u_disk_inner, u_disk_inner + 0.15, radius) *
                             (1.0 - smoothstep(u_disk_outer - 0.7, u_disk_outer, radius));
                vec3 emission = temperatureColor(temp) * doppler_tint * beam * banding * edge;
                emission *= mix(1.0, 0.78 + disk_detail * 0.72, 0.30);
                float alpha = clamp(edge * (0.23 + temp * 0.62) / (1.0 + disk_alpha), 0.0, 0.8);
                disk_accum += emission * alpha * (1.0 - disk_alpha);
                disk_alpha += alpha * (1.0 - disk_alpha);
            }
        }

        float ring = abs(dist - schwarzschild * 1.62);
        lens_glow += exp(-ring * 9.0) * 0.006;

        if (dist > FAR_CLIP) {
            break;
        }
    }

    float lens = clamp(lens_glow + 0.65 / max(min_r * min_r, 0.25), 0.0, 1.0);
    vec3 bg_dir = normalize(rd + normalize(pos) * lens * 0.34);
    vec3 color = starfield(bg_dir, lens);
    color = mix(color, disk_accum + color * (1.0 - disk_alpha), clamp(disk_alpha, 0.0, 1.0));

    float photon = exp(-abs(min_r - schwarzschild * 1.48) * 14.0);
    vec2 horizon_uv = p / max(3.25, schwarzschild * 1.25) + 0.5;
    vec3 singularity_plate = texture(u_singularity_tex, clamp(horizon_uv, vec2(0.0), vec2(1.0))).rgb;
    float horizon_mask = smoothstep(0.72, 0.0, length(horizon_uv - 0.5));
    color = mix(color, color * 0.72 + singularity_plate * 0.66, horizon_mask * 0.45);
    color += (vec3(1.0, 0.62, 0.20) + singularity_plate * 0.45) * photon * 0.23;

    if (captured) {
        float rim = exp(-abs(min_r - schwarzschild) * 28.0);
        color = mix(color, singularity_plate * 0.10, 0.94);
        color += (vec3(0.9, 0.48, 0.16) + singularity_plate * 0.35) * rim * 0.35;
    }

    color = color / (color + vec3(1.0));
    color = pow(color, vec3(0.4545));
    fragColor = vec4(color, 1.0);
}
"""


@dataclass
class Controls:
    mass: float = 4.0
    disk_inner: float = 1.55
    disk_outer: float = 7.3
    camera_distance: float = 11.0
    camera_height: float = 2.25
    camera_orbit: float = 0.0
    benchmark_requested: bool = False
    quit_requested: bool = False


@dataclass
class Metrics:
    fps: float = 0.0
    frame_ms: float = 0.0
    width: int = 1280
    height: int = 720
    active_tier: str = "Interactive"
    score: Optional[float] = None
    tier_results: dict[str, float] = field(default_factory=dict)


@dataclass
class SharedState:
    controls: Controls = field(default_factory=Controls)
    metrics: Metrics = field(default_factory=Metrics)
    lock: threading.Lock = field(default_factory=threading.Lock)


class HudThread(threading.Thread):
    def __init__(self, state: SharedState) -> None:
        super().__init__(daemon=True)
        self.state = state

    def run(self) -> None:
        dpg.create_context()
        dpg.bind_theme(self._build_theme())
        with dpg.window(label="Black Hole Diagnostics", tag="hud_window", width=430, height=500):
            dpg.add_drawlist(width=392, height=126, tag="hud_scope")
            dpg.add_spacer(height=3)
            with dpg.group(horizontal=True):
                dpg.add_text("FPS", color=(85, 226, 255, 255))
                dpg.add_text("--", tag="fps_text")
                dpg.add_spacer(width=16)
                dpg.add_text("FRAME", color=(255, 176, 77, 255))
                dpg.add_text("--", tag="frame_text")
            with dpg.group(horizontal=True):
                dpg.add_text("RES", color=(85, 226, 255, 255))
                dpg.add_text("--", tag="resolution_text")
                dpg.add_spacer(width=16)
                dpg.add_text("MODE", color=(255, 176, 77, 255))
                dpg.add_text("Interactive", tag="tier_text")
            dpg.add_separator()
            with dpg.child_window(height=72, border=True):
                dpg.add_text("BLACK HOLE", color=(85, 226, 255, 255))
                self._add_slider_row("Mass", "mass_value", 4.0, 0.6, 12.0, self._set_mass)
            with dpg.child_window(height=96, border=True):
                dpg.add_text("ACCRETION DISK", color=(255, 176, 77, 255))
                self._add_slider_row("Inner", "inner_value", 1.55, 0.55, 4.5, self._set_inner)
                self._add_slider_row("Outer", "outer_value", 7.3, 3.5, 14.0, self._set_outer)
            with dpg.child_window(height=96, border=True):
                dpg.add_text("CAMERA", color=(85, 226, 255, 255))
                self._add_slider_row("Distance", "camera_distance_value", 11.0, 4.0, 24.0, self._set_camera_distance)
                self._add_slider_row("Height", "camera_height_value", 2.25, -6.0, 8.0, self._set_camera_height)
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label="Run Benchmark", width=185, callback=self._request_benchmark)
                dpg.add_button(label="Quit", width=90, callback=self._request_quit)

        with dpg.window(label="Benchmark Complete", modal=True, show=False, tag="score_modal", no_resize=True, width=390, height=210):
            dpg.add_text("", tag="score_text")
            dpg.add_button(label="Close", callback=lambda: dpg.configure_item("score_modal", show=False))

        dpg.create_viewport(title="Black Hole Benchmark HUD", width=460, height=540)
        dpg.setup_dearpygui()
        dpg.show_viewport()

        score_seen: Optional[float] = None
        while dpg.is_dearpygui_running():
            with self.state.lock:
                metrics = Metrics(
                    fps=self.state.metrics.fps,
                    frame_ms=self.state.metrics.frame_ms,
                    width=self.state.metrics.width,
                    height=self.state.metrics.height,
                    active_tier=self.state.metrics.active_tier,
                    score=self.state.metrics.score,
                    tier_results=dict(self.state.metrics.tier_results),
                )
                quit_requested = self.state.controls.quit_requested
            dpg.set_value("fps_text", f"FPS: {metrics.fps:7.2f}")
            dpg.set_value("frame_text", f"{metrics.frame_ms:7.3f} ms")
            dpg.set_value("resolution_text", f"{metrics.width} x {metrics.height}")
            dpg.set_value("tier_text", metrics.active_tier)
            self._draw_scope(metrics, time.perf_counter())
            if metrics.score is not None and metrics.score != score_seen:
                details = "\n".join(f"{name}: {fps:.2f} FPS" for name, fps in metrics.tier_results.items())
                dpg.set_value("score_text", f"Weighted Score: {metrics.score:.2f}\n\n{details}")
                dpg.configure_item("score_modal", show=True)
                score_seen = metrics.score
            if quit_requested:
                dpg.stop_dearpygui()
            dpg.render_dearpygui_frame()
        dpg.destroy_context()

    @staticmethod
    def _add_slider_row(label: str, value_tag: str, default: float, minimum: float, maximum: float, callback) -> None:
        with dpg.group(horizontal=True):
            dpg.add_text(label)
            dpg.add_slider_float(label="", default_value=default, min_value=minimum, max_value=maximum, width=230, format="", callback=callback)
            dpg.add_text(f"{default:5.2f}", tag=value_tag, color=(160, 229, 240, 255))

    @staticmethod
    def _build_theme() -> int:
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (8, 12, 18, 244))
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (10, 17, 25, 232))
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (12, 16, 22, 250))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (220, 235, 238, 255))
                dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (98, 120, 126, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Border, (53, 178, 205, 115))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (15, 28, 37, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (24, 57, 70, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (34, 83, 96, 255))
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (78, 225, 255, 255))
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (255, 183, 76, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (22, 49, 60, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (36, 92, 108, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (210, 126, 43, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Separator, (67, 174, 198, 120))
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 14, 12)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 7)
        return theme

    @staticmethod
    def _draw_scope(metrics: Metrics, now: float) -> None:
        if not dpg.does_item_exist("hud_scope"):
            return
        dpg.delete_item("hud_scope", children_only=True)
        parent = "hud_scope"
        dpg.draw_rectangle((0, 0), (392, 126), color=(72, 219, 255, 120), fill=(7, 13, 19, 245), rounding=6, thickness=1, parent=parent)
        dpg.draw_rectangle((8, 8), (384, 118), color=(255, 175, 72, 45), fill=(10, 22, 30, 210), rounding=4, thickness=1, parent=parent)
        for x in range(24, 376, 32):
            dpg.draw_line((x, 18), (x, 108), color=(55, 145, 170, 38), thickness=1, parent=parent)
        for y in range(28, 108, 20):
            dpg.draw_line((18, y), (374, y), color=(55, 145, 170, 34), thickness=1, parent=parent)
        dpg.draw_text((20, 16), "RAYTRACE TELEMETRY", color=(122, 232, 255, 230), size=13, parent=parent)
        dpg.draw_text((252, 16), metrics.active_tier.upper()[:16], color=(255, 180, 78, 230), size=13, parent=parent)

        points = []
        phase = now * 3.1
        frame_factor = min(metrics.frame_ms / 32.0, 1.0)
        for i in range(92):
            x = 20 + i * 3.78
            y = 80 + math.sin(i * 0.21 + phase) * 15 + math.cos(i * 0.075 + phase * 0.7) * 7
            y += frame_factor * math.sin(i * 0.47 + phase * 1.5) * 8
            points.append((x, y))
        dpg.draw_polyline(points, color=(82, 231, 255, 210), thickness=2, parent=parent)
        dpg.draw_polyline([(x, y + 6) for x, y in points[::2]], color=(255, 154, 58, 115), thickness=1, parent=parent)

        fps_norm = min(metrics.fps / 120.0, 1.0)
        frame_norm = 1.0 - min(metrics.frame_ms / 33.333, 1.0)
        dpg.draw_rectangle((22, 110), (22 + 150 * fps_norm, 116), color=(78, 225, 255, 190), fill=(78, 225, 255, 165), rounding=2, parent=parent)
        dpg.draw_rectangle((220, 110), (220 + 150 * frame_norm, 116), color=(255, 176, 77, 190), fill=(255, 176, 77, 150), rounding=2, parent=parent)
        dpg.draw_circle((348, 66), 24, color=(84, 226, 255, 155), thickness=2, parent=parent)
        dpg.draw_circle((348, 66), 9 + 9 * fps_norm, color=(255, 176, 77, 160), thickness=2, parent=parent)

    def _set_mass(self, _sender: int, value: float) -> None:
        with self.state.lock:
            self.state.controls.mass = float(value)
        self._set_readout("mass_value", value)

    def _set_inner(self, _sender: int, value: float) -> None:
        with self.state.lock:
            self.state.controls.disk_inner = float(value)
        self._set_readout("inner_value", value)

    def _set_outer(self, _sender: int, value: float) -> None:
        with self.state.lock:
            self.state.controls.disk_outer = float(value)
        self._set_readout("outer_value", value)

    def _set_camera_distance(self, _sender: int, value: float) -> None:
        with self.state.lock:
            self.state.controls.camera_distance = float(value)
        self._set_readout("camera_distance_value", value)

    def _set_camera_height(self, _sender: int, value: float) -> None:
        with self.state.lock:
            self.state.controls.camera_height = float(value)
        self._set_readout("camera_height_value", value)

    @staticmethod
    def _set_readout(tag: str, value: float) -> None:
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, f"{value:5.2f}")

    def _request_benchmark(self) -> None:
        with self.state.lock:
            self.state.controls.benchmark_requested = True

    def _request_quit(self) -> None:
        with self.state.lock:
            self.state.controls.quit_requested = True


class BlackHoleBenchmark:
    def __init__(self, width: int, height: int, smoke_test: float = 0.0, tier_duration: Optional[float] = None) -> None:
        self.width = width
        self.height = height
        self.smoke_test = smoke_test
        self.tier_duration = tier_duration
        self.state = SharedState()
        self.ctx: Optional[moderngl.Context] = None
        self.program: Optional[moderngl.Program] = None
        self.vao: Optional[moderngl.VertexArray] = None
        self.textures: dict[str, moderngl.Texture] = {}
        self.running_benchmark = False
        self.frame_times: list[float] = []
        self.benchmark_index = 0
        self.tier_start = 0.0
        self.benchmark_results: dict[str, float] = {}
        self.last_time = time.perf_counter()
        self.start_time = self.last_time

    def setup(self) -> None:
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 6)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        pygame.display.set_caption("Relativistic Black Hole Raytracer Benchmark")
        self._set_resolution(self.width, self.height)
        self.ctx = moderngl.create_context()
        self.ctx.enable_only(moderngl.NOTHING)
        self.program = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self._load_asset_textures()
        self.program["u_space_tex"] = 0
        self.program["u_disk_tex"] = 1
        self.program["u_singularity_tex"] = 2
        vertices = np.array([-1.0, -1.0, 3.0, -1.0, -1.0, 3.0], dtype="f4")
        vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(self.program, [(vbo, "2f", "in_pos")])
        HudThread(self.state).start()

    def _load_asset_textures(self) -> None:
        assert self.ctx is not None
        self.textures = {
            "space": self._load_texture(SPACE_ASSET, (2, 4, 10)),
            "disk": self._load_texture(DISK_ASSET, (255, 120, 30)),
            "singularity": self._load_texture(SINGULARITY_ASSET, (8, 6, 10)),
        }

    def _load_texture(self, path: Path, fallback_rgb: tuple[int, int, int]) -> moderngl.Texture:
        assert self.ctx is not None
        if path.exists():
            surface = pygame.image.load(str(path)).convert()
        else:
            surface = pygame.Surface((1, 1))
            surface.fill(fallback_rgb)
        data = pygame.image.tostring(surface, "RGB", True)
        texture = self.ctx.texture(surface.get_size(), 3, data)
        texture.repeat_x = True
        texture.repeat_y = True
        texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        texture.build_mipmaps()
        return texture

    def _set_resolution(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)
        pygame.display.set_mode((self.width, self.height), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE, vsync=0)
        if self.ctx is not None:
            self.ctx.viewport = (0, 0, self.width, self.height)
        with self.state.lock:
            self.state.metrics.width = self.width
            self.state.metrics.height = self.height

    def run(self) -> None:
        self.setup()
        try:
            while self._should_continue():
                self._handle_events()
                now = time.perf_counter()
                dt = max(now - self.last_time, 1e-6)
                self.last_time = now
                self._maybe_start_benchmark()
                self._update_benchmark(now)
                self._render(now, dt)
        finally:
            with self.state.lock:
                self.state.controls.quit_requested = True
            time.sleep(0.05)
            pygame.quit()

    def _should_continue(self) -> bool:
        with self.state.lock:
            quit_requested = self.state.controls.quit_requested
        if quit_requested:
            return False
        if self.smoke_test > 0 and time.perf_counter() - self.start_time >= self.smoke_test:
            return False
        return True

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                with self.state.lock:
                    self.state.controls.quit_requested = True
            elif event.type == pygame.VIDEORESIZE and not self.running_benchmark:
                self._set_resolution(max(320, event.w), max(240, event.h))
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    with self.state.lock:
                        self.state.controls.quit_requested = True
                elif event.key == pygame.K_b:
                    with self.state.lock:
                        self.state.controls.benchmark_requested = True

        pressed = pygame.key.get_pressed()
        if not self.running_benchmark:
            orbit_delta = 0.0
            if pressed[pygame.K_LEFT] or pressed[pygame.K_a]:
                orbit_delta -= 0.025
            if pressed[pygame.K_RIGHT] or pressed[pygame.K_d]:
                orbit_delta += 0.025
            if orbit_delta:
                with self.state.lock:
                    self.state.controls.camera_orbit += orbit_delta

    def _maybe_start_benchmark(self) -> None:
        with self.state.lock:
            requested = self.state.controls.benchmark_requested
            self.state.controls.benchmark_requested = False
        if requested and not self.running_benchmark:
            self.running_benchmark = True
            self.benchmark_index = 0
            self.benchmark_results = {}
            self.frame_times = []
            self.tier_start = time.perf_counter()
            name, width, height, _duration, _weight = BENCHMARK_TIERS[0]
            self._set_resolution(width, height)
            with self.state.lock:
                self.state.metrics.active_tier = f"Benchmark {name}"
                self.state.metrics.score = None
                self.state.metrics.tier_results = {}

    def _update_benchmark(self, now: float) -> None:
        if not self.running_benchmark:
            return
        name, _width, _height, duration, _weight = BENCHMARK_TIERS[self.benchmark_index]
        duration = self.tier_duration if self.tier_duration is not None else duration
        if now - self.tier_start < duration:
            return
        avg_fps = self._filtered_fps(self.frame_times)
        self.benchmark_results[name] = avg_fps
        self.benchmark_index += 1
        if self.benchmark_index >= len(BENCHMARK_TIERS):
            score = sum(self.benchmark_results[tier] * weight for tier, _w, _h, _d, weight in BENCHMARK_TIERS)
            self.running_benchmark = False
            with self.state.lock:
                self.state.metrics.active_tier = "Interactive"
                self.state.metrics.score = score
                self.state.metrics.tier_results = dict(self.benchmark_results)
            return
        next_name, width, height, _duration, _weight = BENCHMARK_TIERS[self.benchmark_index]
        self.frame_times = []
        self.tier_start = now
        self._set_resolution(width, height)
        with self.state.lock:
            self.state.metrics.active_tier = f"Benchmark {next_name}"

    @staticmethod
    def _filtered_fps(frame_times: list[float]) -> float:
        if not frame_times:
            return 0.0
        ordered = sorted(frame_times)
        trim = int(len(ordered) * 0.01)
        if trim > 0 and len(ordered) > trim * 2:
            ordered = ordered[trim:-trim]
        avg_ms = statistics.fmean(ordered)
        return 1000.0 / max(avg_ms, 1e-6)

    def _render(self, now: float, dt: float) -> None:
        assert self.ctx is not None and self.program is not None and self.vao is not None
        with self.state.lock:
            controls = Controls(
                mass=self.state.controls.mass,
                disk_inner=self.state.controls.disk_inner,
                disk_outer=self.state.controls.disk_outer,
                camera_distance=self.state.controls.camera_distance,
                camera_height=self.state.controls.camera_height,
                camera_orbit=self.state.controls.camera_orbit,
            )

        if self.running_benchmark:
            elapsed = now - self.tier_start
            controls.camera_orbit = elapsed * 0.18 + self.benchmark_index * 1.7
            controls.camera_distance = 10.5 + math.sin(elapsed * 0.21) * 1.2
            controls.camera_height = 2.1 + math.sin(elapsed * 0.17) * 0.65

        disk_outer = max(controls.disk_outer, controls.disk_inner + 0.4)
        camera = (
            math.sin(controls.camera_orbit) * controls.camera_distance,
            controls.camera_height,
            math.cos(controls.camera_orbit) * controls.camera_distance,
        )

        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.textures["space"].use(location=0)
        self.textures["disk"].use(location=1)
        self.textures["singularity"].use(location=2)
        self.program["u_resolution"] = (float(self.width), float(self.height))
        self.program["u_time"] = float(now - self.start_time)
        self.program["u_mass"] = float(controls.mass)
        self.program["u_disk_inner"] = float(controls.disk_inner)
        self.program["u_disk_outer"] = float(disk_outer)
        self.program["u_camera_pos"] = camera
        self.vao.render(mode=moderngl.TRIANGLES, vertices=3)
        pygame.display.flip()

        frame_ms = dt * 1000.0
        fps = 1.0 / dt
        if self.running_benchmark:
            self.frame_times.append(frame_ms)
        with self.state.lock:
            self.state.metrics.fps = fps
            self.state.metrics.frame_ms = frame_ms
            self.state.metrics.width = self.width
            self.state.metrics.height = self.height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relativistic black hole raytracer benchmark.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--smoke-test", type=float, default=0.0, help="Run for N seconds and exit.")
    parser.add_argument("--tier-duration", type=float, default=None, help="Override benchmark tier duration for testing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = BlackHoleBenchmark(args.width, args.height, smoke_test=args.smoke_test, tier_duration=args.tier_duration)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
