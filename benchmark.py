#!/usr/bin/env python3
"""
Relativistic black hole raytracer and performance benchmark.

Main render path:
- pygame creates an OpenGL 4.6 core-profile window.
- ModernGL compiles and drives the fullscreen GLSL raymarcher.
- GLFW owns an independent OpenGL control window rendered by Skia.
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

import glfw
import moderngl
import numpy as np
import pygame
import skia
from OpenGL import GL

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
CONTROL_PANEL_WIDTH = 980
CONTROL_PANEL_HEIGHT = 620


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

mat2 rot2(float a) {
    float s = sin(a);
    float c = cos(a);
    return mat2(c, -s, s, c);
}

vec2 swirlUV(vec2 uv, float spin, float shear) {
    vec2 q = uv - 0.5;
    float r = max(length(q), 0.001);
    float angle = spin + shear / (r * 1.9 + 0.08);
    q = rot2(angle) * q;
    return q + 0.5;
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
                vec3 doppler_tint = mix(
                    vec3(1.18, 0.18, 0.04),
                    vec3(0.42, 0.76, 1.95),
                    smoothstep(-0.55, 0.65, doppler)
                );
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
    vec2 horizon_base = p / max(3.25, schwarzschild * 1.25) + 0.5;
    float r_halo = length(horizon_base - 0.5);
    vec2 inner_uv = swirlUV(horizon_base, u_time * 0.10, 0.18);
    vec2 outer_uv = swirlUV(horizon_base, -u_time * 0.035, -0.08);
    vec3 singularity_inner = texture(u_singularity_tex, clamp(inner_uv, vec2(0.0), vec2(1.0))).rgb;
    vec3 singularity_outer = texture(u_singularity_tex, clamp(outer_uv, vec2(0.0), vec2(1.0))).rgb;
    vec3 singularity_plate = mix(singularity_outer, singularity_inner, smoothstep(0.62, 0.18, r_halo));
    float halo_angle = atan(horizon_base.y - 0.5, horizon_base.x - 0.5);
    float rotational_lobe = sin(halo_angle + u_time * 0.45);
    vec3 blue_boost = vec3(0.55, 0.78, 1.35);
    vec3 red_falloff = vec3(1.25, 0.46, 0.18);
    singularity_plate *= mix(red_falloff, blue_boost, smoothstep(-0.45, 0.65, rotational_lobe));
    float horizon_mask = smoothstep(0.72, 0.0, r_halo);
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


@dataclass(frozen=True)
class SliderControl:
    label: str
    field_name: str
    minimum: float
    maximum: float
    x: float
    y: float
    width: float
    accent: tuple[int, int, int]


class HudThread(threading.Thread):
    def __init__(self, state: SharedState) -> None:
        super().__init__(daemon=True)
        self.state = state
        self.sliders = [
            SliderControl("SINGULARITY MASS", "mass", 0.6, 12.0, 72.0, 418.0, 350.0, (77, 225, 255)),
            SliderControl("DISK INNER RADIUS", "disk_inner", 0.55, 4.5, 72.0, 482.0, 350.0, (255, 178, 78)),
            SliderControl("DISK OUTER RADIUS", "disk_outer", 3.5, 14.0, 72.0, 546.0, 350.0, (255, 98, 54)),
            SliderControl("CAMERA DISTANCE", "camera_distance", 4.0, 24.0, 532.0, 418.0, 350.0, (77, 225, 255)),
            SliderControl("CAMERA HEIGHT", "camera_height", -6.0, 8.0, 532.0, 482.0, 350.0, (180, 232, 214)),
            SliderControl("CAMERA ORBIT", "camera_orbit", -math.pi, math.pi, 532.0, 546.0, 350.0, (190, 226, 255)),
        ]
        self.dragging_slider: Optional[SliderControl] = None
        self.mouse_pos = (0.0, 0.0)
        self.frame_history: list[float] = []
        self.score_seen: Optional[float] = None
        self.score_dialog_visible = False
        self.panel_image: Optional[skia.Image] = skia.Image.open(str(PANEL_ASSET)) if PANEL_ASSET.exists() else None
        self.typeface = skia.Typeface.MakeFromName("DejaVu Sans Mono", skia.FontStyle.Normal())

    def run(self) -> None:
        if not glfw.init():
            raise RuntimeError("glfw.init() failed for the Skia control panel window.")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.STENCIL_BITS, 8)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
        window = glfw.create_window(
            CONTROL_PANEL_WIDTH,
            CONTROL_PANEL_HEIGHT,
            "Black Hole Benchmark Control Console",
            None,
            None,
        )
        if window is None:
            glfw.terminate()
            raise RuntimeError("Could not create the Skia control panel OpenGL window.")

        glfw.make_context_current(window)
        glfw.swap_interval(1)
        glfw.set_cursor_pos_callback(window, self._on_cursor)
        glfw.set_mouse_button_callback(window, self._on_mouse_button)
        glfw.set_key_callback(window, self._on_key)
        context = skia.GrDirectContext.MakeGL()

        try:
            while not glfw.window_should_close(window):
                glfw.poll_events()
                with self.state.lock:
                    quit_requested = self.state.controls.quit_requested
                if quit_requested:
                    glfw.set_window_should_close(window, True)
                    break
                self._render_window(window, context)
                glfw.swap_buffers(window)
                time.sleep(1.0 / 120.0)
        finally:
            context.abandonContext()
            glfw.destroy_window(window)
            glfw.terminate()
            with self.state.lock:
                self.state.controls.quit_requested = True

    @staticmethod
    def _color(r: int, g: int, b: int, a: int = 255) -> int:
        return skia.ColorSetARGB(a, r, g, b)

    @staticmethod
    def _paint(color: int, style: skia.Paint.Style = skia.Paint.kFill_Style, stroke_width: float = 1.0) -> skia.Paint:
        paint = skia.Paint(Color=color, AntiAlias=True)
        paint.setStyle(style)
        paint.setStrokeWidth(stroke_width)
        return paint

    def _font(self, size: float) -> skia.Font:
        return skia.Font(self.typeface, size)

    def _render_window(self, window: glfw._GLFWwindow, context: skia.GrDirectContext) -> None:
        fb_width, fb_height = glfw.get_framebuffer_size(window)
        if fb_width <= 0 or fb_height <= 0:
            return
        GL.glViewport(0, 0, fb_width, fb_height)
        backend_target = skia.GrBackendRenderTarget(
            fb_width,
            fb_height,
            0,
            8,
            skia.GrGLFramebufferInfo(0, GL.GL_RGBA8),
        )
        surface = skia.Surface.MakeFromBackendRenderTarget(
            context,
            backend_target,
            skia.kBottomLeft_GrSurfaceOrigin,
            skia.kRGBA_8888_ColorType,
            skia.ColorSpace.MakeSRGB(),
        )
        if surface is None:
            return
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
            controls = Controls(
                mass=self.state.controls.mass,
                disk_inner=self.state.controls.disk_inner,
                disk_outer=self.state.controls.disk_outer,
                camera_distance=self.state.controls.camera_distance,
                camera_height=self.state.controls.camera_height,
                camera_orbit=self.state.controls.camera_orbit,
            )
        now = time.perf_counter()
        self.frame_history.append(metrics.frame_ms)
        self.frame_history = self.frame_history[-180:]
        if metrics.score is not None and metrics.score != self.score_seen:
            self.score_seen = metrics.score
            self.score_dialog_visible = True

        canvas = surface.getCanvas()
        canvas.clear(self._color(5, 7, 10))
        canvas.save()
        canvas.scale(fb_width / CONTROL_PANEL_WIDTH, fb_height / CONTROL_PANEL_HEIGHT)
        self._draw_console(canvas, metrics, controls, now)
        if self.score_dialog_visible:
            self._draw_score_dialog(canvas, metrics)
        canvas.restore()
        surface.flushAndSubmit()

    def _draw_console(self, canvas: skia.Canvas, metrics: Metrics, controls: Controls, now: float) -> None:
        if self.panel_image is not None:
            source = skia.Rect.MakeWH(self.panel_image.width(), self.panel_image.height())
            dest = skia.Rect.MakeWH(CONTROL_PANEL_WIDTH, CONTROL_PANEL_HEIGHT)
            canvas.drawImageRect(self.panel_image, source, dest, skia.SamplingOptions())
            canvas.drawRect(dest, self._paint(self._color(5, 8, 11, 82)))
        else:
            canvas.drawRect(
                skia.Rect.MakeWH(CONTROL_PANEL_WIDTH, CONTROL_PANEL_HEIGHT),
                self._paint(self._color(32, 35, 36)),
            )

        self._draw_outer_frame(canvas)
        self._draw_screen(canvas, skia.Rect.MakeXYWH(54, 64, 382, 242), "RAYTRACE TELEMETRY")
        self._draw_screen(canvas, skia.Rect.MakeXYWH(544, 64, 382, 242), "BENCHMARK MATRIX")
        self._draw_trace(canvas, skia.Rect.MakeXYWH(78, 118, 334, 154), metrics, now)
        self._draw_metric_stack(canvas, metrics)
        self._draw_gauge(canvas, (490.0, 207.0), 44.0, (controls.mass - 0.6) / (12.0 - 0.6), "MASS", (77, 225, 255))
        self._draw_frame_meter(canvas, skia.Rect.MakeXYWH(448, 318, 84, 18), metrics)
        self._draw_lamps(canvas, metrics)
        for slider in self.sliders:
            self._draw_slider(canvas, slider, getattr(controls, slider.field_name), now)
        self._draw_button(canvas, skia.Rect.MakeXYWH(324, 350, 134, 36), "RESET CAMERA", (178, 214, 220))
        self._draw_button(canvas, skia.Rect.MakeXYWH(480, 350, 150, 36), "RUN BENCH", (255, 178, 78))
        self._draw_button(canvas, skia.Rect.MakeXYWH(652, 350, 84, 36), "QUIT", (255, 92, 62))

    def _draw_outer_frame(self, canvas: skia.Canvas) -> None:
        frame = skia.Rect.MakeXYWH(24, 24, CONTROL_PANEL_WIDTH - 48, CONTROL_PANEL_HEIGHT - 48)
        canvas.drawRRect(
            skia.RRect.MakeRectXY(frame, 22, 22),
            self._paint(self._color(190, 198, 193, 92), skia.Paint.kStroke_Style, 2.0),
        )
        inset = skia.Rect.MakeXYWH(38, 38, CONTROL_PANEL_WIDTH - 76, CONTROL_PANEL_HEIGHT - 76)
        canvas.drawRRect(skia.RRect.MakeRectXY(inset, 16, 16), self._paint(self._color(8, 11, 13, 130)))

    def _draw_screen(self, canvas: skia.Canvas, rect: skia.Rect, label: str) -> None:
        canvas.drawRRect(skia.RRect.MakeRectXY(rect, 10, 10), self._paint(self._color(1, 6, 9, 230)))
        canvas.drawRRect(
            skia.RRect.MakeRectXY(rect, 10, 10),
            self._paint(self._color(69, 219, 255, 115), skia.Paint.kStroke_Style, 1.3),
        )
        canvas.drawString(
            label, rect.x() + 18, rect.y() + 31, self._font(14), self._paint(self._color(120, 238, 255, 238))
        )
        for x in range(int(rect.x() + 18), int(rect.right() - 8), 28):
            canvas.drawLine(x, rect.y() + 50, x, rect.bottom() - 18, self._paint(self._color(61, 164, 184, 35)))
        for y in range(int(rect.y() + 58), int(rect.bottom() - 8), 26):
            canvas.drawLine(rect.x() + 14, y, rect.right() - 14, y, self._paint(self._color(61, 164, 184, 31)))

    def _draw_trace(self, canvas: skia.Canvas, rect: skia.Rect, metrics: Metrics, now: float) -> None:
        history = self.frame_history if len(self.frame_history) >= 2 else [metrics.frame_ms for _ in range(32)]
        path = skia.Path()
        for index, frame_ms in enumerate(history):
            x = rect.x() + rect.width() * index / max(len(history) - 1, 1)
            normalized = min(frame_ms / 34.0, 1.0)
            wave = math.sin(index * 0.19 + now * 2.2) * 5.0
            y = rect.bottom() - normalized * (rect.height() - 16.0) + wave
            if index == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        canvas.drawPath(path, self._paint(self._color(78, 231, 255, 230), skia.Paint.kStroke_Style, 2.3))
        fps_width = min(metrics.fps / 160.0, 1.0) * (rect.width() - 16)
        frame_width = max(0.0, 1.0 - min(metrics.frame_ms / 33.333, 1.0)) * (rect.width() - 16)
        canvas.drawRect(
            skia.Rect.MakeXYWH(rect.x() + 8, rect.bottom() - 18, fps_width, 4),
            self._paint(self._color(75, 226, 255, 185)),
        )
        canvas.drawRect(
            skia.Rect.MakeXYWH(rect.x() + 8, rect.bottom() - 10, frame_width, 4),
            self._paint(self._color(255, 176, 77, 175)),
        )

    def _draw_metric_stack(self, canvas: skia.Canvas, metrics: Metrics) -> None:
        labels = [
            ("FPS", f"{metrics.fps:7.2f}", (76, 226, 255)),
            ("FRAME", f"{metrics.frame_ms:7.3f} ms", (255, 178, 78)),
            ("RES", f"{metrics.width} x {metrics.height}", (148, 234, 218)),
            ("MODE", metrics.active_tier.upper(), (255, 98, 54)),
        ]
        x = 574
        y = 120
        for index, (label, value, color) in enumerate(labels):
            row_y = y + index * 43
            canvas.drawString(label, x, row_y, self._font(13), self._paint(self._color(*color, 225)))
            canvas.drawString(value[:24], x + 92, row_y, self._font(20), self._paint(self._color(225, 236, 232, 238)))

    def _draw_gauge(
        self,
        canvas: skia.Canvas,
        center: tuple[float, float],
        radius: float,
        value: float,
        label: str,
        color: tuple[int, int, int],
    ) -> None:
        value = min(max(value, 0.0), 1.0)
        cx, cy = center
        rect = skia.Rect.MakeXYWH(cx - radius, cy - radius, radius * 2, radius * 2)
        canvas.drawCircle(cx, cy, radius + 10, self._paint(self._color(5, 7, 8, 175)))
        canvas.drawCircle(cx, cy, radius, self._paint(self._color(12, 17, 18, 240)))
        canvas.drawArc(rect, 140, 260, False, self._paint(self._color(175, 190, 184, 120), skia.Paint.kStroke_Style, 6))
        canvas.drawArc(
            rect, 140, 260 * value, False, self._paint(self._color(*color, 230), skia.Paint.kStroke_Style, 6)
        )
        for tick in range(11):
            t = tick / 10.0
            angle = math.radians(220 - 260 * t)
            inner = radius - (13 if tick % 2 == 0 else 8)
            outer = radius - 2
            canvas.drawLine(
                cx + math.cos(angle) * inner,
                cy - math.sin(angle) * inner,
                cx + math.cos(angle) * outer,
                cy - math.sin(angle) * outer,
                self._paint(self._color(225, 232, 220, 160), skia.Paint.kStroke_Style, 1.3),
            )
        angle = math.radians(220 - 260 * value)
        canvas.drawLine(
            cx,
            cy,
            cx + math.cos(angle) * (radius - 18),
            cy - math.sin(angle) * (radius - 18),
            self._paint(self._color(*color, 250), skia.Paint.kStroke_Style, 3.2),
        )
        canvas.drawCircle(cx, cy, 7, self._paint(self._color(238, 228, 205, 245)))
        label_width = self._font(13).measureText(label)
        canvas.drawString(
            label, cx - label_width / 2, cy + radius + 28, self._font(13), self._paint(self._color(*color, 235))
        )

    def _draw_lamps(self, canvas: skia.Canvas, metrics: Metrics) -> None:
        lamps = [
            ("INTERACTIVE", "Interactive" in metrics.active_tier, 84, 363, (77, 225, 255)),
            ("720P", "720p" in metrics.active_tier, 222, 363, (120, 238, 180)),
            ("1080P", "1080p" in metrics.active_tier, 770, 363, (255, 178, 78)),
            ("4K", "4K" in metrics.active_tier, 870, 363, (255, 96, 58)),
        ]
        for label, active, x, y, color in lamps:
            alpha = 245 if active else 85
            canvas.drawCircle(x, y, 9, self._paint(self._color(*color, alpha)))
            ring_alpha = 46 if active else 18
            canvas.drawCircle(x, y, 15, self._paint(self._color(*color, ring_alpha), skia.Paint.kStroke_Style, 2))
            canvas.drawString(label, x + 18, y + 5, self._font(12), self._paint(self._color(226, 234, 228, 220)))

    def _draw_frame_meter(self, canvas: skia.Canvas, rect: skia.Rect, metrics: Metrics) -> None:
        value = min(metrics.frame_ms / 33.333, 1.0)
        canvas.drawRRect(skia.RRect.MakeRectXY(rect, 6, 6), self._paint(self._color(8, 11, 12, 210)))
        canvas.drawRRect(
            skia.RRect.MakeRectXY(rect, 6, 6),
            self._paint(self._color(255, 178, 78, 115), skia.Paint.kStroke_Style, 1.0),
        )
        canvas.drawRect(
            skia.Rect.MakeXYWH(rect.x() + 5, rect.y() + 6, (rect.width() - 10) * value, 5),
            self._paint(self._color(255, 178, 78, 190)),
        )
        label_width = self._font(11).measureText("FRAME")
        canvas.drawString(
            "FRAME",
            rect.x() + rect.width() / 2 - label_width / 2,
            rect.bottom() + 14,
            self._font(11),
            self._paint(self._color(255, 178, 78, 225)),
        )

    def _draw_slider(self, canvas: skia.Canvas, slider: SliderControl, value: float, now: float) -> None:
        y = slider.y
        rail_x = slider.x + 190
        rail_w = slider.width - 200
        norm = (value - slider.minimum) / max(slider.maximum - slider.minimum, 1e-6)
        norm = min(max(norm, 0.0), 1.0)
        color = slider.accent
        panel_rect = skia.Rect.MakeXYWH(slider.x, y - 22, slider.width, 54)
        canvas.drawRRect(skia.RRect.MakeRectXY(panel_rect, 8, 8), self._paint(self._color(9, 12, 13, 168)))
        canvas.drawRRect(
            skia.RRect.MakeRectXY(panel_rect, 8, 8),
            self._paint(self._color(*color, 64), skia.Paint.kStroke_Style, 1.0),
        )
        pulse = 0.56 + 0.44 * math.sin(now * 2.3 + slider.x * 0.01)
        canvas.drawCircle(slider.x + 18, y + 3, 7, self._paint(self._color(*color, int(135 + 96 * pulse))))
        canvas.drawCircle(slider.x + 18, y + 3, 13, self._paint(self._color(*color, 40), skia.Paint.kStroke_Style, 2))
        canvas.drawString(
            slider.label, slider.x + 36, y - 1, self._font(12), self._paint(self._color(225, 234, 230, 220))
        )
        canvas.drawString(f"{value:6.2f}", slider.x + 36, y + 20, self._font(18), self._paint(self._color(*color, 235)))
        slot = skia.Rect.MakeXYWH(rail_x, y - 5, rail_w, 12)
        canvas.drawRRect(skia.RRect.MakeRectXY(slot, 6, 6), self._paint(self._color(3, 5, 6, 245)))
        canvas.drawRect(
            skia.Rect.MakeXYWH(rail_x + 4, y - 1, max((rail_w - 8) * norm, 1.0), 4),
            self._paint(self._color(*color, 180)),
        )
        for tick in range(9):
            tx = rail_x + 8 + (rail_w - 16) * tick / 8
            canvas.drawLine(
                tx, y - 12, tx, y + 14, self._paint(self._color(205, 214, 207, 70), skia.Paint.kStroke_Style, 1)
            )
        cap_x = rail_x + 8 + (rail_w - 16) * norm
        cap = skia.Rect.MakeXYWH(cap_x - 8, y - 18, 16, 38)
        canvas.drawRRect(skia.RRect.MakeRectXY(cap, 4, 4), self._paint(self._color(211, 219, 211, 250)))
        canvas.drawLine(
            cap_x - 4,
            y - 13,
            cap_x + 4,
            y - 13,
            self._paint(self._color(255, 255, 255, 125), skia.Paint.kStroke_Style, 1),
        )
        canvas.drawLine(
            cap_x - 4, y + 16, cap_x + 4, y + 16, self._paint(self._color(25, 29, 28, 145), skia.Paint.kStroke_Style, 1)
        )

    def _draw_button(self, canvas: skia.Canvas, rect: skia.Rect, label: str, color: tuple[int, int, int]) -> None:
        hovering = self._rect_contains(rect, *self.mouse_pos) and not self.score_dialog_visible
        fill_alpha = 205 if hovering else 165
        canvas.drawRRect(skia.RRect.MakeRectXY(rect, 8, 8), self._paint(self._color(9, 11, 12, fill_alpha)))
        canvas.drawRRect(
            skia.RRect.MakeRectXY(rect, 8, 8),
            self._paint(self._color(*color, 185 if hovering else 105), skia.Paint.kStroke_Style, 1.4),
        )
        label_width = self._font(13).measureText(label)
        canvas.drawString(
            label,
            rect.x() + rect.width() / 2 - label_width / 2,
            rect.y() + 26,
            self._font(13),
            self._paint(self._color(*color, 240)),
        )

    def _draw_score_dialog(self, canvas: skia.Canvas, metrics: Metrics) -> None:
        canvas.drawRect(
            skia.Rect.MakeWH(CONTROL_PANEL_WIDTH, CONTROL_PANEL_HEIGHT), self._paint(self._color(0, 0, 0, 154))
        )
        rect = skia.Rect.MakeXYWH(285, 165, 410, 288)
        canvas.drawRRect(skia.RRect.MakeRectXY(rect, 14, 14), self._paint(self._color(7, 10, 12, 245)))
        canvas.drawRRect(
            skia.RRect.MakeRectXY(rect, 14, 14),
            self._paint(self._color(255, 178, 78, 170), skia.Paint.kStroke_Style, 1.8),
        )
        canvas.drawString("BENCHMARK COMPLETE", 327, 213, self._font(20), self._paint(self._color(255, 178, 78, 245)))
        score = metrics.score if metrics.score is not None else 0.0
        canvas.drawString(
            f"WEIGHTED SCORE  {score:8.2f}", 327, 263, self._font(23), self._paint(self._color(79, 230, 255, 245))
        )
        y = 304
        for name, fps in metrics.tier_results.items():
            canvas.drawString(
                f"{name:<6} {fps:8.2f} FPS", 350, y, self._font(17), self._paint(self._color(226, 236, 230, 235))
            )
            y += 32
        self._draw_button(canvas, self._score_close_rect(), "CLOSE", (255, 178, 78))

    @staticmethod
    def _score_close_rect() -> skia.Rect:
        return skia.Rect.MakeXYWH(421, 396, 138, 36)

    @staticmethod
    def _rect_contains(rect: skia.Rect, x: float, y: float) -> bool:
        return rect.x() <= x <= rect.right() and rect.y() <= y <= rect.bottom()

    def _on_cursor(self, window: glfw._GLFWwindow, x: float, y: float) -> None:
        self.mouse_pos = self._map_mouse(window, x, y)
        if self.dragging_slider is not None:
            self._set_slider_from_x(self.dragging_slider, self.mouse_pos[0])

    def _on_mouse_button(self, window: glfw._GLFWwindow, button: int, action: int, _mods: int) -> None:
        if button != glfw.MOUSE_BUTTON_LEFT:
            return
        if action == glfw.PRESS:
            x_raw, y_raw = glfw.get_cursor_pos(window)
            self.mouse_pos = self._map_mouse(window, x_raw, y_raw)
            x, y = self.mouse_pos
            if self.score_dialog_visible:
                if self._rect_contains(self._score_close_rect(), x, y):
                    self.score_dialog_visible = False
                return
            for slider in self.sliders:
                if slider.x <= x <= slider.x + slider.width and slider.y - 24 <= y <= slider.y + 28:
                    self.dragging_slider = slider
                    self._set_slider_from_x(slider, x)
                    return
            self._handle_button_press(x, y)
        elif action == glfw.RELEASE:
            self.dragging_slider = None

    def _on_key(self, _window: glfw._GLFWwindow, key: int, _scancode: int, action: int, _mods: int) -> None:
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            self._request_quit()
        elif key == glfw.KEY_B:
            self._request_benchmark()

    def _map_mouse(self, window: glfw._GLFWwindow, x: float, y: float) -> tuple[float, float]:
        win_width, win_height = glfw.get_window_size(window)
        mapped_x = x * CONTROL_PANEL_WIDTH / max(win_width, 1)
        mapped_y = y * CONTROL_PANEL_HEIGHT / max(win_height, 1)
        return mapped_x, mapped_y

    def _set_slider_from_x(self, slider: SliderControl, x: float) -> None:
        rail_x = slider.x + 190
        rail_w = slider.width - 200
        norm = min(max((x - rail_x - 8) / max(rail_w - 16, 1.0), 0.0), 1.0)
        value = slider.minimum + norm * (slider.maximum - slider.minimum)
        self._set_control(slider.field_name, value)

    def _set_control(self, field_name: str, value: float) -> None:
        with self.state.lock:
            value = float(value)
            if field_name == "disk_inner":
                value = min(value, self.state.controls.disk_outer - 0.4)
            elif field_name == "disk_outer":
                value = max(value, self.state.controls.disk_inner + 0.4)
            setattr(self.state.controls, field_name, value)

    def _handle_button_press(self, x: float, y: float) -> None:
        if self._rect_contains(skia.Rect.MakeXYWH(324, 350, 134, 36), x, y):
            with self.state.lock:
                self.state.controls.camera_distance = 11.0
                self.state.controls.camera_height = 2.25
                self.state.controls.camera_orbit = 0.0
        elif self._rect_contains(skia.Rect.MakeXYWH(480, 350, 150, 36), x, y):
            self._request_benchmark()
        elif self._rect_contains(skia.Rect.MakeXYWH(652, 350, 84, 36), x, y):
            self._request_quit()

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
    parser.add_argument(
        "--tier-duration", type=float, default=None, help="Override benchmark tier duration for testing."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = BlackHoleBenchmark(args.width, args.height, smoke_test=args.smoke_test, tier_duration=args.tier_duration)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
