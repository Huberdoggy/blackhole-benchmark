# Black Hole Benchmark

High-fidelity real-time relativistic black hole raytracer and hardware benchmark. The application uses pygame to create an OpenGL 4.6 core-profile context, ModernGL to drive a fullscreen GLSL fragment raymarcher, and Dear PyGui for a separate diagnostics/control viewport.

The render path models inverse-cube gravitational light deflection around a dynamic Schwarzschild radius, a configurable accretion disk on the horizon plane, Doppler-style blue/red disk beaming, a lensed starfield, and staged SwarmUI visual assets for the space canvas, accretion disk, singularity halo, and HUD art direction.

## See It In Action

![Distance shot](./src/assets/screenshots/static_view2.png)
![Front view of control sliders and midrange singularity](./src/assets/screenshots/static_view1.png)
![Close up, off center underneath, manipulated ring proximity](./src/assets/screenshots/static_view3.png)
![Personal benchmark results](./src/assets/screenshots/bench_results.png)

## Capabilities

- GPU raymarching in a GLSL fragment shader through ModernGL.
- Dynamic controls for mass, disk inner radius, disk outer radius, camera distance, and camera height.
- Programmatic accretion disk with radial temperature falloff and directional Doppler tinting.
- Warped starfield that responds to gravitational lensing around the event horizon.
- Animated singularity halo using the staged `high_fidel_singularity.png` texture as live shader input.
- Dear PyGui diagnostics viewport with FPS, frame time, resolution, active mode, analog gauges, and physical slider styling based on `vivid_control_panel.png`.
- Automated benchmark routine for 720p, 1080p, and 4K tiers with trimmed frame-time statistics.
- Weighted hardware score:
  `Score = (Avg_720p_FPS * 1.0) + (Avg_1080p_FPS * 2.5) + (Avg_4K_FPS * 6.0)`.
- SwarmUI orchestration script for staging visual assets from `render_backlog.md`.

## Developer Setup

Prerequisites:

- Python 3.11 in the project pyenv virtualenv.
- NVIDIA offload helper at `/usr/local/bin/nv-run.sh`.
- OpenGL 4.6-capable driver and GPU.
- Python packages: `moderngl`, `pygame`, `dearpygui`, `numpy`, `websocket-client`, `black`, and `flake8`.
- Optional for asset generation: SwarmUI running at `http://localhost:7801` with `z_image_turbo-Q6_K.gguf` available.

Run the app:

```bash
/usr/local/bin/nv-run.sh python benchmark.py
```

Run a short smoke test:

```bash
/usr/local/bin/nv-run.sh python benchmark.py --smoke-test 3
```

Run formatter and linter checks:

```bash
python -m black benchmark.py orchestrator.py
python -m flake8 benchmark.py orchestrator.py
```

Generate or stage SwarmUI assets from the backlog:

```bash
python orchestrator.py
```

Use `python orchestrator.py --dry-run` to validate backlog parsing and write a manifest without sending render requests.
