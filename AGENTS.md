
# Primary Objective
You are the lead developer and supervisor asigned with overseeing the logical implementations and milestone sign-offs for a high-fidelity, real-time **Relativistic Black Hole Raytracer & Performance Benchmark**.

**IMPORTANT:** Whenever a task requires or refers to the artistic style of a programmatic asset (new or existing), you MUST delegate the work by explicitly spawning your `turbo-renders` teammate.

---

## 1. Core Physics & Pipeline
* **API**: Build a GPU-accelerated
ModernGL workflow using an OpenGL 4.6 core profile.
* **Math**: Implement a 3D
raymarching engine executed entirely within a GLSL fragment shader.
* **Gravitational Lensing**: Program the fragment shader to calculate the Schwarzschild radius
based on a dynamic mass variable. For every pixel ray, approximate Einstein's light deflection by
modifying the ray direction vector dynamically at each integration step using an inverse-cube
distance relationship to the singularity.

## 2. Visual Fidelity Elements
* **Accretion Disk**: Render a flat geometric plasma disk around the horizon plane ($Y=0$)
bounded by configurable inner and outer radii. Apply an exponential decay temperature profile to
mimic realistic thermal emissions.
* **Relativistic Effects**: Calculate a Doppler beaming vector approximation. Shift the intensity and
color spectrum of the disk toward high-energy blue on the side rotating toward the camera
viewpoint, and dim it toward low-energy deep red on the retreating side.
* **Warped Starfield**: Implement a procedural, high-frequency pseudo-random noise function in
the shader to generate a background starfield that undergoes gravitational lensing distortion near
the event horizon.

## 3. Diagnostic HUD Integration
* **Asynchronous Overlay**: Integrate Dear PyGui to display engine diagnostics in an independent
viewport window.
* **Real-time Diagnostics**: Track and output precise performance metrics, specifically frames per
second (FPS) and precise frame processing times in milliseconds.
* **Control Uniforms**: Bind interactive interface sliders to dynamically update the shader's uniform
variables for singularity mass, inner/outer disk boundaries, and camera proximity vectors.

## 4. Automated Benchmark Mode Execution
* **Automated Stress Test**: Implement a dedicated "Run Benchmark" routine that overrides active user camera inputs and cycles the simulation through three fixed testing tiers: 10 seconds at 720p,
10 seconds at 1080p, and 10 seconds at 4K resolution.
* **Data Logging**: Collect frame-time metrics continuously during the routine. Filter out the top
and bottom 1% anomalies to ensure statistical consistency.
* **Hardware Scoring**: Compute a final weighted performance score utilizing the equation: `Score
= (Avg_720p_FPS * 1.0) + (Avg_1080p_FPS * 2.5) + (Avg_4K_FPS * 6.0)`. Display this final score in
a modal dialog box upon stress test completion.

---

## Operational Playbook
1. Environment Check: Probe the local system environment (pyenv virtualenv preconfigured), verify available dependencies; install missing.
2. Script Generation: Write the monolithic application to `benchmark.py`.
3. Multi-Worker Orchestration & Handoff Script: Write orchestrator.py
4. Read pending asset definitions from [render_backlog.md](./render_backlog.md).
5. Generate payloads targeting `z_image_turbo-Q6_K.gguf` utilizing physics-grounded
prompts.
6. Main Thread Handoff: Copy completed renders to a temporary staging area
(`./src/assets/staging/`).

**PAUSE HERE**: Request user manually review staged files, incorporate only when given the all-clear.

7. Update backlog tasks; denote passing renders as complete via checkmarks.

8. Hard Verification:
    - Attempt execution immediately. Prefix script invocation with ```/usr/local/bin/nv-run.sh```; ensures offloading to discrete graphics. 
    - If a compilation or runtime error occurs, parse the `stderr` logs, refactor the faulty script sections, and re-execute. Do not halt execution until a persistent canvas renders successfully.

---

## No-Touch Zones:
Refrain from making any modifications to the exisiting contents of:
- The mentioned GPU offload script `nv-run.sh`
- [nuke_and_run.sh](nuke_and_run.sh), used for cold starts