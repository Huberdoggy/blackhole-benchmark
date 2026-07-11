# SwarmUI API Reference for Physics Sandbox

This project stages SwarmUI-generated bitmap assets into `src/assets/staging/` for use by `benchmark.py`. The reliable path so far is a single-image websocket request at 512-class dimensions, using `z_image_turbo-Q6_K.gguf`, CFG `1.0`, and `8` steps.

## Known Working Settings

- API root: `http://localhost:7801`
- Session route: `POST /API/GetNewSession`
- Generation route: websocket `/API/GenerateText2ImageWS`
- Model ID exposed to the API: `z_image_turbo-Q6_K.gguf`
- Model path on disk: `/home/huberdoggy/shared_os/swarm_models/diffusion_models/z_image_turbo-Q6_K.gguf`
- CFG scale: `1.0`
- Steps: `8`
- Images per queued render: `1`
- Output handoff directory: `src/assets/staging/`

The websocket route proved more reliable than blocking HTTP generation for long renders. The orchestrator opens a session, sends a JSON payload to `/API/GenerateText2ImageWS`, listens for progress/image events, handles `invalid_session_id` by requesting a fresh session, and maps returned `View/local/raw/...` image references back to SwarmUI's output tree before copying to staging.

## Reliable Asset Dimensions

The current staged assets were generated successfully at smaller sizes suitable for a 6 GB VRAM card:

| Asset | Size |
| --- | ---: |
| `deep_space_canvas.png` | 512x512 |
| `vivid_control_panel.png` | 512x384 |
| `photoreal_accretion_disk.png` | 512x512 |
| `high_fidel_singularity.png` | 512x512 |

Larger 768-to-1024-class attempts previously stalled or entered backend waiting states on this hardware. Queueing one image at a time is the safer default.

## Payload Shape

The working payload fields are:

```json
{
  "images": 1,
  "prompt": "Physics-grounded asset prompt...",
  "negativeprompt": "blurry, low resolution, cartoon, anime, flat colors, compression artifacts",
  "model": "z_image_turbo-Q6_K.gguf",
  "width": 512,
  "height": 512,
  "cfgscale": 1.0,
  "steps": 8,
  "seed": 123456,
  "extra_metadata": "{\"handoff\":\"physics_sandbox/src/assets/staging\"}",
  "session_id": "from GetNewSession"
}
```

The API image event may return either a string image reference or an object containing an `image` property. `orchestrator.py` handles both forms.

## VAE and Qwen Text Model Experiment

The requested hypothesis was whether explicitly parameterizing these files would improve 1024x1024 success:

- `/home/huberdoggy/shared_os/swarm_models/VAE/Flux/UltraFlux-vae.safetensors`
- `/home/huberdoggy/shared_os/swarm_models/text_encoders/Qwen3-4B-UD-Q6_K_XL.gguf`

On July 11, 2026, `ListT2IParams` was queried with a valid session id. The exact parameter nomenclature for this server is:

- VAE field id: `vae`
- VAE value for UltraFlux: `Flux/UltraFlux-vae`
- Qwen field id: `qwenmodel`
- Qwen value used: `Qwen3-4B-UD-Q6_K_XL.gguf`

There is no exposed field named `text_encoder` in this server's text-to-image parameter list. Use `qwenmodel` for the Qwen add-on instead.

`turbo-renders` designed the following smoke test. For this hardware, it was run as one 1024 image per variant, sequentially, rather than queueing four images at once.

```json
{
  "model": "z_image_turbo-Q6_K.gguf",
  "vae": "Flux/UltraFlux-vae",
  "qwenmodel": "Qwen3-4B-UD-Q6_K_XL.gguf",
  "width": 1024,
  "height": 1024,
  "cfgscale": 1.0,
  "steps": 8,
  "images": 1,
  "seed": 424242,
  "prompt": "Centered relativistic black hole singularity, luminous rotational photon halo, smooth gravitational lensing arcs, dense pin-sharp starfield, blue-white hot rim with amber plasma undertones, scientific cinematic realism, crisp high-contrast 1024 square benchmark asset."
}
```

Results:

- Automatic add-ons succeeded and staged `src/assets/staging/swarm_smoke_auto_1024.png`.
- Explicit `vae` plus `qwenmodel` succeeded and staged `src/assets/staging/swarm_smoke_explicit_1024.png`.
- Both files were verified as valid 1024x1024 PNGs.
- The two outputs were visually very similar: centered black hole, circular horizon, coherent luminous halo, and sharp starfield.

Conclusion: explicitly setting `vae=Flux/UltraFlux-vae` and `qwenmodel=Qwen3-4B-UD-Q6_K_XL.gguf` is accepted by the API and can successfully produce 1024x1024 output. This single controlled pass did not show a clear advantage over automatic settings, because the automatic variant also succeeded and produced nearly the same visual result. The main limiting factor still appears to be GPU workload and queue behavior rather than missing VAE/text-model nomenclature.

Success criteria:

- The image completes without backend, model, VAE, or Qwen errors.
- Each output is a valid 1024x1024 image.
- The event horizon remains centered and circular enough for shader use.
- The halo has coherent rotational arcs rather than smeared fog or random streak noise.
- The starfield remains sharp outside the halo.
- No output is fully black, washed out, checkerboarded, corrupted, or malformed.

Failure criteria:

- Any generation crashes, hangs, returns blank or corrupt output, or drops below requested size.
- The halo collapses into flat glow with no readable directionality.
- The image has severe color clipping, broken contrast, or unusable texture structure.

Run the reproducible smoke test with:

```bash
python orchestrator.py --swarm-smoke-test
```

Use `--dry-run` to write the smoke-test manifest without sending generation requests.

## Operational Notes

- Keep `images` at `1` for normal backlog work on this GPU.
- Prefer square 512 textures for shader-sampled space, disk, and singularity assets.
- Use `512x384` for the panel reference asset; the application uses it for visual direction rather than as a rigid UI bitmap.
- Preserve `extra_metadata` handoff fields so generated images can be traced back to backlog components.
- Do not mark `render_backlog.md` tasks complete until staged renders have been manually reviewed.
