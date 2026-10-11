# Autonomous Kinesiology Graph

**Type an exercise name; get a biomechanically described record and a technical animation in your database.**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-agent%20workflow-1C3C3C)
![Gemini](https://img.shields.io/badge/Gemini-structured%20output-4285F4?logo=googlegemini&logoColor=white)
![Manim](https://img.shields.io/badge/Manim-rendering-2D6A4F)
![Supabase](https://img.shields.io/badge/Supabase-Postgres%20%2B%20Storage-3ECF8E?logo=supabase&logoColor=white)
![Tests](https://img.shields.io/badge/tests-100%20passing-52B788)

<p align="center">
  <img src="docs/demo.gif" alt="Generated technical animation of a dumbbell Romanian deadlift" width="800">
</p>

<p align="center"><sub>Generated end to end by the pipeline from the words "peso muerto rumano" (dumbbell variant): coaching steps with the active one highlighted, muscle tension in green, force arrows and live joint angles.</sub></p>

---

## What it does

- **Understands a loose request.** Type an exercise in Spanish (typos and slang welcome). The agent proposes its real variants (barbell, dumbbell, single-leg, cable, etc.), shows which ones are already in the database, and lets you pick several.
- **Writes the exercise record.** Force, mechanic, equipment and category; primary and secondary muscles; setup, execution steps, breathing and common mistakes. Everything is validated against a typed schema.
- **Animates the technique.** It produces a 1280×720 technical video and a still: a side-view anatomical figure with tension-coloured muscles, force vectors, joint angles and step-by-step coaching.
- **Loads it idempotently.** Media goes to Supabase Storage and the record to Postgres. Each chosen variant is processed in parallel.

## How it works

```
"peso muerto rumano"
        │
        ▼
 Variant resolver (Gemini) ──► DB lookup ──► pick variants in the terminal
        │
        ▼   one LangGraph run per variant, in parallel
 ┌──────────────────────────────────────────────────────────────┐
 │ content ─► animation spec ─► Manim render ─► Storage ─► Postgres │
 │ (Gemini)   (Gemini, checked)   (local, free)                     │
 └──────────────────────────────────────────────────────────────┘
```

The key idea: **the LLM does not draw, it describes.** Gemini returns a validated *animation spec* for one repetition: keyframes of joint angles, muscle tension per keyframe, coaching steps, force arrows and the joints whose angles matter. A deterministic renderer built on Manim turns that spec into the video. The output is consistent, technically readable, costs nothing to render, and works for any exercise.

## Engineering highlights

- **Self-healing LLM loop.** Specs go through two checks. Pydantic validators check structure. A geometry engine then checks anatomy: knees bend forward, joints stay within human limits, the load stays over mid-foot in hinges and squats, and the free leg clears the floor. Each violation is sent back to Gemini as a concrete fix ("the load is 0.6 units in front of mid-foot; push the hips further back"), with up to three attempts.
- **Deterministic rendering instead of generative video.** The project started with Imagen and Veo. They were dropped because they could not draw accurate force lines, angles or text, they gave different results on every run, and they were slow and paid. A spec plus a renderer fixed all three.
- **Pure, tested geometry.** Forward kinematics with fixed bone lengths, grounding, interpolation and collision-aware label placement live in plain Python (`skeleton.py`, `layout.py`). That code is unit-tested without Manim, and the scene only draws.
- **Concurrency where it pays.** Variants run in a thread pool. CPU-bound Manim renders are capped by a semaphore. Uploads are serialised over HTTP/1.1 with retries, after real parallel HTTP/2 uploads failed with dropped connections.
- **Data integrity.**
  - One deterministic slug rule is shared by the DB lookup, the upsert key and the Storage path.
  - Overwriting an existing exercise needs explicit confirmation.
  - The upsert never replaces a good media URL with an empty one.
  - Every upload is verified by comparing the remote object size with the local file.
- **Failure isolation.** Graph nodes never raise. They report errors in the state and the graph decides what to do next. One failing variant never stops the others, and the summary says exactly what happened to each one.
- **Tested offline.** About 100 tests run with every external client (Gemini, Postgres, Storage, Manim) faked. Opt-in tests run real renders.

## Tech stack

| Area | Tools |
|---|---|
| Agent orchestration | LangGraph (`StateGraph` with conditional retry edges) |
| LLM | Google Gemini via `langchain-google-genai`, Pydantic structured output |
| Rendering | Manim (Cairo), NumPy, PyAV for still extraction |
| Storage & data | Supabase Postgres (`psycopg` 3) and Supabase Storage |
| Quality | pytest, monkeypatched fakes, typed models (Pydantic v2) |

## Project structure

```
main.py                 Interactive CLI: request → variants → concurrent generation → summary
src/
  graph.py              LangGraph wiring and retry routing
  domain/               Pydantic models: exercise record, animation spec, graph state
  nodes/                One node per step: content, spec, render, storage, database
  rendering/            Geometry (skeleton, layout), anatomical body, Manim scene, render runner
  resolution/           Variant resolver (natural language → concrete variants)
  pipeline/             Concurrent execution of the selected variants
  cli/                  Terminal dialogue (Spanish)
  infrastructure/db/    Supabase Storage client and read-only repository
samples/                Hand-tuned animation specs (prompt examples and test fixtures)
render_spec.py          Render any spec locally without API keys
upload_media.py         Re-upload rendered media for stored exercises
init_db.py              Database schema
tests/                  Offline test suite
```

## Getting started

**Prerequisites:** Python 3.12, a Gemini API key, and a Supabase project with two **public** Storage buckets: `exercise_videos` and `exercise_images`.

```bash
git clone https://github.com/JuanXiRP/autonomous-kinesiology-graph.git
cd autonomous-kinesiology-graph
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # then fill in your keys
python init_db.py                 # creates the `exercises` table
python main.py "peso muerto rumano"
```

Each generated variant costs two Gemini calls, plus one call for the variant list, and about 4 minutes of local rendering.

**Try the renderer without any keys:**

```bash
python render_spec.py samples/single_leg_romanian_deadlift_spec.json dumbbell "Single-Leg Romanian Deadlift (Dumbbell)"
# add --frame 3.5 to render a single frame in ~10 seconds
```

The output goes to `media/previews/`.

## Example session

The terminal speaks Spanish; the data is stored in English. Output trimmed.

```text
¿Qué ejercicio quieres generar? peso muerto rumano

  1) Peso muerto rumano con barra — romanian_deadlift_barbell  [ya en BD, 08/10/2026]
  2) Peso muerto rumano con mancuernas — romanian_deadlift_dumbbell
  3) Peso muerto rumano a una pierna con mancuerna — single-leg_romanian_deadlift_dumbbell
  4) Peso muerto rumano con pesa rusa — romanian_deadlift_kettlebell
  ...

¿Cuáles quieres generar? (ej. 1,3 · 1-6 · todos · Enter para cancelar): 1,2,3
  'Peso muerto rumano con barra' ya existe. ¿Regenerarlo y sobrescribirlo? (s/N): N
¿Generar? (s/N): s

romanian_deadlift_dumbbell | 🎞️ [Render] Rendering animation with Manim...
single-leg_romanian_deadlift_dumbbell | 🎞️ [Render] Rendering animation with Manim...
...
📋 Resumen
  <per variant: creado / actualizado / guardado sin vídeo / falló, with its video and image URLs>
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest            # offline suite, no network
pytest -m slow    # real Manim renders of the sample specs
```

## Limitations & roadmap

- **Schematic body.** The figure is a vector illustration. A photoreal restyle of the rendered keyframes is a possible next step.
- **Stances.** Split stances (lunges, split squats) are not representable yet. Single-leg work uses a free leg.
- **Muscle names.** Names coming from the LLM are not normalised to a fixed vocabulary yet.
- **Render time.** About 4 minutes per variant on a laptop CPU. Renders run two at a time.
- **Companion app.** The vocabulary still needs mapping to the free-exercise-db format used by the companion fitness app.

## License

[MIT](LICENSE)

## Author

**JuanXiRP** · [github.com/JuanXiRP](https://github.com/JuanXiRP)
