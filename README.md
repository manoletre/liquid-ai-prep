# Liquid AI prep

Personal lesson series for preparing for applied post-training roles (Liquid AI). Live at https://manoletre.github.io/liquid-ai-prep/

| # | Lesson |
|---|---|
| 01 | [Policy gradient → PPO → GRPO](lessons/01-policy-gradient-to-ppo/) |
| 02 | [Explore vs exploit in GRPO](lessons/02-explore-exploit-grpo/) |
| 03 | [Evaluation metrics, crystal clear](lessons/03-evaluation-metrics/) |

## How the lessons work

Each lesson is one page. Chapters are shown full-screen, one at a time, and every chapter follows the same flow: **question** (write a prediction; the video isn't loaded yet) → **learn** (video with click-to-play steps that auto-pause) → **compare** (your answer next to the model answer, a Got it / Partly / Missed rating, and a "what was unclear" note) → **pen-and-paper practice**. A lesson can end with an interview drill. It always ends with "Revisit your answers". Answers are stored in the browser's localStorage under `lap:<lesson id>:<question id>:answer|rating|note|revealed`.

## Creating a new lesson (player v4)

The working copy lives in `/workspace/ml-tutor` (`manim/`, `build/`). `source/` here mirrors it.

1. **Scenes.** Write `manim/l4_body.py` with one `StepScene` per chapter. Wrap each step in `self.begin("key")` … `self.end()`; `end()` adds the 1.8 s hold and records timestamps to `manim/steps/<Scene>.json`. Concatenate: `cat v3_head.py v3_body.py l4_body.py > l4_all.py`.
2. **Render.** Draft: `../.venv/bin/manim -ql --disable_caching l4_all.py <Scene>`, then check the step-end frames with `python3 endframes.py media/videos/l4_all/480p15 /tmp/ef <Scene,...>`. Final: `../.venv/bin/manim -r 1920,1080 --fps 30 --disable_caching l4_all.py <Scene>`.
3. **Content.** In `build/content.py`, add an `L4` dict like `L3`: `id`, `slug`, `title`, `h1`, `sub`, `key` (symbol chips; add new symbols/macros to `SYM_ALL` / `MACROS` in `build.py`), and `chapters`. Each chapter has `scene`, `src` (`"l4_all"`), `file`, `title`, `short`, `hook` (`{id, q, model}`, the prediction), `steps` (`(key, title, html)`; the keys must match the scene's `begin()` keys), and `check` (`{id, q, model}`, the practice). Optionally add `drill` (a list of `{id, q, model}`) and `summary`. Register it in `LESSONS`. Keep question ids stable once published: they are the localStorage keys.
4. **Build.** `python3 build/build.py 4` encodes the videos and posters and writes `lessons/<slug>/index.html` from `build/template.html`.
5. **Verify.** Serve the repo with `python -m RangeHTTPServer 8765` (it must support seeking) and run `python test_lesson.py 4` after adding the slug to its `SLUG` map. It checks: no video loads or plays before its chapter starts, segments pause at step ends, long answers fit, answers persist across reload, formulas render, and there are no console errors.
6. **Publish.** Add a card to `index.html` and a row above, then commit and push. GitHub Pages serves `main`.

Animations are rendered with Manim (`source/manim/lesson_scenes.py` for lessons 1–2, `source/manim/lesson03_scenes.py` for lesson 3). Each chapter records its step timestamps, and `source/build` generates the lesson pages.
