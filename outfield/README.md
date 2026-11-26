# outfield/

Code for the outfield players (O1 / O2, see `docs/roles/outfield.md`).

| Path | What |
|---|---|
| `config.py` | Engine settings shared by scripts, the env wrapper and training (action set `v4`, 0.1 s steps, magnet off, ...) |
| `env/` | **The env wrapper** everyone trains on: `OutfieldEnv`, `VecEnv`. Interface: [`docs/env_wrapper.md`](../docs/env_wrapper.md). Speed: `python -m outfield.env.bench` |
| `scripted/` | Hand-written plays run against the game's AI |
| `scripted/scene.py` | `outfield_3v3`: three outfield players and a keeper a side, one attack per episode |
| `scripted/play.py` | `TrianglePlay`: the scripted attack (carry, pass when closed down, shoot in range; the other two take the free lanes) |
| `scripted/run.py` | Runs the play and the game's AI from the same seeds and compares them |
| `scripted/overlay.py` | Draws each player's command on recorded frames (`--record`) |

## Run

From the repo root, with the `grf` env:

```bash
# Numbers: 20 seeds, our side played by the script and by the game's AI
PYTHONPATH=. python -m outfield.scripted.run

# Watch it (rendered games play differently from headless ones; don't compare them)
PYTHONPATH=. python -m outfield.scripted.run --watch --seeds 1 2 3
PYTHONPATH=. python -m outfield.scripted.run --watch --mode ai --seeds 1 2 3

# Video with every player's command on screen (out/outfield/videos/)
PYTHONPATH=. python -m outfield.scripted.run --record --seeds 12 1 3
```

Results are saved to `out/outfield/scripted_3v3.json`.

## Seeing the commands

The live game window has no hook for our text, so commands show up in two places:

- `--watch` prints what each player is doing in the terminal whenever it changes, e.g.
  `7.8 s  #1 support: left lane  |  #2 support: right lane  |  #3 pass to #2: press (power 0.24)`.
- `--record` draws it on every frame. Bottom left: each player's intent and the button actually sent
  (`idle` means "keep holding what is held": directions and sprint are sticky). Top right: clock and
  who has the ball. Bottom right: a mini-pitch with our players numbered. "game AI: ..." means the
  player was handed to the game's AI that step (`builtin_ai`).

## Outcomes

Each episode is one attack, ending on the first of:

| Outcome | Meaning |
|---|---|
| goal | we scored |
| keeper | their keeper ended up with the ball (a save or a catch) |
| lost | one of their outfield players won the ball |
| out | the ball went out of play |
| timeout | 30 s passed with no other result |
