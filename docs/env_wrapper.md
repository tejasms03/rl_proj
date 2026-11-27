# The env wrapper: interface contract (v1)

*For O1 (outfield training), G (keeper), M1 and M2 (manager). Owner: O2. Code: `outfield/env/`. Written 2026-11-24.*

The wrapper sits between the game and every network. Raw game data goes in, and each player gets 960 numbers out. Five action choices per player go in, and one game command per player comes out. **Build against the shapes on this page. If one has to change, it changes here first.**

Status: everything below works and is tested, except where it says **provisional** (the instruction layout) or **to be decided** (rewards).

---

## 1. Quick start

```python
from outfield.env import OutfieldEnv, VecEnv, random_actions

env = OutfieldEnv("11_vs_11_easy_stochastic", seed=1)   # any GRF scenario, lab_* scene or outfield_3v3
obs = env.reset()
obs, rewards, terminated, truncated, info = env.step(actions)
env.close()

venv = VecEnv(8, scenario="11_vs_11_easy_stochastic", seeds=range(8))  # one game per process
```

Run from the repo root with the `grf` env and `PYTHONPATH=.`. Tests: `pytest tests/test_env_*.py`. Speed: `python -m outfield.env.bench`.

## 2. What the wrapper decides for you

| Setting | Value | Why |
|---|---|---|
| Our side | Always the left team, attacking +x | v1 plays the game's AI only; mirroring for self-play comes later behind the same interface |
| Opponent | The game's AI | v1 scope |
| Players controlled | All of ours. Slot *i* is player *i* of the scenario (0 = keeper) | The engine only fixes which agent drives which player when every player is controlled |
| Keeper (slot 0) | `keeper="builtin"` (default): the game's AI. `keeper="agent"`: your network drives him | G plugs in without a wrapper change |
| Engine settings | `outfield/config.py`: action set `v4`, 0.1 s steps, v1.0 physics, kick power on, magnet off, `keeper_auto_save` on, `skill_stats` on | CONTEXT §3 and PLAN.md v1.2 |
| AI plays the ball | `ai_on_ball=True` (default): our player *on* the ball is handed to the game's AI | Shortcut rank 1 (CONTEXT §8). Loose balls stay the policy's job: a player must reach the ball before the AI takes over |

## 3. Observation

```
obs["actor"]    float32[11, 960]   per player: 869 base + 91 instruction block
obs["state"]    float32[2016]      critic only: global state
obs["present"]  bool[11]           False = padded slot (drills with fewer players)
obs["masks"]    {"move": bool[11,17], "speed": bool[11,3], "ball": bool[11,8], "defend": bool[11,5]}
```

**The 869 base numbers** follow `docs/observation_spec.md`: self 58, ball 13, teammates 10 × 47, opponents 11 × 28, context 20. The exact column of every block is in `outfield/env/observation.py` (`ACTOR_LAYOUT`, `SELF`, `BALL`, `MATE`, `OPP`, `CONTEXT`). Choices the spec left open:

- **Frame.** Metres with +y up the screen (`y_m = y × −83.6`), then the spec's scaling.
- **Teammate rows** are the other 10 slots in slot order, with self skipped. Absent slots are all zeros, and so is an absent player's own row.
- **Ranges.** Everything stays within ±1.5, except relative positions and distances. Those span the whole pitch, up to 108 m ÷ 55 ≈ 2.
- **Possession.** The ball-owner one-hot (my team / theirs / nobody) uses the 1 s possession memory. Per-player "has the ball" uses the engine's raw owner, so it marks only the player actually touching it.
- **Held buttons.** With analog movement the 8 direction entries are almost always 0. The engine only reports exact compass directions there. Sprint, dribble, pressure, team pressure and keeper rush are reported correctly.

**The critic state (2016)** is laid out in `observation.STATE_GLOBAL`, `STATE_PLAYER` and `state_layout()`:
- global (25): ball position, velocity and spin, the owner, score, clock, game mode, pitch
- every player of both teams in slot order (22 × 45), with presence and full attributes
- every player's 91-number block (11 × 91)

How to combine it with the actor rows is O1's choice.

## 4. Actions

```
actions = {"move": int[11], "speed": int[11], "ball": int[11], "defend": int[11], "power": float[11]}
```

| Head | Size | Values |
|---|---|---|
| move | 17 | 0 = stand still; *k* = 1…16 is the direction (k − 1) × 22.5°, counter-clockwise from "toward their goal" (+x), +y up the screen |
| speed | 3 | 0 walk, 1 sprint, 2 close control |
| ball | 8 | 0 none, 1 short pass, 2 high pass, 3 long pass, 4 shot, 5 curl shot, 6 chip, 7 through ball |
| defend | 5 | 0 none, 1 pressure, 2 team pressure, 3 sliding, 4 keeper rush |
| power | [0, 1] | kick power, used only when a kick is pressed |

**How choices become commands:**
- **Movement** goes every step as the engine's analog direction. It was measured within about 2° of the target in all 16 directions, and it still works after a hand-off to the game's AI and back.
- **One button per step.** The priority is: a kick, then sliding, then one change of held buttons (sprint, close control, pressure, team pressure, keeper rush), else nothing. Held buttons change only when your choice changes.
- **Masks** (pass them to your policy's softmax; a masked choice is sent as 0):
  - Kicks are allowed only with the ball within 2 m and below 2.5 m, or with a loose ball whose path comes that close within 1.5 s.
  - Defensive actions are allowed only while they have the ball (possession memory).
  - **The keeper (slot 0) has a smaller set** (agreed with G, 2026-11-24). He walks or sprints, distributes with short, high or long passes, and saves through `keeper_auto_save`:
    - speed: no close control (2)
    - ball: no shot, curl shot, chip or through ball (4–7)
    - defend: none at all, so no pressure (1), team pressure (2), sliding (3) or keeper rush (4). Keeper rush tells the *AI* keeper to come out, so it is an outfield button.
  - Absent slots, and slot 0 with `keeper="builtin"`, allow only choice 0.

**`info["overridden"]: bool[11]`** is True when your choice for that player did not reach the game. **Leave those steps out of the policy loss.** It happens when:
- the keeper is left to the game's AI,
- the player is on the ball and `ai_on_ball=True`, or
- a kick is winding up: after a kick is pressed the player keeps its direction until the ball leaves his foot, or for up to 1 s. Handing him to the AI earlier would cancel the kick (CONTEXT §7.2).

**`info["kicks"]`** lists kick events: `{slot, button, power, step, event}`, where `event` is `pressed`, `left_foot` or `expired`. It is the raw material for any shot-based reward term.

## 5. Instruction block (91): PROVISIONAL

The sections and sizes come from `docs/training_plan.md` §2.1. **The order inside each section and the scaling are a proposal**, implemented in `outfield/env/instructions.py`. They need M1's and G's agreement, and only that file changes when the layout is frozen.

| Columns | Block | Contents |
|---|---|---|
| 0–13 | I1 role | role family one-hot (GK, CB, FB, DM, CM, AM, W, ST, none), side one-hot (left, centre, right), slot ÷ 10, taker flag |
| 14–34 | I2 duties | defending D0–D8 (9), attacking A0–A7 (8), transition T0–T3 (4), each one-hot |
| 35–63 | I3 targets | opponent slot one-hot (11 + none), teammate slot one-hot (11 + none), anchor x ÷ 55, anchor y ÷ 36 (metres, +y up), radius ÷ 10 m, line height 0–1, press intensity 0–1 |
| 64–69 | I4 dials | line_height, press_trigger, width, tempo, risk, compactness |
| 70–84 | I5 set play | restart type (6), set-play duty (5), zone x ÷ 55, zone y ÷ 36, arrival time ÷ 10 s, time since restart ÷ 10 s (all zero in v1) |
| 85–90 | M memory | **written by the wrapper, never by the manager:** last 3 button actions as class ÷ 3 (none 0, pass ⅓, shot ⅔, defend 1), steps since my last touch ÷ 100, steps since possession changed ÷ 100 (both capped at 1), instruction changed in the last 5 s (0/1) |

**Where instructions come from.** The manager side writes 85 numbers per player, columns 0–84:
- `env.step(actions, instructions=float32[11, 85])` uses yours for that step.
- With `instructions=None`, the built-in sampler supplies them. It is on by default, `sampler=None` turns it off, and it is seeded by the env's seed.

**What the sampler draws.** It re-draws the whole team every 30–60 s of game time:
- **Random draws use v1 duties only.** Defending is D1, D2, D4, D5 or D7. Attacking is A1–A4. Transition is T0 or T2. The keeper gets D0, D1 (hold zone between the posts: anchor at the centre of our goal line, radius 3.7 m) or D8 (with a line height), and A0, A1 (a spot near our goal) or A2 (available for a back pass).
- **Targets match the duty.** Mark gets an opponent (nearer ones are likelier). Cover gets a teammate. Hold zone and hold shape get an anchor near the player's kick-off spot.
- **20% of draws are presets:** high press, low block, possession, counter, wing play or false nine, built from v1 duties. Where a preset needs a duty that isn't in v1, the player gets the closest v1 duty (false nine: support instead of drop to receive).

**Open items for M1 and G:**
1. **Combination grammar.** Which duties may go together isn't written down anywhere. Until it is, draws are independent per player.
2. **A3 width has no "side" field in I3.** Proposal: the anchor's y gives the side (±30 m).
3. ~~Keeper duties.~~ **Settled with G (2026-11-24):** defending D0 none, D1 hold zone (between the posts: the centre of our goal line, 3.7 m; added 2026-11-26), D8 sweep; attacking A0 none, A1 hold shape (a spot near our goal), A2 support (available for a back pass). Short vs long distribution stays learned through the risk and tempo dials; it is not a duty. Set-piece positions come with I5 later.
4. **The memory block.** "Last 3 actions" is squeezed into 3 numbers (class ÷ 3) so the block stays at 6, as §2.1 says.

## 6. Episodes, rewards, info

- **`terminated`:** a goal or another end that a drill's rules define (out of play, lost the ball), or the final whistle of an `11_vs_11_*` match.
- **`truncated`:** a drill's time limit, so returns can be bootstrapped.
- **`info["outcome"]`** at the end: `goal_for`, `goal_against`, `lost_ball`, `out_of_play`, `time`, `other`.
- **`rewards`** is a dict of `float32[11]`. **To be decided:** the reward terms need a team discussion first. v1 has only a placeholder, `score` (+1 for a goal for, −1 against, for every present player), so the loop and a PPO smoke test can run. New terms will be new keys, so the signature won't change.
- **`info["duty_ok"]`** is NaN until the duty checks land (week 2). `info["plan"]` is the name of the current instruction draw (`random`, a preset name, or `given`). `info["raw"]` holds the raw observations, for debugging.

**`reset(seed=s)` replays game *s* exactly** on a fresh engine. `reset()` with no seed starts the next game on the same engine. A finished game in `VecEnv` resets by itself. Its last observation is in `infos[k]["final_obs"]`.

## 7. Speed

Measured with `python -m outfield.env.bench` on a 10-core Mac (4 performance cores), with random valid actions:

| Scenario | 1 game | 8 games in parallel (total) |
|---|---|---|
| `11_vs_11_easy_stochastic` | ~310 steps/s | ~1,180 steps/s |
| `outfield_3v3` | ~560 steps/s | ~1,940 steps/s |

**Where the time goes.** The wrapper's own work is about 10% of a step. The rest is the engine plus the game's Python layer.

**What this machine can and can't tell you.** Scaling stops at the performance-core count, so it says little about a many-core cloud machine. Rerun the benchmark there before planning hours (CONTEXT §9).

## 8. Not in v1

- Our team on the right (mirroring).
- Opponents other than the game's AI (the scripted team, self-play).
- Reward terms.
- Duty checks (week 2: they will port `grf_tools/duty_lab/duties.py` with the ×10 velocity rule).
