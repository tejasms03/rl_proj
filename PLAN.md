# GRF+ : patched Google Research Football engine

A fork of Google Research Football (GRF) 2.10.2 with more realistic football mechanics. We build and test it first, then freeze it, then train RL agents on it.

*Last updated 2026-10-09.*

**Order:** finish patches → test → freeze `v1.0` → RL. Any change to physics or actions after RL starts makes trained agents obsolete. After the freeze, changes go into `v1.1` and need retraining.

---

## 0. Setup

| Item | Decision |
|---|---|
| Location | Its own repo, `~/projects/rl_project` (name is temporary) |
| Python env | conda env `grf`: Python 3.10, conda-forge boost-python 1.85, cmake, SDL2 |
| Source | `upstream/`: unmodified GRF 2.10.2 sdist, committed first so every patch is a readable diff |
| Build | `build.sh`: one command, rebuilds the engine and installs it into the `grf` env |
| Tests | `tests/`, run with the `grf` env's pytest |
| Git | Don't use a blanket `data/` rule in `.gitignore`; the engine's game assets live in `upstream/third_party/gfootball_engine/data/` and must be tracked |

### Layout

```
rl_project/
  PLAN.md                this file
  build.sh               configure + compile + pip install into env `grf`
  upstream/              GRF source (patched in place, history in git)
  scenarios/             our custom scenarios (test drills, stat demos)
  tests/
    test_build.py        engine imports, env steps, all scenarios load
    test_equivalence.py  all flags off ⇒ same as upstream
    test_physics.py      ball-flight measurements vs targets
    test_mechanics.py    per-patch behaviour checks
    test_match_stats.py  AI-vs-AI full matches look like football
  tools/
    measure.py           runs drills, logs ball trajectories / outcomes to CSV
    replay_review.py     dumps videos of selected drills for visual checks
```

---

## 1. Patch list (v1.0)

Every patch has an on/off switch in the scenario config, and **every switch defaults to off**. With all switches off, the engine must match upstream exactly (§2, test T1). That lets us compare with and without each patch, and keeps old scenarios and replays valid.

### P0: build fixes (done in scratch, needs moving into the repo)

Already worked out while installing:
- `cmake_minimum_required` policy: build with `CMAKE_POLICY_VERSION_MINIMUM=3.5`, or bump the version in `CMakeLists.txt`
- Make CMake use the env's Python (`Python_ROOT_DIR`, `Python_FIND_FRAMEWORK=NEVER`), not Homebrew's 3.14
- Don't link `libpython` into the module; use `-undefined dynamic_lookup` on macOS. Otherwise the process gets two Python runtimes and fails with `__init__() should return None`
- Python dependencies: pin `gym==0.25.2`, add `six`

**Done when:** a fresh clone + `build.sh` gives a working env; test T0 passes.

### P1: per-player attributes (no new actions)

**What:** individual stats per player instead of everyone at 0.6, plus preferred foot and height.

**How:**
- `FormationEntry` (`gamedefines.hpp`) gets a stats map, `preferred_foot`, `height`
- Expose them in Python via `ai.cpp` bindings and `scenario_builder.AddPlayer(..., stats={...}, foot='R', height=1.80)`
- Serialize them in `ProcessState`, so env save/restore still works
- `PlayerData` (`data/playerdata.cpp:517`) takes values from the formation entry instead of the hard-coded 0.6; missing keys stay 0.6
- Height: the engine already scales animations by `playerHeight` (`humanoid.cpp:1916`); we pass the per-player value through
- Optional: include each player's stats in the `raw` observation so agents can see them

**Stats available (from `utils.hpp:38`):** physical_balance, physical_reaction, physical_acceleration, physical_velocity, physical_stamina, physical_agility, physical_shotpower, technical_standingtackle, technical_slidingtackle, technical_ballcontrol, technical_dribble, technical_shortpass, technical_highpass, technical_header, technical_shot, technical_volley, mental_calmness, mental_workrate, mental_resilience, mental_defensivepositioning, mental_offensivepositioning, mental_vision

**Done when:**
- All defaults ⇒ identical to upstream
- In a drill with all-0.9 vs all-0.3 attackers, the 0.9 team scores clearly more (CI excludes 0)
- A fast striker (velocity 0.9) beats a slow defender (0.3) to a loose ball in a footrace drill

### P2: ball physics upgrade (no new actions)

What the engine does now (`onthepitch/ball.cpp`): gravity, quadratic drag with a single constant `drag = 0.015`, a Magnus "swerve" term with a hand-tuned curve, grass friction and spin effects, and constant bounce.

**Changes, each with its own switch:**

| Sub-patch | Change | Physical basis | Switch |
|---|---|---|---|
| P2a drag crisis | Drag coefficient depends on speed: Cd ≈ 0.45 below about 10–12 m/s, ≈ 0.20–0.25 above, with a smooth transition. This gives dipping and knuckling power shots | Wind-tunnel data for match balls (Asai, Goff & Carré). Calibrate the threshold to a modern ball. Note: the current constant 0.015 roughly equals the post-crisis value | `ball_drag_model` |
| P2b spin decay | Spin decays exponentially in flight (time constant of a few seconds), so curl bends most early | Aerodynamic torque on a spinning sphere | `ball_spin_decay` |
| P2c Magnus rework | Replace the hand-tuned `swerveAmount` curve with lift that depends on the spin-to-speed ratio | Lift coefficient vs spin parameter, from published football data | `ball_magnus_model` |
| P2d knuckle | Small random sideways force for hard, low-spin shots (deterministic under a fixed seed) | Asymmetric wake / seam effects at the drag-crisis transition | `ball_knuckle` |
| P2e pitch conditions | Expose `grassHeight`, `bounce`, `linearBounce` and friction as scenario settings with presets: `dry`, `normal`, `wet`, `long_grass` | Wet grass skids and bounces lower | `pitch_preset` |

**Note:** the engine predicts the ball's future path (`Ball::Predict`), and the AI and animations use it. All new physics must go inside that same loop so predictions stay consistent; otherwise players will misjudge balls.

**Done when:**
- Measured with `tools/measure.py`:
  - A 30 m/s shot with no spin keeps its speed over 20 m in line with post-crisis drag
  - A 15 m/s chip loses speed faster (pre-crisis)
  - A curl shot from 20 m bends about 1–3 m sideways (TV analyses of real free kicks: about 2–4 m over 25 m)
- A wet pitch gives measurably lower bounce and faster skid than dry
- AI-vs-AI goal rates stay within the realistic band (T5)

### P3: new kick actions (changes the action list)

**New actions:**

| Action | Behaviour | Built on |
|---|---|---|
| `curl_shot` | Shot with extra side spin; bends inward, toward the far post. Slightly less power | Shot path (`GetShotVector`) + spin (`zRot`) |
| `through_ball` | Ground pass aimed into space ahead of the receiver's run, not at their feet; hit harder | Pass targeting in `AIfunctions.cpp:1079/1096`: more lead time, aimed along the run and toward goal |
| `chip` | Lofted shot or pass with backspin, high and slow; for lobbing the keeper | High-pass path + backspin |

**How:**
- C++: add `game_curl_shot`, `game_through_ball`, `game_chip` and their release actions to the end of `enum Action` (`gfootball_actions.h`), so existing numbers don't change
- Bind them in `ai.cpp`, and map each to a function type plus a modifier flag on the player's command
- Python: add `CoreAction`s in `football_action_set.py` and a new action set `v3` = default 19 + 3 = **22 actions**. The default action set stays the same, so old code still works
- Gamepad: map the new actions to unused buttons in `gamepad.py`
- Built-in AI: optional, and off by default for v1.0; AI opponents use only the old actions unless a switch is on

**Done when:**
- `curl_shot`: higher scoring rate than `shot` from the edge of the box toward the far post in a drill; measured bend in the right direction
- `through_ball`: the receiver gets the ball further ahead of the defender than with `short_pass`, in a timed-run drill
- `chip`: scores more often than `shot` against an advanced keeper (rushing out)
- Gamepad: each new action can be triggered by hand

### P4: situational accuracy (no new actions)

**What:** shot and pass error depends on the situation, not just randomness. Also first touch.

**How:**
- Error size is scaled by these factors (each bounded; the formula is written down in the code):
  - Defensive pressure: distance and number of opponents within about 2 m
  - Body angle between facing and target
  - Ball height at contact (volleys and half-volleys harder)
  - Weak foot (needs P1's `preferred_foot`)
  - The relevant stat (`technical_shot` / `technical_shortpass` / …) and `mental_calmness`
- Where: the random-error terms in `humanoid_utils.cpp:481–525` (shots) and the pass equivalents
- First touch: when receiving, a speed-dependent miscontrol chance (ball bounces away further), scaled by `technical_ballcontrol`

**Done when:**
- In a drill, shots under pressure have a lower on-target rate than unpressured ones (CI excludes 0)
- Weak-foot shots are measurably less accurate
- A 30 m/s incoming pass gives longer first-touch bounce than 10 m/s
- With all-default settings the overall error is about the same as upstream, so the patch redistributes error by situation rather than making everything harder

---

## 2. Testing

| ID | Test | What it shows | When |
|---|---|---|---|
| **T0** | Build smoke test | Imports, create env, step 1000 random actions; every built-in scenario loads | Every build |
| **T1** | Equivalence | All switches off ⇒ with the same seed and the same actions, ball and player positions match upstream **exactly** over 3 full matches. Upstream is recorded once as reference files | Every build |
| **T2** | Determinism | `deterministic=True`, same seed ⇒ two runs give identical results, with every patch switched on | Every build |
| **T3** | Physics measurements | Drill trajectories vs the targets in P2; results saved to CSV and plotted | Per P2 change |
| **T4** | Mechanics drills | The "done when" checks for P1, P3 and P4, run as repeated drills (≥ 500 episodes each) with bootstrap CIs | Per patch |
| **T5** | Match realism | 50 AI-vs-AI 11v11 matches per config. Goals/match, shots, shot conversion, pass completion, possession must fall in realistic bands (real top leagues: about 2.5–3 goals per match, about 10% conversion, about 80% pass completion; GRF's own baseline is recorded first) | Before freeze |
| **T6** | Visual review | Videos of each drill + 2 full matches, watched by a person: does curl look like curl, does a through ball look like a through ball | Before freeze |
| **T7** | Save/restore | `env.get_state()` / `set_state()` keep all new fields; the restored game continues identically | Every build |
| **T8** | RL smoke test | PPO trains for about 1 M steps on `academy_empty_goal_close` with the v3 action set; reward goes up. This catches broken observations or actions before a long run | Before freeze |

**Speed:** upstream runs about 840 steps/s (random actions, academy drill, this Mac). The patched engine must stay above about 700 steps/s, or RL gets noticeably slower.

---

## 3. Order of work

1. **P0**: copy the source into `upstream/`, apply the build fixes, write `build.sh`. Make the first commit the unmodified source so the build-fix diff is visible
2. **Test harness first**: T0, T1 (record upstream reference files *before* any patch), T2, T7, and `tools/measure.py`
3. **P1** attributes. Everything else uses it
4. **P2** ball physics, one sub-patch at a time, each measured with T3
5. **P4** situational accuracy (needs P1)
6. **P3** new actions, last, because it changes the action list
7. **T5 + T6 + T8**, then freeze **v1.0** with a git tag
8. **RL**: separate plan

## 4. Open questions to settle during the work

- **Real numbers for P2:** which ball to model and where to get the drag/lift curves. Starting point: published wind-tunnel studies of modern match balls.
- **Engine units:** the ball code looks like it uses metres and seconds, but this needs checking before physical constants mean anything.
- **Should the AI opponent use P3 actions?** Off for v1.0 keeps the opponent the same as standard GRF and makes comparisons with published results possible.
- **Should observations include player stats?** Changes the observation size; decide before the freeze.
