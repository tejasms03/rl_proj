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

### P1: per-player attributes (no new actions) — **done**

**Correction to the original premise:** players are *not* all 0.6. Each team is built from 11 hard-coded profiles (`data/teamdata.cpp:277`), assigned by the order players are added to the scenario. Both teams get the same profiles. Slot 0 (usually the keeper) has a centre-back profile, and slots 5–10 are attackers with many stats at 1.0. The 0.6 constructor is only used for officials. Stats are then scaled by team difficulty and fatigue.

**As built:**
- `AddPlayer(..., stats={...}, height=1.85, preferred_foot='left')` in scenario files
- `player_overrides` env option, applied by slot to **any** scenario:
  `{'left' | 'right': {slot | '*': {'stats': {...}, 'height': m, 'preferred_foot': 'left' | 'right'}}}`
- Raw observations gain `left_team_attributes` / `right_team_attributes` (players × 24: 22 stats, height, preferred foot), fetched once per episode; columns in `football_env_core.PLAYER_ATTRIBUTE_NAMES`
- Overrides are saved in env state, so save/restore keeps them
- Tests: `tests/test_p1_attributes.py`, plus a P1 config in the determinism and save/restore tests

**Finding: stats matter only a little in GRF.**
- Top sprint speed is `8 m/s × (0.9 + 0.1 × velocity)`, so 7.2–8.0 m/s across the whole stat range (`playerbase.cpp:133`). Real players range from about 7.5 to about 9.8 m/s.
- A faster player still wins a footrace for a loose ball 15/15 (AI on both sides), so the effect is consistent, just small.
- Attackers with *every* stat at 0.9 vs 0.3 score 33% vs 26% in `academy_3_vs_1_with_keeper` (300 episodes each, p ≈ 0.06).
- Addressed by **P1b** below.

### P1b: realistic player physics (switch `realistic_player_physics`) — **done**

**What:** top speed and acceleration ranges calibrated to real professional players, so stats matter.

**As built:**
- Top sprint speed = 7.8 + 2.0 × velocity m/s (upstream: 8 × (0.9 + 0.1 × velocity) = 7.2–8.0)
- Acceleration force factor = 0.45 + 0.35 × acceleration (upstream 0.7 + 0.3 × acceleration)
- Two upstream caps had to follow the player's own top speed, or nobody could exceed 8 m/s: the cornering brake (`humanoidbase.cpp`, `topVelocity`) and the clamp of animation speed toward the *requested* speed, which is in 8 m/s units (`desiredVelocity` is now stretched like the animation)
- Switches can be set by a scenario (`builder.config().realistic_player_physics = True`) or as an env option of the same name, which wins; see `scenario_builder.PATCH_SWITCHES`
- Tool: `python -m grf_tools.sprint [--realistic]`; drill `drill_sprint` (the ball sits with our own keeper: with a loose ball, the engine steers a fast enough controlled player toward it)

**Calibration** (standing start, timed from first movement):

| Velocity / acceleration | 10 m | 20 m | 40 m | Top speed | Reference |
|---|---|---|---|---|---|
| 0 / 0 | 2.02 s | 3.36 s | 5.92 s | 7.8 m/s | slow centre-back / keeper, ~5.6–5.9 s for 40 m |
| 0.5 / 0.5 | 1.79 s | 2.96 s | 5.24 s | 8.8 m/s | elite pro average ≈ 1.78 / 3.0 / 5.2 s |
| 1 / 1 | 1.60 s | 2.66 s | 4.70 s | 9.8 m/s | fastest wingers ≈ 4.7–4.9 s, ~36 km/h |
| upstream 0.5 / 0.5 | 1.74 s | 3.06 s | — | 7.6 m/s | |

**Effect:** in `academy_3_vs_1_with_keeper` (AI attackers, 300 episodes each), all stats 0.9 vs 0.3 score 35% vs 18% with the switch on, against 33% vs 26% upstream. AI-vs-AI 11v11: 2.12 goals/match both on and off (8 matches each).

**Note:** sprint animations are stretched to the higher speeds, so very fast players may look like they glide slightly when rendered. Physics and AI timing use the same top speed, so play is consistent.

**Original design notes:**

**What:** individual stats per player, plus preferred foot and height.

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

### P2: ball physics (switches `realistic_ball_physics`, `pitch_condition`) — **done**

**Change from the original plan:** one switch for flight + kicks instead of five independent ones. Physical drag/lift only make sense with physical kick spins and with pass power solved for the new flight, otherwise passes land in the wrong place. Knuckleballs (P2d) are deferred: the built-in AI predicts the exact ball path, so random wobble would be fully anticipated.

**As built (`onthepitch/ball.cpp`, shared by the ball simulation and the pass solver):**
- Drag crisis: Cd 0.47 below ~10 m/s to 0.22 above ~14 m/s (centre 11.5 m/s); `airK = ρA/2m = 0.0541 /m`
- Magnus lift `C_L = 0.35·tanh(S/0.3)`, `S = r·ω⊥/v`; spin decays in flight with τ = 5 s
- Realistic grass: rolling resistance `0.012·v² + 0.75 m/s²` (FIFA standard: 4–8 m from ~3 m/s; upstream ~2.3 m)
- Kicks: lofted passes get backspin (high 45 rad/s, long 30 rad/s); pass-curl sidespin capped at ~40 rad/s; shots unchanged
- Pass solvers: lofted/long passes solve launch speed by simulating the flight (`RealisticSpeedForCarry`); ground passes solve it in closed form for an arrival speed (`RealisticGroundPassSpeed`), on the current pitch
- `pitch_condition`: `normal` (upstream values unless realistic) / `dry` / `wet` / `long_grass`
- Test hook `launch_ball` + `grf_tools.ballistics`; `grf_tools.matchstats`; `grf_tools.record` (videos)
- Bug fixed on the way: `Quaternion::GetRotationMultipliedBy` returns NaN when |w| drifts above 1, which crashed full matches; spin decay uses a safe small-angle scaling instead

**Findings:**
- Observations report spin scaled like positions (x ÷ 54.4, y ÷ −83.6); upstream kick spins are ~40–50 rad/s topspin on passes, 10–22 rad/s backspin on lofted balls, small sidespin
- With the current (strictly realistic) pace the built-in AI creates fewer chances: 24 AI-vs-AI matches give 1.4 goals and 4.6 shots per match vs 2.6 and 10.3 upstream. The AI's pass-risk heuristic assumes upstream ball speeds. Addressed by the `game` pace below

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

## 3. Phase 0: freeze the environment (locked spec, 2026-10-09)

Decided with the user: the game should feel like modern FIFA / EA FC, the agents control all 11 players with full information, and they learn kick power themselves.

**Pace** (`ball_pace`: `game` default when `realistic_ball_physics` is on; `realistic` = current behaviour):

| Kick | Launch | Trajectory | Lands |
|---|---|---|---|
| Ground pass | 14–22 m/s (upstream tempo) | low | arrives firm, 8–12 m/s |
| Driven long ball | 26–32 m/s | 12–18°, backspin | ~90% of the way, runs on |
| Lofted pass | 24–30 m/s | 25–30°, backspin | on the target |
| Shot | unchanged (~28 m/s) | | |

**Pace — done.** Measured at game pace (receiver 40 m away): long pass 26 m/s at 17°, arrives at 38.7 m; high pass 22 m/s at 28°, peak ~5 m; ground pass (15 m) 14 m/s. The ground-pass solver integrates the rolling deceleration numerically (the closed form assumed slow-ball drag throughout and over-hit long ground passes).

**Kick power — done** (switch `kick_power_control`):
- Agents send `(action, power)` per player; the engine keeps the last power sent until a new one arrives (a kick executes a step or two after the press). Without a power, humans get the hold-to-charge gauge
- Power maps linearly to intended distance: ground pass 3–35 m, driven long ball 15–50 m (lands at ~90%), lofted pass 10–60 m — trimmed from the first proposal to what the engine's kick-power cap allows. Shots: power feeds the existing shot-power path
- Two assists instead of one: `pass_direction_assist` (default 1: aim snaps toward the teammate nearest where you point — agents only have 8 directions) and `pass_power_assist` (default 0: power sets the distance exactly). 1 = upstream fully automatic; between 0 and 1 the upstream semi-assisted blend, which fades when you aim far from any teammate
- Calibrated with `grf_tools.measure.power_sweep` on `drill_power`: every power lands within 4% of the target (the 3 m tap within 0.6 m). Solver corrections from measurement: ground ×1.1 (loft + first bounce), driven long ball solved for 80% (run-up adds ~12%)
- `gfootball.env.wrappers.KickPowerWrapper`: continuous `Tuple(MultiDiscrete, Box)` or `bins=21` → `MultiDiscrete`
- With the switch off, a power value has no effect (tested)

**Archetypes — done** (`gfootball/env/archetypes.py`): 19 archetypes over GK, CB, FB, DM, CM, W, ST (e.g. stopper / ball-playing / aerial CB; poacher / target man / false nine / pressing forward). Every archetype has the same stat budget (mean 0.70) and none dominates another of its position. Use them at kickoff through `player_overrides` (stats + height) or mid-game.

**Mid-game swaps — done:** `env.unwrapped.set_player_stats('left', i, {...})` and `env.unwrapped.apply_archetype('left', i, 'ST', 'target_man')` change a player's base stats during a match (index = observation order); `*_team_attributes` updates immediately. Height is fixed at kickoff; `set_state()` does not undo swaps. Tested: a sprinting player goes from 7.8 to 9.8 m/s after a swap.

**Linux build — written, not yet tested:** `docker/Dockerfile` (Ubuntu 22.04, conda env, conda-forge compiler, `build.sh`, runs the tests). The extension no longer links libpython on any Unix. Training starts on the user's M1 Pro; test the image when the teammates' machines are used.

**Acceptance:** T1 still exact with every switch off; passes land within ±10% of the intended distance at each power; the user signs off on re-recorded clips.

**Known limitation (accepted 2026-10-09):** the built-in AI is hand-written rules tuned for upstream physics, so it creates fewer chances on the new physics: 24 AI-vs-AI matches at `game` pace give 1.5 goals / 4.9 shots per match vs 2.6 / 10.3 upstream (upstream grass inside realistic mode: 1.2 / 7.0, so grass is not the single cause). RL agents train on the new physics and are unaffected; the built-in AI is just a softer early benchmark. Optional backlog: retune its pass-risk timing and close control for the new physics.

**Cut first if late:** P3 extra kicks (curl, through ball, chip), P4 situational accuracy.

## 3b. Project timeline

| Phase | Dates | What | Who |
|---|---|---|---|
| 0. Freeze environment | Oct 12 – Oct 25 | the spec above, tag v1.0 | Claude; the user designs the agent interface |
| 1. RL foundations | Oct 19 – Nov 1 | own PPO with action + power heads, parallel envs, logging | user |
| 2. 11-player agents | Nov 2 – Nov 15 | MAPPO with shared weights, per-player stats/role inputs, randomized archetypes/tactics, curriculum | user |
| **Checkpoint** | **Nov 15** | beats the easy built-in AI in 11v11, else fall back to 5-vs-5 | |
| 3. Style / player conditioning | Nov 16 – Nov 29 | style-conditioned vs player-centric training | user |
| 4. Manager | Nov 30 – Dec 13 | manager picks tactics and archetypes every ~5 game-minutes | user |
| 5. Experiments, write-up | Dec 14 – Dec 24 | style × player grid, chemistry via passing networks, demo | user |

## 4. Findings so far

- **Engine units are metres and seconds.** Pitch 110 × 72 m (`gamedefines.hpp:244`). Observations scale positions by x ÷ 54.4, y ÷ −83.6, z unchanged (`defines.hpp:55`). Physics runs at 100 Hz; one env step = 10 ticks = 0.1 s by default.
- **One game per process.** The first env created in a process fixes game-level settings such as `physics_steps_per_frame` for every later env in that process. `grf_tools.trace.make_env` refuses to mix tick rates, and `grf_tools.measure` runs in its own process. For RL, every parallel env needs its own process (`SubprocVecEnv` or equivalent).
- **Baseline kicks (unpatched engine, 20 seeds, `python -m grf_tools.measure`).** Medians:

  | Kick | Launch speed | Launch angle | Peak height | Travel |
  |---|---|---|---|---|
  | shot (from 22 m) | 27.9 m/s | 4° | 0.27 m | 21 m (to goal) |
  | short pass (to 15 m) | 18.2 m/s | 6° | 0.34 m | 13 m |
  | long pass (to 40 m) | 31.0 m/s | 11° | 3.0 m | 30 m |
  | high pass (to 40 m) | 32.2 m/s | 20° | 4.6 m | 33 m |

  Shots and short passes look plausible. **Lofted balls are too flat:** a real 40 m lofted pass peaks at about 8–10 m with a steeper launch. This belongs to P2 (with P3's chip).

## 5. Open questions to settle during the work

- **Should the AI opponent use P3 actions?** Off for v1.0 keeps the opponent the same as standard GRF and makes comparisons with published results possible.
- **Observations:** raw observations include every player's stats (P1). The per-player observation for the agents is the user's design (full information decided).
- **Chemistry** (later, phases 3–4): per-pair score that grows with minutes and passes together; low chemistry adds noise to teammate positions / pass accuracy. Gives the manager a cost for swaps.
