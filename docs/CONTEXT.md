# Touchline: project context

*Everything the team needs to know about the project so far, in one place. Last updated 2026-11-20.*

---

## The project in plain words

**We are teaching computer players to play football as a team, with a coach.**

- The **game** is a free football video game made for AI research (Google Research Football). We changed it a little to make it more realistic.
- **Eleven player AIs** learn to play. Each one is told a simple job ("press their player on the ball", "mark their striker", "stay wide") and learns to do that job well.
- One **manager AI** (the coach) watches the match and decides which job each player gets, and changes the jobs during the match when something isn't working.
- They all **learn by trial and error** (reinforcement learning): they play thousands of matches, get points for good outcomes (goals, winning the ball) and slowly get better.

**The question we answer:** does a coach AI that changes its players' jobs during a match do better than sticking to one fixed tactic?

**What is done:** the game is set up and improved, the plan is written, and we have checked by hand that 18 of the 21 jobs can actually be done in the game.

**What is left:** write the learning code, train the players, train the manager, measure the results, present.

---

## Contents

1. [Where everything is](#1-where-everything-is)
2. [Team and roles](#2-team-and-roles)
3. [Decisions made (and why)](#3-decisions-made-and-why)
4. [The game (patched Google Research Football)](#4-the-game-patched-google-research-football)
5. [How the learning works](#5-how-the-learning-works)
6. [The duty lab: can every instruction be carried out?](#6-the-duty-lab-can-every-instruction-be-carried-out)
7. [Lessons from the lab that training must respect](#7-lessons-from-the-lab-that-training-must-respect)
8. [Shortcuts that save training time](#8-shortcuts-that-save-training-time)
9. [Compute](#9-compute)
10. [How we will judge success](#10-how-we-will-judge-success)
11. [Known weak spots of the approach](#11-known-weak-spots-of-the-approach)
12. [Documents](#12-documents)
13. [Open items](#13-open-items)
14. [Glossary](#14-glossary)

---

## 1. Where everything is

**Repository:** https://github.com/tejasms03/rl_proj (public). Local folder: `~/projects/rl_project`.

| Path | What it is |
|---|---|
| `upstream/` | The game (Google Research Football 2.10.2) with our patches, C++ engine + Python env |
| `build.sh` | Rebuilds the engine and installs it into the `grf` conda env |
| `tests/` | Automated tests (physics, determinism, controls, kicks, archetypes) |
| `PLAN.md` | The engine patch plan (frozen as `v1.0`) |
| `docs/training_plan.md` | **The RL plan**: duties, observation, actions, rewards, manager, stages, shortcuts (§4.1) |
| `docs/observation_spec.md` | What each player sees (869 numbers + 91 for instructions = 960) |
| `docs/manager_reward_shaping_plan.md` | Manager reward notes |
| `docs/project_scope.pdf` | 5-page scope briefing for the team (assumes no football/RL background) |
| `grf_tools/` | Our tools: play with a gamepad, watch matches, record clips, measure physics |
| `grf_tools/duty_lab/` | **The duty lab** (§6): scripted version of every instruction, tested in 11v11 |
| `out/duty_lab/` | Lab results: `summary.md`, and per duty a before/after video, path picture and numbers |

**Setup:** conda env `grf` (Python 3.10). Build with `./build.sh`. Run tests with the `grf` env's `pytest`.

**Useful commands:**

```bash
# Watch an AI-vs-AI El Clasico (no controller needed)
~/projects/rl_project/elclasico.sh --watch

# Play it yourself with a PlayStation controller
~/projects/rl_project/elclasico.sh

# Run the duty lab (all tests, 12 seeds; needs the screen awake for videos)
cd ~/projects/rl_project && ~/miniconda3/envs/grf/bin/python -m grf_tools.duty_lab.run --seeds 1 2 3 4 5 6 7 8 9 10 11 12

# Same, numbers only (works with the screen asleep, about 10 s per test)
cd ~/projects/rl_project && ~/miniconda3/envs/grf/bin/python -m grf_tools.duty_lab.run press mark --no-video --seeds 1 2 3 4 5 6 7 8 9 10 11 12
```

---

## 2. Team and roles

Five people, all with the same RL background. One (the project lead) is the group's football person. **Split agreed 2026-11-20: 2 on the manager, 2 on the outfield players, 1 on the goalkeeper.** Each group has a detailed task list:

| Group | People | Owns | Shared piece also owned | Task list |
|---|---|---|---|---|
| **Manager** | M1 (project lead), M2 | Tactic sheets, duty definitions, the manager network (M1); manager env, what it sees, the rule-based team and style-switching opponents (M2) | PPO core, cloud machines and the GPU-hours log, demo and clips (M1); H2 evaluation (M2) | [`docs/roles/manager.md`](roles/manager.md) |
| **Outfield** | O1, O2 | One shared network for the 10 outfield players: training and curriculum (O1); env wrapper, duty checks, rewards, drills (O2) | Env wrapper used by everyone (O2); H1/H3 evaluation (O1) | [`docs/roles/outfield.md`](roles/outfield.md) |
| **Goalkeeper** | G | A **separate keeper network**: keeper duties, drills, rewards, training | — | [`docs/roles/goalkeeper.md`](roles/goalkeeper.md) |

**Critical path:** M1's PPO core and O2's env wrapper (week 1): everything else waits on them. The manager group trains first on M2's rule-based team, so it doesn't wait for trained players.

**Rules that stop people blocking each other:**
- Fix the interfaces first (instruction block, observation layout, reward inputs/outputs, checkpoint format); then everyone works behind them.
- Everyone reviews someone else's code across groups.
- If outfield training falls behind, M2 helps O1: that's where most training hours go.

GitHub collaborators invited 2026-11-20: chaniketh, bernardbalko, ram10000-nyu (crc9983 not found; to confirm).

Claude (AI assistant) built the engine patches and the duty lab, and reviews; the RL code is written by the team.

---

## 3. Decisions made (and why)

| Date | Decision | Why |
|---|---|---|
| 2026-10-09 | Game feel like FIFA/EA FC (physical ball, game-paced kicks); agent controls kick power (0 to 1) | Feels right to play; the agent learns how hard to hit the ball |
| 2026-10-09 | Players see "anything a human sees on the screen" | Same view in training and in a real match |
| 2026-10-09 | Engine **frozen as `v1.0`** | Any physics/action change after training starts makes trained agents obsolete |
| 2026-11-05 | 5-person group; P1 owns manager + PPO core | Football knowledge needed for tactics |
| 2026-11-09 | Repo public at `tejasms03/rl_proj` | Team access |
| 2026-11-11 | The manager **directs players individually** (per-player duties + team dials), not just one of six fixed tactics; the six tactics are starting points | Real managers make targeted changes ("double their winger"); fixed presets are too rigid |
| 2026-11-11 | Duties are built from **building blocks** (who to position against, how close, which side, how to move, when) | New duties without new code; one generic duty check |
| 2026-11-11 | Judge every addition against simpler versions with pass marks set in advance | So we know if it helps or only complicates |
| 2026-11-17 | **11v11 only, final.** Small drills are fine as training steps, but everything submitted is 11v11 | Project lead's decision |
| 2026-11-18 | Screen lane is **not** in the first version of duties | It didn't beat the game's AI in the lab |
| 2026-11-18 | Shortcuts: the game's AI plays the ball early on + clone the lab scripts (§8) | Biggest time savings for the effort |
| 2026-11-18 | Ball-winning duties scored on outcome; attacking duties tested with a willing passer (§6) | Time-in-position was the wrong yardstick for them |
| 2026-11-20 | Preliminary report written | Course requirement |
| 2026-11-23 | **Step length = the game's default:** 10 ticks per step, one step = 0.1 s (10 decisions per game second) | The game's standard; fewer steps per match than the lab's 0.04 s; `observation_spec.md` is written for it |
| 2026-11-23 | Cloud machines, GPU-hours log and demo clips moved from the keeper role to M1 | M1 has no early work |
| 2026-11-23 | **Action set `v4`** for training: `v3` + `builtin_ai`, `keeper_rush`, `pressure`, `team_pressure` (+ their release buttons), 29 in all | Press-type duties need `pressure`; the "AI plays the ball" shortcut needs `builtin_ai`; `full` has useless extras |
| 2026-11-23 | **Our keeper saves by himself** (`keeper_auto_save` engine switch), not by handing him to the game's AI | A controlled keeper can't save at all otherwise (0/20 vs 20/20 conceded); the network keeps positioning |
| 2026-11-23 | **Magnet off** (`use_magnet: false`) for our players | The magnet overrides where the network sends a player; duties need exact movement |
| 2026-11-23 | **No retune of the game's AI** | Re-measured: 2.00 goals/match on our physics vs 2.19 original (within noise); only shots are fewer (5.4 vs 9.8) |

Also: substitutions in this project mean **swapping player archetypes** (manager v3), and **set-piece routines** are planned (manager v3).

---

## 4. The game (patched Google Research Football)

Google Research Football (GRF) is an open-source 3D football game for AI research. Agents can control all 11 players of a team; the game also has its own rule-based AI.

**What we changed** (every change has an on/off switch; all off = original game, checked by tests):
- **Ball physics:** realistic flight, spin and grass; "game pace" tuned to feel like FIFA.
- **Kicks:** agent-controlled kick power with an assist level; extra kick types (curl shot, chip, through ball).
- **Player archetypes:** 19 profiles (target man, false nine, ball-winning midfielder, sweeper keeper, ...) with equal stat budgets, swappable mid-match.
- **Situational accuracy:** shooting under pressure is harder.
- **Human play (demo only, off for training):** PlayStation controller, defending mode, selection/landing rings, custom kits, player looks (skin, hair), `--watch` spectator mode.
- **Offside:** checked and correct (same rate as original GRF); a diagnostic log is off by default.

**Our squad idea:** a squad with one of each archetype (including both goalkeeper types), 11 on the pitch at a time. During training, **randomise which archetypes play and in which slot every match**, or the policy only learns one line-up.

---

## 5. How the learning works

Full detail: `docs/training_plan.md`.

**Players (one shared network for all 11, trained with MAPPO):**
- Each player sees the match plus its **instruction block**: its three duties (defending, attacking, transition), their targets (which opponent, which teammate, which zone) and six team dials.
- During training a **critic** sees the whole match, which helps judge whether a team move was good. In a real match only the player's own view is used.
- **Actions** every frame: move direction (stop + 16), speed (walk/sprint/close control), ball action (none, short, high, long, shot, curl, chip, through ball), defensive action (none, pressure, team pressure, sliding, keeper rush), kick power (0 to 1).
- **Reward:** goals and results, plus small shaping (ball progress, shot quality, losing the ball in our third). Early on, a small bonus while the player's duty check is true; it fades to zero so the final team is judged on results only.

**Duties** (first version marked ✓):

| Defending | | Attacking | | Transition | |
|---|---|---|---|---|---|
| D1 Hold zone ✓ | D5 Mark ✓ | A1 Hold shape ✓ | A5 Drop to receive | T1 Counter-press | |
| D2 Press ✓ | D6 Screen lane | A2 Support ✓ | A6 Arrive in the box | T2 Recover ✓ | |
| D3 Contain | D7 Hold line ✓ | A3 Width ✓ | A7 Rest defence | T3 Break | |
| D4 Cover ✓ | D8 Keeper sweep | A4 Run in behind ✓ | | | |

**Team dials** (0 to 1): line height, press trigger, width, tempo, risk, compactness.

**Manager (PPO, decides every ~10 s of game time and at restarts):**
- Sees: score, clock, opponent tendencies (pass length, long balls, their line height, ...), and how the current plan is working (shots, xG, possession, turnovers).
- Does: v1 picks one of six tactics (high press, low block, possession, counter, wing play, false nine); v2 edits individual players' duties and nudges the dials; v3 swaps archetypes and picks set-piece routines.
- Reward: goal difference after each decision, a small xG term, the result, and a small cost per change (no thrashing).
- Trained with the players frozen. It can start on the rule-based team (§8) before player training is finished.

**Training stages:**

| Stage | What | Done when |
|---|---|---|
| 0 | Small drills, one duty at a time (2v2 press + cover, 1v1 mark, ...) | each duty check true ≥ 70% of drill time |
| 1a | 11v11 vs the game's AI, easy → hard; no instructions at first (basic football), then random duties | beats the hard AI over 100 games |
| 1b | 11v11 + self-play against past versions | beats hard AI; duties satisfied ≥ 70% |
| 2 | Random mid-match tactic switches | switching visibly changes play, ≤ 10% win-rate loss |
| 3 | Manager v1 (players frozen) | beats every fixed tactic; reacts to a style switch within 60 s |
| 3b | Manager v2/v3 | beats manager v1 |

**One shared policy, not one per duty:** basics are learned once, switching duties mid-play is smooth, and duties can be combined. Train all drills mixed so no duty is forgotten. A separate per-duty policy is only a debugging tool.

---

## 6. The duty lab: can every instruction be carried out?

**Idea:** before training anything, write a hand-made script for every instruction and run it in the real game. If a script can't make a player do a duty, a trained player won't either.

**How each test works:**
- A full **11v11** scene set up for the situation (they build from the back, they attack our half, we build, we attack, a winger crosses, ...).
- Run twice from the **same random seed**: **before** = the player is left to the game's own AI; **after** = he follows the instruction. The only difference is the instruction. Every run replays exactly from its seed.
- Outputs per duty in `out/duty_lab/<duty>/`: `compare.mp4` (before on top, after below, mini-pitch with the instructed players in orange), `trace.png` (top-down path), `metrics.json`. All duties: `out/duty_lab/summary.md`.
- 12 seeds per test (24 for screen lane).

**Three ways of scoring, depending on the duty:**

| Kind | Scored on | Pass mark |
|---|---|---|
| Positional (line, zone, mark, cover, shape, ...) | Share of time the duty check is true ("met" = in position or closing in at ≥ 1 m/s) | Met ≥ 70% of the time it applies, and more than before |
| Ball-winning (press, contain, double team, cover-shadow press) | How their spells on the ball end once we get within 3 m | Win at least as many as the game's AI, and faster |
| Attacking with a pass (support, width, run in behind, ...) | A scripted "willing passer" plays it to him at his duty's moment, same rule in both runs | More passes reach him than reach the AI player in that slot |

**Results (21 instructions: 18 work, 3 don't):**

| Duty | Result | Key number (before → after) |
|---|---|---|
| Hold the line (back four) | ✅ | in position 9% → 91%; 7.2 → 0.5 m off the line |
| Hold zone | ✅ | in position 76% → 100% |
| Keeper sweeps | ✅ | 4.3 → 11.5 m off his line, in position 19% → 69% |
| Mark | ✅ | in position 13% → 48%; 6.2 → 3.5 m from his man |
| Cover | ✅ | in position 10% → 62% |
| Press | ✅ | balls won 1 → 7; 0.8 → 0.3 s to win it |
| Contain | ✅ | balls won 7 → 14 (but its real job is to delay, not win the ball) |
| Double team | ✅ | balls won 19 → 19, twice as fast (2.8 → 1.3 s) |
| Press with cover shadow | ✅ | balls won 1 → 6 |
| Hold shape | ✅ | in position 41% → 100% |
| Rest defence | ✅ | in position 22% → 100% |
| Inverted full-back | ✅ | in position 0% → 49% |
| Pin their full-back | ✅ on position, **no effect** | in position 17% → 62%, but their full-back stays ~31 m from his goal either way |
| Support | ✅ | passes reaching him 5 of 12 → 18 of 22 |
| Width | ✅ | 0 → 6 of 21 (many long passes go out) |
| Run in behind | ✅ | 0 → 3 of 10 |
| Drop to receive | ✅ | 19 of 31 → 25 of 39 |
| Overlap | ✅ | 0 of 2 → 8 of 12 |
| **Screen lane** | ❌ | their midfielder got the ball 10 → 16 times (worse than the AI) |
| **Arrive in the box** | ❌ | on the spot 8% → 49%, but only 1 of 13 crosses reaches him either way |
| **Third-man run** | ❌ | 4 of 13 → 2 of 9 |

**What the fails mean:** these three are possible to carry out but a hand-written script isn't good enough; they're better learned than hand-coded. Screen lane is out of the first version; the manager can use "press with cover shadow" to cut off a midfielder instead.

**Football lessons from the lab:**
- One presser alone rarely reaches the ball before it moves; **pressing works as a team job**.
- Positional duties (line, zone, cover, mark) are easy to carry out; duties that depend on **timing with a pass** (third-man run, crosses) are hard.
- Some scenes needed football common sense to be fair: an overlap needs the winger to hold the ball up with a moment of space; a cross needs the winger to reach the byline first.
- Not yet scripted: transition duties (counter-press, recover, break) and set pieces.

---

## 7. Lessons from the lab that training must respect

These are bugs or traps we hit; the training code must handle them (also in `docs/training_plan.md` §4.1).

1. **Possession blinks.** Between a dribbler's touches the game says *nobody* has the ball. Without a short memory (keep the last owner for up to 1 s of loose ball), every duty that follows the ball switches off and on several times a second.
2. **Kicks have a wind-up** of about 8 frames. Handing a player to the game's AI (`builtin_ai`) during that time resets his controls and the kick silently never happens. Keep the player under our control until the ball has left his foot.
3. **Action set (settled 2026-11-23).** The frozen `v3` set has no pressure, team pressure, keeper rush or builtin_ai. We train on **`v4`** (`v3` plus those four and three release buttons). `sticky_actions` has 13 entries with `v4`.
4. **The other team's controls are mirrored.** For the right-hand team, "right" means forward. Matters for scripted opponents and self-play.
5. **Rendering changes the game.** A run with video and a run without, from the same seed, play out differently. Never compare one with the other.
6. **Velocities (settled 2026-11-23).** `*_direction` is the movement over 1/ticks seconds, so **speed per second = direction × ticks per step**. At the default 10 ticks that is exactly one step (0.1 s), as `docs/observation_spec.md` assumes. The lab ran at 4 ticks (0.04 s steps), where direction covers 0.25 s. Measured: fastest player 0.80 m per step at 10 ticks, 0.32 m at 4 ticks (both 8 m/s). Lab figures quoted in steps are 0.04 s steps: the kick wind-up of ~8 lab steps is ~3 steps at 0.1 s.
7. **One engine per Python process.** Creating a second environment in the same process with a different `physics_steps_per_frame` silently keeps the first setting. Run each setting in its own process.
8. **Video needs the screen awake.** On a Mac, rendering stalls when the display sleeps; use `--no-video` for overnight runs.
9. **A keeper we control can't save** unless `keeper_auto_save` is on: the game's dive/catch logic only runs for its own AI keeper (Bernard's shot test, 2026-11-23).
10. **`keeper_rush` is an outfield button.** It tells the game's AI keeper to come out; pressed by our own keeper it does nothing. Mask it off for our keeper. (The lab's scripted sweep pressed it too; its result came from movement alone.)

---

## 8. Shortcuts that save training time

Ranked by time saved for the effort (decided 2026-11-18):

| Rank | Idea | Notes |
|---|---|---|
| 1 | **The game's AI plays the ball** early on: whenever one of our players has the ball he is handed to the AI; our policy learns everything off the ball (the duties) first, then takes the ball back | Very cheap. Until the hand-back, goals/wins are partly the AI's; report it that way. Watch the kick wind-up trap (§7.2). |
| 2 | **Clone the lab scripts first** (behaviour cloning), then PPO | Skips the aimless early phase. Scripts' exact commands are known. |
| 3 | The AI as opponent, easy → hard | Free curriculum, already planned |
| 4 | The AI covers moments a duty doesn't (loose balls) inside the rule-based team | The lab already does this |
| 5 | Clone the game's AI | Skip: the AI moves players directly and never presses our buttons, so there's no record of its commands |

**Plan: 1 + 2.** Lab scripts off the ball + AI on the ball = a complete **rule-based team**. It is (a) the cloning teacher, (b) the manager's first training partner and (c) a baseline to beat. This is P5's team.

---

## 9. Compute

- **About 300 hours of free university cloud GPU time.** Platform details still to confirm (session limits decide whether we need notebooks or job scripts).
- **The game runs on the CPU only**; the GPU speeds up the network. Speed depends mostly on **CPU cores**: one game per core, many games in parallel. Pick machines with many vCPUs; a mid-range GPU is enough.
- **Rules:** never debug on cloud time (5-minute local smoke test first); save checkpoints every 15–30 min and be able to resume; one owner per machine; keep a shared hours log.
- **Rough budget:** setup + speed test 10 h, drills 80 h, 11v11 80 h, H3 experiment 40 h, extra seeds 40 h, manager 20 h, reserve 30 h. Manager-on-rule-based-team and evaluations run on laptops.
- First measurement to make: **steps per second** on the cloud machine; it decides how many seeds and how long 11v11 training can be.

---

## 10. How we will judge success

**Hypotheses:**
- **H1:** players follow their duties, and switching tactics measurably changes where they play and win the ball.
- **H2:** a manager that changes instructions mid-match beats the best single fixed tactic on goal difference.
- **H3:** the early duty bonus speeds up learning.
- **H4 (stretch):** players trained on building blocks can carry out duties they never saw in training.

**Comparison ladder** (same compute, same opponents, 2–3 seeds, 100+ matches each):

| Version | What |
|---|---|
| B0 | No duties: plain MAPPO, goals only |
| B1 | Named duties |
| B2 | Building-block duties |
| M0 | Best fixed tactic |
| M1 | Manager picking presets |
| M2 | Manager directing individual players |

Pass marks fixed in advance: following duties may cost up to 10% win rate vs B0; B2 kept only if within ~5% of B1; M2 kept only if it beats M1 on varied opponents; the project's main claim holds if M1 or M2 beats M0. Every rung that fails is reported honestly, and the best version that held up is presented.

**Opponents must vary.** If the opponent never changes style, one tactic just wins and the manager learns nothing. P5's rule-based team must also play as opponents with different tactic sheets, sometimes switching mid-match. Also test against a human with a controller.

---

## 11. Known weak spots of the approach

Even if everything works:
- The manager can only use duties we teach; anything outside the list (planned overlaps beyond the script, baiting a press) is impossible.
- It reacts in ~10-second steps; the players must handle the moment.
- It only knows the opponent through the stats we measure.
- Rare duty combinations may be executed badly, and the manager might exploit them.
- Constant pressing tires players; fatigue is in the game but substitutions are archetype swaps only.
- Set pieces are planned (manager v3), not in v1.
- Individual skill (finishing, 1v1) isn't fixed by tactics.
- Humans will find tricks the team can't handle; report them as limitations.

---

## 12. Documents

| Document | Where | Notes |
|---|---|---|
| Training plan | `docs/training_plan.md` | The RL specification, most up to date |
| Observation spec | `docs/observation_spec.md` | Check §7.6 before using its velocity units |
| Scope briefing (PDF) | `docs/project_scope.pdf` | For teammates new to football/RL; dated 2026-11-06, still mentions 5v5 as a fallback (superseded by the 11v11 decision) |
| Duty lab results | `out/duty_lab/summary.md` + per-duty videos | Section 6 |
| Preliminary report | Claude Docs link (owner can share) | 2 pages for the course |
| Pitch deck and doc | Claude artifacts (owner can share) | One-off pitch to the professor, done; not maintained |

---

## 13. Open items

- Decide the PPO framework (PyTorch?) and add it to the report.
- Confirm the cloud platform and its session limits; measure steps per second.
- Build the rule-based team from the lab scripts + AI on the ball (P5).
- Duty lab, still to do: transition duties (counter-press, recover, break), set pieces, and optionally box arrival with varied crosses.
- Randomise archetype line-ups per match in training.
- The professor's comments on the preliminary report, once received.

---

## 14. Glossary

| Term | Meaning |
|---|---|
| **GRF** | Google Research Football, the game we use |
| **Agent** | An AI that makes decisions (a player or the manager) |
| **Policy** | The neural network that decides what an agent does |
| **Reinforcement learning (RL)** | Learning by trial and error from rewards |
| **PPO** | Proximal Policy Optimization, the standard RL algorithm we use |
| **MAPPO** | PPO for many agents: one shared policy for all players, plus a critic that sees everything during training |
| **Critic** | A second network that estimates how good a situation is, used only to train the policy |
| **Duty** | A simple job given to a player (press, mark, give width, ...) |
| **Duty check** | A function that says whether a player is doing his duty right now |
| **Shaping reward** | A small extra reward to help early learning (here: for doing your duty); faded out later |
| **Curriculum** | Training from easy to hard (drills → 11v11 vs easy AI → harder) |
| **Self-play** | Training against copies of our own team |
| **Behaviour cloning** | First training a network to copy a teacher, then improving it with RL |
| **Semi-MDP / options** | Decisions that last a while (the manager decides every ~10 s, players every frame) |
| **Seed** | A number that fixes the game's randomness; same seed = same match |
| **builtin_ai** | A command that hands a player to the game's own AI for one frame |
| **Archetype** | A player profile (target man, sweeper keeper, ...) with its own strengths |
| **Rule-based team** | A team run by hand-written scripts (our lab scripts + the AI on the ball) |
| **xG** | Expected goals: how likely a shot was to score |
