# Training plan: players first, then the manager

Date: 2026-10-13. Supersedes the stage order in `manager_reward_shaping_plan.md` (that doc's reward and opponent-modelling detail still applies). PPO logic is the user's; plumbing, observations, drills and logging are Claude's.

## 0. Principles

1. **Two levels, one interface.** Players act; the manager only writes the **instruction block** that every player observes. The manager never touches actions.
2. **Train one level at a time.** Players learn with random instructions; then they are frozen and the manager learns on top.
3. **Tactics are built from duties.** Every tactic is an assignment of a few simple building blocks (press, cover, mark, hold the line, support, run in behind, ...) to players, plus a handful of team settings. Off the ball, duties are instructed; on the ball (pass, dribble, shoot) decisions are learned from outcomes and only shaped by team settings (risk, tempo).
4. **Only instruct what can be rewarded.** Every duty has a measurable test and a shaping reward; an input with no reward behind it is noise the policy learns to ignore.
5. **Rich description, coarse decisions.** The observation describes duties and targets in detail; the number of choices the networks must make starts small and grows (v1 → v2 → v3).
6. **Factorised actions.** Players decide *where to move*, *button action* and *how hard* as separate heads.
7. **Fixed shapes from day one.** All instruction slots exist in the observation from Stage 1 (unused ones are zero), so the network input never changes later.

Positions are in the team's own frame (attacks toward +x), metres, normalised as in `docs/observation_spec.md` (x / 55, y / 36). "Slot" = formation position 0-10 (0 = GK).

---

## 1. Building blocks (duties)

Each player holds one **defending duty**, one **attacking duty** and one **transition duty**. The active one depends on who has the ball. **v1** = enabled in the first version; the rest are reserved slots, enabled once v1 works.

### Defending duties (they have the ball)

| id | Duty | Target / setting | Test (shaping reward when true) | v1 |
|---|---|---|---|---|
| D0 | none | | | ✓ |
| D1 | Hold zone | anchor (x, y), radius | within radius of anchor (anchor shifts with the ball, see §2.4) | ✓ |
| D2 | Press | ball carrier, intensity | distance to carrier falling, or ≤ 2 m and engaging | ✓ |
| D3 | Contain | ball carrier | goal-side of ball, 1.5-3 m, no sliding | |
| D4 | Cover | a teammate (the presser) | between that teammate's opponent and our goal, 5-10 m behind him | ✓ |
| D5 | Mark | an opponent | goal-side of that opponent, ≤ 3 m | ✓ |
| D6 | Screen lane | an opponent | ≤ 2 m from the line between the carrier and that opponent | |
| D7 | Hold line | line height | x within ±2 m of the line, level with other D7 players | ✓ |
| D8 | Sweep (GK) | line height | between our line and goal, first to balls played behind it | |

### Attacking duties (we have the ball)

| id | Duty | Target / setting | Test | v1 |
|---|---|---|---|---|
| A0 | none | | | ✓ |
| A1 | Hold shape | anchor (x, y), radius | within radius of anchor shifted forward | ✓ |
| A2 | Support | ball carrier | 8-15 m from carrier with an open passing lane (no opponent ≤ 1.5 m from the line) | ✓ |
| A3 | Width | side | within 5 m of that touchline | ✓ |
| A4 | Run in behind | channel (y) | beyond the last defender, onside, moving toward goal | ✓ |
| A5 | Drop to receive | | between their midfield and defensive lines, ≤ 15 m from the carrier | |
| A6 | Attack zone | zone (x, y) | inside the zone when a cross or cutback is played | |
| A7 | Rest defence | zone (x, y) | behind the ball and inside the zone | |

### Transition duties (first 5 s after possession changes)

| id | Duty | Test | v1 |
|---|---|---|---|
| T0 | none (fall into the next duty) | | ✓ |
| T1 | Counter-press | presses the new carrier within 5 s of losing the ball | |
| T2 | Recover | sprinting toward the defensive anchor after losing the ball | ✓ |
| T3 | Break | sprinting forward after winning the ball | |

### Team settings (shared, [0, 1])

| Setting | Low | High | v1 |
|---|---|---|---|
| line_height | deep | high | ✓ |
| press_trigger | press only on bad touches / back passes | press always | ✓ |
| width | narrow | wide | ✓ |
| tempo | slow build-up | fast | |
| risk | safe passes | risky passes and shots | |
| compactness | stretched | compact between lines | |

### Set plays (the same duties, with timing)

At a restart each player gets a **set-play duty**: deliver (taker, to a zone), attack zone (arrive at a zone at time t), rest defence, mark, hold zone. A "routine" is just one assignment of these.

### Example tactics as duty assignments

| Tactic | Assignment |
|---|---|
| High press | ST: D2 press; nearest CM: D4 cover the ST; other CMs: D5 mark their CMs; CBs: D7, line 0.8; press_trigger 0.9 |
| Low block | all: D1 hold zone, line 0.15, press_trigger 0.2, width 0.3 |
| Possession | CMs: A2 support; FBs: A3 width; CBs: A1 hold shape; risk 0.3 |
| Counter | ST, W: A4 run in behind; T3 break; line 0.3 |
| Wing play | FBs: A3 width; W: A4 run in behind; ST: A6 attack box |
| False nine | ST: A5 drop to receive; W: A4 run in behind |
| Sweeper keeper | GK: D8 sweep; CBs: D7, line 0.75 |

---

## 2. Player level

### 2.1 Player state (observation)

`docs/observation_spec.md` base (866) **plus** the instruction block:

| Block | Size | Contents |
|---|---|---|
| I1 Role | 14 | role family one-hot (GK, CB, FB, DM, CM, AM, W, ST, none: 9), side one-hot (3), slot / 10 (1), set-piece taker flag (1) |
| I2 Duties | 21 | defending duty one-hot (D0-D8: 9), attacking duty one-hot (A0-A7: 8), transition duty one-hot (T0-T3: 4) |
| I3 Duty targets | 29 | opponent target one-hot (11 + none: 12), teammate target one-hot (11 + none: 12), zone / anchor (x, y) (2), radius (1), line height (1), press intensity (1) |
| I4 Team settings | 6 | line_height, press_trigger, width, tempo, risk, compactness |
| I5 Set play | 15 | restart type one-hot (kickoff, goal kick, free kick attack, free kick defend, corner attack, corner defend: 6), set-play duty one-hot (deliver, attack zone, rest defence, mark, hold zone: 5), zone (x, y) (2), arrival time (1), time since restart (1) |
| M Memory | 6 | last 3 button-action classes (none/pass/shot/defend), steps since my last touch, steps since possession changed, instruction changed in the last 5 s |

**Total: 866 + 91 = 957.** Duties not yet enabled stay zero. Critic extras as in `observation_spec.md`, plus every player's instruction block.

### 2.2 Player actions (factorised heads)

Sent to the env as the existing tuple `(action, power, dx, dy)`; no engine change needed.

| Head | Type | Values | Maps to |
|---|---|---|---|
| A. Move direction | categorical (17) | idle + 16 directions | `dx, dy` |
| B. Move speed | categorical (3) | walk, sprint, close control | sticky sprint / dribble toggles, sent on change |
| C. Ball action | categorical (8) | none, short, high, long, shot, curl shot, chip, through ball | button action |
| D. Defensive action | categorical (5) | none, pressure, team pressure, sliding, keeper rush | button action |
| E. Kick power | Beta on [0, 1] | continuous | `power` |

One button per step: C wins if not "none" and the player can kick, otherwise D. Masks: C only with the ball near (≤ 2 m) or coming to me; D only when the opponent has the ball; keeper rush GK only; E counted only when C is a kick.

### 2.3 Player reward

Team-shared unless marked *(own)*. Shaping is annealed toward 0 by the end of Stage 2.

| Term | Weight | Definition |
|---|---|---|
| Goal | ±1 | score change |
| Match result | +1 / 0 / -1 | final whistle |
| Ball progression | 0.002 / m | possession-team gain in x while we keep the ball |
| Shot quality | 0.05 × xG-proxy | per own shot (distance/angle table) |
| Turnover in own third | -0.03 | |
| Active duty satisfied *(own)* | 0.002 / step | the active duty's test (§1) is true |
| Set-play duty *(own)* | 0.05 | in the zone at the arrival time / delivery reaches the zone |

### 2.4 Anchors

A hold-zone anchor is not a fixed point: it shifts with the ball (anchor_x + 0.35 × (ball_x - anchor_x) in possession, smaller out of possession) and is clipped by line_height for defenders. Formation = the 11 anchors.

---

## 3. Manager level

### 3.1 When it acts

Every 10 s of game time, plus at restarts, goals, red cards and half time. A change must be kept ≥ 10 s (masked). Set-play choices only at restarts.

### 3.2 Manager state (v1)

| Block | Size | Contents |
|---|---|---|
| S1 Match context | 14 | score diff (clip ±3), time left, half, added-time flag, game mode one-hot (7), players on pitch ×2, time since my last change |
| S2 Own setup | 11 × 32 + 6 = 358 | per slot: role family (9), side (3), defending duty (9), attacking duty (8), stamina used, yellow card, on pitch; plus team settings (6) |
| S3 Opponent tendencies | 10 × 2 + 2 = 22 | over 30 s and 120 s windows: mean pass length, short-pass share, central-pass share, long-ball share, crosses / min, dribbles / min, seconds from regain to our final third, their press distance to our carrier, their line height, their possession share; plus an event count per window (confidence) |
| S4 Plan effectiveness | 10 | last 120 s: shots for/against, xG-proxy for/against, possession, territory, own-third turnovers, regains in their half, goals for/against |

**Total: 404.** Opponent features come from observable events only (pass = kick to controlled reception by a teammate); more features (switches of play, channels, carries) are added in v2 once the extractor is validated on recorded games.

### 3.3 Manager actions

| Version | Heads | Values |
|---|---|---|
| **v1** | tactic preset | keep, or one of 6 full assignments (high press, low block, possession, counter, wing play, false nine): duties for all 11 slots + team settings |
| v2 | slot edit | slot (none / 0-10) → defending duty (9), attacking duty (8), target (opponent or teammate, 12) |
| v2 | team settings tweak | ±0.2 on each of the 6 settings |
| v3 | archetype swap | slot + archetype (19), at restarts only, max 5 per match |
| v3 | set-play routine | one of 4 set-play assignments per restart type + taker slot |

Masks: one GK; GK-only duties (D8) only for slot 0; 10 s commitment; swap budget.

### 3.4 Manager reward

| Term | Definition |
|---|---|
| Window goal difference | goals for - against over the 120 s after a decision |
| xG difference | 0.2 × (our xG-proxy - theirs) over the window |
| Match result | +1 / 0 / -1 |
| Change cost | -0.01 per change (no thrashing) |

No reward for picking a "correct" counter; outcomes only.

---

## 4. Stages and exit criteria

| Stage | Trains | Opponents | Instructions | Exit when |
|---|---|---|---|---|
| 0. Duty drills | player policy | scripted / built-in AI in small drills | one duty at a time: press + cover 2v2, hold-the-line 4v3, support rondo 4v2, run-in-behind 2v1, mark 1v1, recover after turnover | each v1 duty test true ≥ 70 % of drill time |
| 1a. 5v5 basics | player policy | built-in AI easy → hard | random duty assignments (v1 duties), re-drawn every 30-60 s | beats built-in hard AI over 100 games (mid-November go/no-go) |
| 1b. 11v11 basics | player policy | built-in AI + past snapshots | random assignments incl. the 6 presets | beats hard AI; duties satisfied ≥ 70 % |
| 2. Tactics are real | player policy | + preset-conditioned self-play | random mid-match preset switches, set-play drills, later v2 duties | switching preset measurably changes play (e.g. false nine avg x drops ≥ 8 m; high press regains in their half ×2) with ≤ 10 % win-rate loss |
| 3. Manager v1 | manager (players frozen) | pool: hard AI, preset-conditioned self-play that switches presets mid-match, imitation model of the user | manager-chosen preset | beats every fixed preset on the pool; reacts to an opponent style switch within 60 s |
| 3b. Manager v2-v3 | manager | same + held-out styles | slot edits, settings, swaps, set plays | beats manager v1 |
| 4. Fine-tune (optional) | both, small learning rates | same | manager-chosen | no regression on held-out opponents |

**Preset-conditioned self-play:** the opponent is the same player policy given a random preset that changes mid-match; the main source of varied opponents for the manager.

**Imitation opponent:** gamepad matches already log every input (replay system); a behaviour-cloning model trained on them gives a "plays like the user" opponent.

---

## 5. Build order

| # | Item | Owner |
|---|---|---|
| 1 | Duty tests (§1) as functions of the raw observation, with unit tests | Claude |
| 2 | Instruction block in the observation, presets, random assignment sampler | Claude |
| 3 | Factorised action adapter + masks | Claude |
| 4 | Reward module, every term logged separately | Claude |
| 5 | Duty drills (stage 0) and 5v5 scenario | Claude |
| 6 | Parallel env workers, logging, checkpoints | Claude |
| 7 | PPO: heads, GAE, clipped loss, update loop, MAPPO critic | **user** |
| 8 | Opponent-tendency extractor (S3) + plan stats (S4) | Claude |
| 9 | Preset-conditioned opponent wrapper, snapshot pool | Claude |
| 10 | Manager env wrapper (decision points, masks, frozen players) | Claude |
| 11 | Manager PPO | **user** |
| 12 | Behaviour-cloning opponent from gamepad logs | Claude scaffold, user trains |

## 6. Known limits

- The engine has no formation / role / pressing commands: duties exist only through the instruction block and duty rewards. Stage 2's exit test checks they actually change play.
- Set plays are rare in normal matches; they need restart drills.
- Pass/reception events are inferred from observations, so tendency features are noisy; the manager also gets event counts as confidence.
- All thresholds, weights and preset values are starting points to tune.
