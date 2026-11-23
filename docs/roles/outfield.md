# Role: Outfield players (2 people)

*Owners: O1 ________ · O2 ________ · Written 2026-11-20 from the team's planning so far. Background for everything here: [`docs/CONTEXT.md`](../CONTEXT.md).*

---

## Your job in one paragraph

You build and train **the outfield network**: one AI shared by all 10 outfield players. Each player reads the match plus its instructions from the manager (its duties: e.g. *press*, *mark their #9*, *give width*) and learns, by trial and error, to play football and to carry those instructions out. You also build the **plumbing everyone uses**: what a player sees, how its decisions turn into game commands, the duty checks and the rewards.

---

## Where you fit in the final system

```
Manager AI  ──instruction block──►  YOU: Outfield network (10 players)  ─┐
                               └──►  Goalkeeper network (1)              ─┴─►  the game (11v11)  ──rewards──► back
```

- **One shared network** for all 10 outfield players. It knows its position from a role input and its job from the instruction block. During training a **critic** sees the whole match (MAPPO).
- Duties are **drawn at random** during training and re-drawn every 30-60 s, so the network learns to follow *any* instruction, which is what the manager needs.
- After you train it, it is **frozen** and the manager is trained on top of it.

---

## Decisions already made that affect you

| Decision | What it means for the outfield |
|---|---|
| **One shared network, not one per duty** | Basics learned once, smooth switching between duties, duties can combine. Train all drills mixed so no duty is forgotten. A per-duty network is only a debugging tool. |
| **11v11 only** for everything submitted (2026-11-17) | Small drills are fine as training steps; reported results are full matches. |
| **The keeper is a separate network** (2026-11-20) | You train the 10 outfield players; G trains the keeper. Both use your env wrapper. |
| **Duties are building blocks** (2026-11-11) | One generic duty check ("how far is the player from satisfying this recipe?") instead of 20 separate ones. |
| **v1 duties** | D1 hold zone, D2 press, D4 cover, D5 mark, D7 hold line, A1 hold shape, A2 support, A3 width, A4 run in behind, T2 recover. Screen lane is out of v1. |
| **Duty lab results** (CONTEXT §6) | 18 of 21 duties work when scripted: the checks are tested and the scripts can be copied as a warm start. Weak when scripted: screen lane, box arrival, third-man run, so expect those to be the hardest to learn. |
| **Outcome scoring for ball-winning duties** | Press, contain, double team: reward winning the ball and closing in, not only being in position, because their success ends the duty. |
| **Shortcuts** (2026-11-18) | **Rank 1:** early on, any of our players with the ball is handed to the game's AI; your network learns everything off the ball first, then takes the ball back. **Rank 2:** clone the lab scripts before PPO. |
| **Archetypes randomised** | Every match, draw which archetypes play and in which slot, or the network only learns one line-up. |
| **Comparison ladder** | B0 no duties, B1 named duties, B2 building-block duties; following duties may cost up to 10% win rate vs B0; B2 kept only if within ~5% of B1. |

**Lab findings your code must handle** (CONTEXT §7):
- **Possession blinks:** keep the last owner for up to 1 s of loose ball, or duty checks flicker.
- **Kick wind-up:** a kick takes about 8 frames; handing the player to the game's AI in that time cancels it. With shortcut 1, keep a player with our network until the ball has left his foot.
- **Action set:** use the full set (or extend `v3`): the defensive head needs pressure, team pressure and keeper rush.
- **Mirrored side:** for the right-hand team, "right" is forward; one wrapper makes both sides look like "we attack to the right".
- **Velocity units:** the lab measured `*_direction` ≈ movement over 0.25 s at 4 ticks per step; `docs/observation_spec.md` assumes per step at 10 ticks. Check before freezing the observation.
- **Rendering changes the game:** compare rendered runs only with rendered runs.

---

## O2: plumbing, duty checks, rewards, drills

### A. Env wrapper (week 1: everyone waits on this)

1. **Observation encoder:** 957 numbers per player (866 base from `docs/observation_spec.md` + 91 instruction block), normalised to roughly −1…1, metres first.
2. **Padding:** scenes with fewer than 11 players per side (academy drills) fill empty slots with zeros + an "absent" flag; the critic must cope too.
3. **Mirroring:** our network on either side sees "we attack to the right", and its actions are mirrored back.
4. **Possession memory** (1 s) in everything that asks "who has the ball".
5. **Action heads → game commands:** move direction (stop + 16), speed (walk / sprint / close control, sent as sticky toggles on change), ball action, defensive action, kick power; one button per step; masks (ball actions only near the ball, defensive actions only when they have it, keeper rush keeper only); the kick wind-up rule.
6. **Parallel games:** many games at once (one per CPU core), batching observations for the GPU.

### B. Duty checks and rewards (weeks 1-2)

7. **Duty checks:** port `grf_tools/duty_lab/duties.py` (each duty already has `ok`, `target` and the "closing in" rule) to work on the training observation; one generic check over the building blocks; unit tests.
8. **Rewards** (`docs/training_plan.md` §2.3): goals ±1, result, ball progression, shot quality, losing the ball in our third, plus the duty bonus while the check is true, **faded to zero by the end of stage 2**. Log every term separately.
9. **H3 experiment set-up:** the same training with and without the duty bonus.

### C. Drills (week 2)

10. **Stage-0 duty drills** (not built yet): 2v2 press + cover, 4v3 hold the line, 4v2 support rondo, 2v1 run in behind, 1v1 mark, recover after losing the ball. Register them like the lab scenes.
11. **Finish the duty lab:** script the transition duties (counter-press: balls won within 5 s; recover: back in position; break: forward runs after winning it), same method as the others.

---

## O1: training, curriculum, evaluation

### A. Training pipeline (weeks 1-2)

1. **MAPPO** on M1's PPO core: one shared actor for the 10 outfield players, a centralised critic, rollouts from O2's parallel games.
2. **Instruction sampler:** random duties per player from the v1 list (respecting M1's combination grammar), re-drawn every 30-60 s; sometimes the six presets.
3. **Archetype sampler:** random line-up from the squad every episode.
4. **Checkpoints** every 15-30 min, resumable; never debug on cloud time.

### B. Curriculum (weeks 2-3)

Built from what the game ships with plus ours (CONTEXT, and the list we made on 2026-11-20). Run the phases as **one mixed random pool**, full matches included early:

5. **Warm start (optional):** copy the rule-based team (lab scripts + AI on the ball) in 11v11 lab scenes and `11_vs_11_easy`.
6. **Phase A, off the ball (AI plays the ball):** `academy_single_goal_versus_lazy` → `11_vs_11_easy_stochastic`; no instructions; all outfield roles.
7. **Phase B, finishing:** `academy_empty_goal_close` → `empty_goal` → `run_to_score` → `run_to_score_with_keeper`; forwards, wingers, attacking mids.
8. **Phase C, combination play:** `pass_and_shoot`, `run_pass_and_shoot`, `3_vs_1_with_keeper`, `counterattack_easy` → `hard`; forwards and midfielders.
9. **Phase D, defending:** the academy drills with **our network on the defending side**, `drill_defend` (1v1), lab scenes `attack`, `build_up`, `high_line`; defenders and defensive mids.
10. **Phase E, set pieces:** `academy_corner` both ways.
11. **Stage-0 duty drills** (O2) in the pool once built.
12. **Phase F / stage 1a:** `11_vs_11_easy` → `stochastic` → `hard`, our network takes the ball back; no instructions at first, then random duties.
13. **Stage 1b / 2:** self-play against past versions; random mid-match tactic switches.
14. Joint matches with G's keeper network.

### C. Evaluation (weeks 3-4)

15. **Can they play?** Win rate and goal difference over 100 matches vs the game's AI at easy / medium / hard.
16. **H1:** share of time each duty check is true; do tactics visibly change play (where the ball is won, line height, width)?
17. **H3:** learning speed and final win rate with vs without the duty bonus.
18. **Ladder B0 / B1 / B2** with the same compute; several seeds each.

---

## What you need from others, and what they need from you

| From / to | What | When |
|---|---|---|
| M1 → you | PPO core | Week 1 |
| M1 ↔ O2 | Instruction block contract (duty list, targets, dials) | Week 1 |
| **O2 → everyone** | Env wrapper | Week 1 (critical path) |
| M2 → you | Rule-based team (cloning teacher, opponent) | Week 2 |
| G → you | Cloud machines, hours log | Week 1 |
| **You → M1, M2** | Trained, frozen outfield network | Week 3 |
| **You ↔ G** | Joint training with the keeper; one checkpoint format | Weeks 2-3 |

## Budget note

Most of the ~300 cloud GPU hours go to outfield training (CONTEXT §9: drills ~80 h, 11v11 ~80 h, H3 ~40 h, extra seeds ~40 h). Measure steps per second first; it decides how many seeds you can afford.
