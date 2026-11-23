# Role: Goalkeeper network (1 person)

*Owner: ________ · Written 2026-11-20 from the team's planning so far. Background for everything here: [`docs/CONTEXT.md`](../CONTEXT.md).*

---

## Your job in one paragraph

You build and train **the goalkeeper's own network**: a separate, smaller AI that plays in goal for our team. It learns where to stand, how to stop shots, when to come off its line, and how to restart play. It takes instructions from the manager like every other player (for example "sweep behind a high line") and plays alongside the outfield network in full 11v11 matches. You also run the cloud machines and make the demo clips.

---

## Where you fit in the final system

```
Manager AI  ──instructions──►  Outfield network (10 players)  ─┐
   (every ~10 s)          └──►  Goalkeeper network (YOU)     ─┴─►  the game (11v11)  ──rewards──► back to all
```

- The **manager** sends every player, keeper included, an instruction block: its duties (for the keeper: e.g. *sweep*, *hold position*), the team's line height, and so on.
- The **outfield network** controls the 10 outfield players. **Your network** controls the keeper. Both read the same kind of observation and send the same kind of action, so they slot into one team.
- **Training order:** first you train the keeper with the game's own AI playing the outfield; then the keeper and the outfield network play full matches together; then the manager is trained on top of both (players frozen).

---

## Decisions already made that affect you

| Decision | What it means for the keeper |
|---|---|
| **11v11 only** for everything submitted (2026-11-17) | Final keeper results come from full matches. Small drills are fine for training. |
| **Separate keeper network** (2026-11-20 split) | You own it end to end. It does not share weights with the outfield network, so it needs its own training data. |
| **Same observation and action format as the outfield** | You use the outfield group's env wrapper (O2) rather than building your own. Agree any keeper-only additions with them. |
| **Full action set** (lab finding) | The frozen `v3` action set has no *keeper rush*. Training must use the full set or extend `v3`. |
| **Archetypes are randomised** | The squad has **two keeper archetypes**. Train on both, every match drawn at random, so the network handles either. |
| **Sweeping (D8) works when scripted** (duty lab) | Scripted keeper stands 11.5 m off his line behind a high line (vs 4.3 m for the game's AI) and is in position 69% of the time (vs 19%). Use it as a check and as a teacher. |
| **Shortcuts** (2026-11-18): AI plays the ball early on; clone the lab scripts first | Early on, distribution can be left to the game's AI while the keeper learns positioning and shot-stopping. Cloning a scripted keeper is a possible warm start. |
| **Kick wind-up trap** (lab finding) | A kick takes about 8 frames. Handing the keeper to the game's AI during that time cancels the kick. Matters for goal kicks and distribution. |
| **Possession memory** (lab finding) | The game briefly reports nobody on the ball between a dribbler's touches. Use the shared possession memory in your checks and rewards. |
| **Mirrored side** (lab finding) | When our team plays on the right, controls are mirrored. The shared wrapper handles it; don't bypass it. |
| **Rendering changes the game** | Compare rendered runs only with rendered runs. Use `--no-video` for numbers. |

---

## Your tasks, in order

### A. Find out what the keeper can do (week 1)

1. Check what a keeper controlled by our network can actually do in the engine: does he dive and save on his own once in position, or only when sent a command? What does *keeper rush* do exactly? Write it down; everything else depends on it.
2. Check what the keeper sees in the observation (ball height and speed matter for shots and crosses) and agree any keeper-only features with O2.
3. Agree the keeper's part of the instruction block with M1 (manager): which duties a keeper can be given (*hold position*, *sweep* with a line height, set-piece positions) and the line-height dial.

### B. Script the keeper duties in the duty lab (week 1-2)

The lab is `grf_tools/duty_lab/` (see CONTEXT §6). Sweep is done. Add:

4. **Shot positioning:** stand on the line between the ball and the centre of the goal, a few metres off the line. Check: within 1 m of that spot when they shoot.
5. **Coming for crosses:** claim a cross into the six-yard box when he gets there first. Scored on crosses claimed vs the game's keeper.
6. **Distribution:** after a save or a goal kick, pass short to a free defender or long to a forward. Scored on passes reaching a teammate.
7. **Set-piece positions:** where to stand for corners and free kicks.
8. Run each before/after on 12 seeds like the other duties and record the results in `out/duty_lab/summary.md`.

### C. Training environments (week 2)

The game has **no keeper drills**, so build them from what exists, with our keeper on the **defending** side:

9. Academy drills played the other way round: `academy_run_to_score_with_keeper`, `academy_pass_and_shoot_with_keeper`, `academy_run_pass_and_shoot_with_keeper`, `academy_3_vs_1_with_keeper`, `academy_counterattack_easy/hard`, `academy_corner`: the game's AI attacks, our keeper defends.
10. Shooting drills turned around: `drill_shot`, `drill_pressure`, `drill_box` with the game's AI shooting at our keeper.
11. `high_line` lab scene for sweeping; full 11v11 (`11_vs_11_easy` → `hard`) for everything together.
12. Mix all of these into one random pool so nothing is forgotten.

### D. Reward (week 2)

13. Main terms: goals conceded (negative), saves (a shot on target that doesn't go in), crosses claimed, sweeper clearances, distribution that reaches a teammate.
14. A small bonus while the keeper's duty check is true, faded to zero over training (same rule as the outfield).
15. Log every reward term separately.

### E. Training (weeks 2-3)

16. Optional warm start: copy the scripted keeper (behaviour cloning).
17. Train with the PPO core (M1), first with the game's AI as the outfield, then together with the outfield network (coordinate with O1).
18. Randomise the keeper archetype every episode.
19. Save checkpoints in the agreed format; never debug on cloud time.

### F. Evaluation (week 3-4)

20. Against the game's own keeper in the same scenes and seeds: goals conceded per shot faced, save rate, crosses claimed, distribution completed, sweep events.
21. Duty compliance: share of time the keeper's duty check is true.
22. A few matches against a human with a controller.

### G. Shared jobs you own

23. **Cloud machines:** setup script, steps-per-second measurement, checkpoint/resume, auto-shutdown, and the **shared GPU-hours log** (about 300 h in total; see CONTEXT §9).
24. **Demo and clips** for the presentation (`elclasico.sh --watch`, duty lab videos, trained-team clips).

---

## What you need from others, and what they need from you

| From | What | When |
|---|---|---|
| M1 | PPO core | Week 1 |
| M1 | Keeper fields in the instruction block | Week 1 |
| O2 | Env wrapper (observation, actions, padding, mirroring, possession memory) | Week 1 |
| O1 | Joint training with the outfield network; checkpoint format | Week 2-3 |
| **You → everyone** | Cloud machines working, hours log, speed measurement | Week 1 |
| **You → M2** | A trained keeper for the rule-based team and manager training | Week 3 |

## Open questions to settle early

- What does the engine do automatically for a controlled keeper (task 1)?
- Does the keeper need any observation features the outfield doesn't?
- How much distribution should the keeper learn vs leave to the game's AI in the first version?
