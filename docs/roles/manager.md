# Role: Manager (2 people)

*Owners: M1 ________ (project lead) · M2 ________ · Written 2026-11-20 from the team's planning so far. Background for everything here: [`docs/CONTEXT.md`](../CONTEXT.md).*

---

## Your job in one paragraph

You build and train **the manager**: the coach AI. Every ~10 seconds of game time (and at restarts) it looks at the match (score, clock, how the opponent plays, how our plan is working) and decides what each player should be doing: it loads a tactic, edits individual players' duties ("right-back: mark their #11") and nudges team dials (line height, pressing, width). The players carry the instructions out; the manager learns which instructions win matches. This is the project's headline result: **does a manager that changes instructions mid-match beat the best fixed tactic?**

---

## Where you fit in the final system

```
YOU: Manager AI  ──instruction block──►  Outfield network (10)  ─┐
  (every ~10 s,                     └──►  Goalkeeper network (1) ─┴─►  the game (11v11)
   restarts, big events)  ◄──────── match state, opponent stats, plan stats ──────────┘
```

- **Your output** is the instruction block every player reads: three duties per player (defending, attacking, transition), their targets (which opponent, teammate or zone) and six team dials. That format is the **contract** between you and the player groups.
- The manager is trained **with the players frozen**. Until real players exist, you train against the **rule-based team** (scripted duties + the game's AI on the ball), which follows any instruction perfectly. Then you retrain on the real players.

---

## Decisions already made that affect you

| Decision | What it means for the manager |
|---|---|
| **The manager directs players individually** (2026-11-11) | Not just one of six tactics: per-player duty edits + team dials. The six tactics (high press, low block, possession, counter, wing play, false nine) are starting points it can load. |
| **Built in steps** | v1: pick a preset (keep or one of 6). v2: edit 1-2 players per decision (which player → which duty → which target) + dial nudges. v3: archetype swaps (our "substitutions") and set-piece routines. Each step starts from the trained previous one; **v1 is the safety net**. |
| **Duties are building blocks** (2026-11-11) | New duties can be defined as recipes without new code. The manager picks from a **library of named duties** you write, not arbitrary recipes. |
| **Duty list for v1** | D1 hold zone, D2 press, D4 cover, D5 mark, D7 hold line, A1 hold shape, A2 support, A3 width, A4 run in behind, T2 recover. **Screen lane is out of v1** (2026-11-18): it lost to the game's AI in the lab. Use *press with a cover shadow* to cut off a midfielder instead. |
| **Duty lab results** (CONTEXT §6) | 18 of 21 duties work when scripted. Positional ones are reliable; pass-timing ones (third-man run, crosses) are weak. A lone press rarely reaches the ball: **pressing works as a team job**. Pinning their full-back is executed but doesn't keep him deeper. |
| **11v11 only** (2026-11-17) | All manager results are full matches. |
| **Reward on outcomes only** | Goal difference over the 120 s after a decision, a small xG term, the match result, a small cost per change. **No reward for picking the "right" counter.** |
| **The opponent must change style** | Against one fixed opponent one tactic just wins and the manager learns nothing. Training and evaluation need style-switching opponents. |
| **Shortcuts** (2026-11-18) | The rule-based team (lab scripts + AI on the ball) is your first training partner, the cloning teacher and a baseline. |
| **Evaluation ladder** | M0 best fixed tactic, M1 manager picking presets, M2 manager directing individuals. M2 is kept only if it beats M1 on varied opponents; the project claim holds if M1 or M2 beats M0. |

---

## M1 (project lead): tasks, in order

### A. Shared foundation (week 1, everyone waits on this)

1. **PPO core:** clipped policy loss, value loss, entropy bonus, GAE advantages, rollout buffer, update loop; mixed actions (discrete heads + a Beta distribution for kick power). Test on CartPole and a simple continuous task before football. Used by the manager, outfield and keeper groups.
2. **Instruction block contract** with O2 (outfield) and G (keeper): the duty list, targets, dials, and the keeper's fields. Freeze it in week 1.

### B. Football content (week 1-2)

3. **Duty definitions:** a short, unambiguous definition of every duty (e.g. "press = within 3 m and closing"). Sign off O2's duty checks against them.
4. **The six tactic sheets:** for each of the 11 positions, the defending, attacking and transition duty plus the dials. These are what the manager can load.
5. **The grammar of sensible duty combinations** (which duty can go with which target), so random training assignments aren't nonsense.

### C. The manager network (weeks 2-3)

6. **v1 head:** keep, or one of the 6 presets. 10 s commitment, decisions also at restarts.
7. **v2 heads:** which player (or none) → which duty → which target; plus ±0.2 on each dial; at most 1-2 edits per decision; a small cost per edit.
8. **Training:** stage 3 of the plan: first on M2's rule-based team, then on the trained players (frozen). Start v2 from the trained v1.

### D. Results (week 4)

9. With M2: the M0 / M1 / M2 comparison and readable decision logs for the presentation ("minute 63: told the right-back to man-mark their #11").

---

## M2: tasks, in order

### A. Manager environment (weeks 1-2)

1. **Manager env wrapper:** decision points (every ~10 s of game time, restarts, after a goal or big chance), action masks (one keeper, keeper-only duties for the keeper only, the 10 s commitment, the swap budget in v3), players frozen, rewards added up between decisions (semi-MDP).
2. **What the manager sees** (404 numbers, see `docs/training_plan.md` §3.2):
   - match state (score, time, phase),
   - our setup (per player: role, side, current duties, stamina, cards; team dials),
   - **opponent tendencies** over the last 30 s and 120 s: pass length, short-pass share, long balls, crosses, dribbles, time from regain to our third, how close they press, their line height, their possession,
   - **how our plan is working** over the last 120 s: shots and xG for and against, possession, territory, turnovers in our third, regains in their half, goals.
3. **Opponent-tendency extractor:** compute those stats from what is visible in the game (a pass = a kick received by a teammate). Validate it on recorded matches.

### B. Rule-based team and opponents (weeks 1-2: everyone needs these)

4. **Rule-based team:** the duty lab scripts off the ball (`grf_tools/duty_lab/duties.py`) + the game's AI on the ball. It must follow **any** instruction block. It is the manager's first training partner, the cloning teacher for the players, and a baseline.
5. **Script the six full tactics in the duty lab** from M1's sheets, plus the team dials (low vs high), and check each one visibly changes play (e.g. high press wins the ball higher than low block). This also tests H1 early.
6. **Style-switching opponents:** the rule-based team on the other side, given different tactic sheets, sometimes switching mid-match. Later: our own trained players given random presets (preset-conditioned self-play).

### C. Evaluation (week 3-4)

7. **H2 evaluation:** 100+ matches per version, 2-3 seeds, against the hard game AI and the style-switching opponents; manager vs every fixed tactic; does it react to a style switch within 60 s?
8. If outfield training is behind, **help O1**: that's where most training hours go.

---

## What you need from others, and what they need from you

| From / to | What | When |
|---|---|---|
| **M1 → everyone** | PPO core | Week 1 (critical path) |
| **M1 ↔ O2, G** | Instruction block contract | Week 1 |
| **M2 → O1, G** | Rule-based team (teacher, opponent, baseline) | Week 2 |
| O1 + G → you | Trained, frozen outfield + keeper networks | Week 3 |
| G → you | Cloud machines and hours log | Week 1 |

## Known weak spots to keep in mind

- The manager can only use duties we teach; anything outside the library is impossible.
- It reacts in ~10 s steps; it only knows the opponent through the stats we measure.
- Rare duty combinations may be carried out badly, and the manager might exploit them: keep the change cost and the combination grammar.
- Fatigue exists in the game; substitutions are archetype swaps only (v3).
