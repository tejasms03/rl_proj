# Agent observation spec (v1)

One observation per controlled player, built from GRF's `raw` observation each step. Principle (decided 2026-10-09): **anything a human sees on screen**. Both teams' positions and movement, the ball, roles, cards, stamina, and player ratings. The actor sees the same thing in training and in a real game. Things a human can't see go to the critic only.

## Conventions

- **Metres first.** Raw positions are in odd units: `x_m = x · 54.4`, `y_m = y · −83.6`, `z` already in metres. Raw velocities (`*_direction`) are per step (0.1 s at the default 10 ticks per step), so `v_m/s = Δ_m / 0.1`.
- **Attack direction.** Every player's observation is "my team attacks toward +x". GRF already mirrors raw observations for agents playing on the right, so with self-play nothing extra is needed. Just never mix the two frames.
- **Normalization targets roughly [−1, 1]:** positions ÷ (55, 36) (half pitch length and width); player speed ÷ 10; ball speed ÷ 30; heights ÷ 5.
- **Categories are one-hot.** Roles (10): GK, CB, LB, RB, DM, CM, LM, RM, AM, CF. Game mode (7): normal, kickoff, goal kick, free kick, corner, throw-in, penalty.
- **Attributes:** the 22 stats are already in 0–1; height → `(h − 1.80) / 0.15`; preferred foot → one-hot (unspecified, right, left).
- **"Relative"** means `other − self`, in metres, then normalized like positions.

## Actor observation

### Self — 55

| Feature | Size | Source | Normalization |
|---|---|---|---|
| position (x, y) | 2 | `left_team[i]` | ÷ (55, 36) |
| velocity (vx, vy) | 2 | `left_team_direction[i]` | m/s ÷ 10 |
| speed | 1 | from velocity | ÷ 10 |
| stamina used | 1 | `left_team_tired_factor[i]` | as is (0 fresh → 1 tired) |
| yellow card | 1 | `left_team_yellow_card[i]` | 0/1 |
| on the pitch | 1 | `left_team_active[i]` | 0/1 |
| role | 10 | `left_team_roles[i]` | one-hot |
| attributes: 22 stats, height, foot | 26 | `left_team_attributes[i]` | see conventions |
| has the ball | 1 | `ball_owned_team == 0 and ball_owned_player == i` | 0/1 |
| held inputs (8 directions, sprint, dribble) | 10 | `sticky_actions` | 0/1 |

### Ball — 13

| Feature | Size | Normalization |
|---|---|---|
| position (x, y, z) | 3 | ÷ (55, 36, 5) |
| relative to self (dx, dy) | 2 | ÷ (55, 36) |
| distance to self | 1 | m ÷ 55 |
| velocity (vx, vy, vz) | 3 | m/s ÷ (30, 30, 10) |
| owner: my team / opponents / nobody | 3 | one-hot |
| owned by me | 1 | 0/1 |

### Teammates — 10 × 47 = 470

One row per teammate (excluding self). Treat as a **set** (shared per-player encoder, e.g. attention), with a mask for players who are off the pitch.

| Feature | Size | Normalization |
|---|---|---|
| relative position (dx, dy) | 2 | ÷ (55, 36) |
| distance to self | 1 | m ÷ 55 |
| position (x, y) | 2 | ÷ (55, 36) |
| velocity (vx, vy) | 2 | m/s ÷ 10 |
| role | 10 | one-hot |
| on the pitch, yellow card, stamina used | 3 | as above |
| has the ball | 1 | 0/1 |
| attributes: 22 stats, height, foot | 26 | see conventions |

### Opponents — 11 × 28 = 308

Same idea as teammates, with **key attributes only** (what you'd read off a player card). The full ratings go to the critic.

| Feature | Size | Normalization |
|---|---|---|
| relative position (dx, dy) | 2 | ÷ (55, 36) |
| distance to self | 1 | m ÷ 55 |
| position (x, y) | 2 | ÷ (55, 36) |
| velocity (vx, vy) | 2 | m/s ÷ 10 |
| role | 10 | one-hot |
| on the pitch, yellow card, stamina used | 3 | as above |
| has the ball | 1 | 0/1 |
| distance to the ball | 1 | m ÷ 55 |
| key attributes: velocity, acceleration, balance, standing tackle, defensive positioning, height | 6 | stats as is, height as in conventions |

### Context — 20

| Feature | Size | Normalization |
|---|---|---|
| opponent goal relative to self (dx, dy) | 2 | ÷ (55, 36) |
| distance to opponent goal | 1 | m ÷ 55 |
| angle to opponent goal (sin, cos) | 2 | as is |
| own goal relative to self (dx, dy) | 2 | ÷ (55, 36) |
| score difference (mine − theirs) | 1 | clip ±5, ÷ 5 |
| time left | 1 | `steps_left` ÷ 3000 |
| game mode | 7 | one-hot |
| pitch condition (normal, dry, wet, long grass) | 4 | one-hot (from the env config) |

**Actor total: 55 + 13 + 470 + 308 + 20 = 866 numbers per player.**

## Critic extras (training only)

The critic can see everything, because it is discarded after training:
- ball spin (`ball_rotation`, undo the position scaling) — 3
- full opponent attributes (all 22 stats, height, foot) instead of the 6 key ones — +20 per opponent
- (optional) a team-level summary: every player's position and velocity in one fixed-order vector, so the critic judges the whole situation at once

## Reserved for later phases

- Manager inputs (phases 3–4): current tactic / style one-hot, each player's archetype one-hot
- Chemistry (phases 3–4): per-teammate chemistry score in the teammate rows

## Checks to unit-test

- Shapes: 11 observations × the sizes above; masks match `*_team_active`
- Ranges: everything roughly within [−1.5, 1.5]
- Mirroring: a position mirrored across the pitch gives the mirrored relative features
- A known situation (e.g. the kickoff of `academy_empty_goal_close`) gives hand-computed numbers
- "Has the ball" agrees between self, ball and teammate rows
