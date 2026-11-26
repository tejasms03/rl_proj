# Adaptive manager and reward-shaping plan

Date: 2026-10-12
Status: design proposal; PPO implementation belongs to the user.

## 1. Objective and scope

Build a coordinated 11-player team that wins against strong human opponents, including opponents who change tactics during a match. Universal human superiority is an aspiration, not a guarantee. Demonstrate strength with held-out human matches and diverse opponent policies.

The manager remains responsible for team playstyle, tactical player assignments and swaps, and attacking/defensive set-play routines. It adapts these decisions to tendencies inferred from observable opposition play. The players learn to execute instructions while retaining enough flexibility to exploit opportunities and cover emergencies.

This document specifies interfaces, rewards, experiments, and implementation order. It does not implement PPO or prescribe every optimizer hyperparameter.

## 2. Architecture and boundaries

### Player policy

Start with one shared policy conditioned on player identity/role, physical attributes, and manager instructions. Use separate categorical action and kick-power outputs, respecting the existing v3 actions and KickPowerWrapper interface. Define the power distribution and likelihood explicitly when implementing PPO; evaluate power likelihood only under the chosen action convention and keep that convention consistent in storage and optimization.

Use the existing actor observation design in docs/observation_spec.md as the base: 869 features per player before manager inputs (960 with the 91-number instruction block). Add the assigned tactical role, team playstyle, formation slot/anchor, pressing duty, and active set-play assignment/phase. Include recent observable context through a short history or recurrent state when needed. Mask inactive players and invalid actions.

A centralized critic may use joint team state and the existing training-only extras. Actor observations during training must match deployment availability. Opponent intent, future actions, and hidden controller inputs cannot enter the deployed manager or player policy.

Shared rewards and a centralized critic are the initial coordination mechanism. Do not begin with 25 independently trained role networks.

### Manager policy

Inputs:
- Observable match state, score, time, fatigue, cards, possession and restart ownership.
- Own current formation, assignments and playstyle.
- Rolling opposition tendencies, uncertainty, and the recent effectiveness of current tactics.

Outputs:
- Playstyle: balanced, controlled possession, direct progression, counterattack, or protect lead.
- Team parameters: defensive width/depth, compactness, pressing intensity and trigger, attacking width, buildup risk, and transition commitment.
- Tactical role assignments and position-slot swaps, constrained to valid lineups.
- Set-play routine and participant assignments when a restart occurs.

Begin with a small discrete menu of valid tactical packages; expand to parameterized choices after the basic system works. Manager outputs condition learned player behavior. The engine does not currently expose a complete formation/pressing command API; do not assume these directives execute themselves.

Initially run manager decisions every 10 seconds of simulation time, with a minimum 10-second commitment. Allow decisions at restarts, half time, goals, red cards, and sustained transitions in opponent behavior. These intervals are starting hypotheses to tune, not established optimal values. A per-step classifier update does not require a per-step tactical switch.

Keep three concepts separate:
1. Physical archetype: the repo's 19 equal-budget attribute profiles.
2. Tactical role: what a player is instructed to do in the current plan.
3. Team playstyle: the coordinated behavior across roles.

Default human benchmark: physical attributes remain fixed during a match; tactical assignments can change. Existing mid-game stat/archetype changes are available for a separately labeled manager experiment. If enabled, enforce the same legal stat budget, immutable height, a fixed swap schedule, and opponent access to equivalent capabilities. Do not improve results by silently increasing player ability.

## 3. Inferring and countering human playstyle

Estimate tendencies from completed observable sequences, not a permanent human label. Track multiple tendencies simultaneously:
- Short-pass frequency and distance, central-pass share, return passes and multi-player combinations.
- Forward progression speed after possession recovery and direct-ball frequency.
- Wide progression, switches of play, crosses, and cutbacks.
- Dribble frequency, carry distance, and attempted penetration channels.
- Pressing proximity, pressure after turnovers, and defensive line height.

Define pass/carry events in metres and simulation seconds. A pass is attributed from a kick/release through a subsequent controlled reception; deflections and loose-ball intervals need explicit handling. When engine event hooks are absent, log uncertainty rather than pretending possession changes identify every event perfectly.

Maintain a recent 30-second window and a slower 120-second window, with event counts and confidence. Use a neutral prior when little evidence exists. Thresholds and window lengths require calibration on recorded games. Feed features and confidence into the manager; descriptive labels are useful diagnostics, not privileged truth.

Example adaptive responses:

| Observed tendency | Candidate response | Coverage obligation |
|---|---|---|
| Central tiki-taka combinations | Narrow midfield, screen central lanes, use coordinated pressure on a poor touch/back pass | Retain wide cover and a defender behind the press |
| Wide switches/crosses | Increase wide coverage, coordinate fullback/winger pressure, cover the box | Avoid opening the central cutback lane |
| Fast direct counters | Lower transition risk, retain defensive cover, protect space behind the line | Preserve an outlet rather than surrendering all attacking threat |
| Repeated central dribbles | Contain with one player and cover with another | Avoid multiple defenders chasing the same carrier |
| High aggressive press | Offer support triangles and test direct outlets behind pressure | Do not force risky short buildup every time |

These are candidate choices to test, not guaranteed counters. For example, detecting tiki-taka does not automatically prove that narrowing is best. Learn/select responses using subsequent team outcomes and held-out evaluation. Opponents can switch styles or deliberately bait a response.

Use confidence thresholds, minimum evidence, and hysteresis to prevent oscillation. Log why the manager changed tactics, what it selected, and how the following period performed. Never reward the manager simply for assigning a style label or selecting the expected counter.

## 4. Reward objective

### Final objective: match outcome

For full-match competitive training, define base team reward as:
- During play: 0.
- True full time: +1 win, 0 draw, -1 loss.

This directly targets winning rather than maximizing goal margin. Broadcast the same team return to all active players; broadcasting is not summing 11 rewards into the team's scoreboard objective.

The engine supplies goal-difference increments (+1/-1 on goals). Keep those as logged events and as a useful early curriculum objective, not automatically as an additional final reward. Explicitly switch from curriculum goal-based tasks to the full-match objective. Do not confuse a goal with episode termination.

Winning can rationally involve retaining a lead. Avoid arbitrary possession, backward-pass, or time-use penalties that conflict with that objective. Apply the game's actual rules and benchmark any deliberate engine exploits separately.

### Discounting

Specify discount in simulation time: gamma(dt) = exp(-dt / tau). Choose tau by examining full-match effective horizon and sensitivity, rather than copying a per-step gamma from a short academy drill. A short horizon can nearly erase full-time feedback.

With fixed episode length and gamma=1, the base return is exactly match outcome. With discounting and variable match lengths, earlier terminal outcomes receive different weight; report that distinction. GAE lambda and rollout length also need a long-horizon sensitivity experiment.

### Default shaping: a shared bounded potential

Use:

    r_train(t) = r_base(t) + gamma(dt) * Phi(s_next) - Phi(s_current)

Start with Phi=0 as the mandatory baseline. Introduce one team potential at a time. Recommended first experiment:

    Phi(s) = 0.10 * [P(win | s) - P(loss | s)]

This gives Phi in [-0.10, +0.10]. A win-probability model is optional infrastructure, not something assumed to exist. Train it on diverse completed match data, validate calibration on held-out opponents and score/time situations, freeze it within a training stage, and version every checkpoint. Skip this potential until there is credible data.

A simpler optional first potential is:

    Phi(s) = 0.05 * possession_sign(s) * progress(s)

where possession_sign is +1 for own controlled possession, -1 for opponent controlled possession, and 0 for an unresolved loose ball. progress is ball advancement from the possessing team's own goal toward the other goal, mapped to [0,1]. Use metres, clamp pitch bounds, and define controlled-possession detection explicitly. This is a coarse teaching signal; it does not measure chance quality or tactical safety.

Use true next state, set Phi=0 at every genuine terminal state, and bootstrap at time-limit/rollout truncations. Do not reset potential at a pass, turnover, goal, or manager switch unless that is an actual terminal boundary. With matching discounts and correct terminal treatment, the discounted potential terms telescope. Bounded potential does not imply a bound on the undiscounted sum of positive shaping events.

Potential shaping preserves the underlying objective under its theoretical assumptions; it does not guarantee neural training success, human superiority, or robustness in changing multi-agent training. Do not clip individual shaping differences or dynamically change Phi mid-trajectory and then claim the same guarantee. Freeze/version potentials and recompute consistently if a stage changes them.

### Signals excluded from the default competitive reward

No standing-in-zone rewards, per-step sprint/marking/proximity bonuses, pass-count bonuses, shot-attempt bonuses, automated counter-choice bonuses, or possession-cycle reward caps. No retrospective rewriting of a player's reward history to punish a set-play failure.

Keep tackles, saves, passing, shot quality, shape, and role compliance as diagnostics. Optional drill rewards or role auxiliary losses are curriculum aids; document their changed objective and anneal them to zero for final competitive training. Tactical instructions are soft: allow a player to leave an anchor to stop a goal or exploit an opening.

## 5. Tactical roles and swaps

Start with six tactical families: keeper, central defender, wide defender, holding midfielder, creator/wide attacker, and striker. Add variants progressively: sweeper keeper, ball-playing/stopper CB, attacking/inverted/defensive fullback, ball winner/regista, box-to-box/mezzala, winger/inside forward, target man/poacher/false nine.

Retain the original document's richer role catalogue as future variants, not 25 separate reward functions. A manager may change a player's tactical role or exchange formation slots when permitted. Validate uniqueness, goalkeeper assignment, inactive-player masks and positional suitability. Include the new assignment in observations immediately and reset any assignment-specific recurrent bookkeeping explicitly.

Use phase-dependent soft anchors relative to teammates, ball and opponent threat. Do not hard-lock coordinates. Evaluate whether swaps improve match outcomes, not whether the agents satisfy a static football stereotype.

## 6. Manager-directed set plays

Correct engine modes: normal=0, kickoff=1, goal kick=2, free kick=3, corner=4, throw-in=5, penalty=6. Mode alone does not identify which team owns the restart; inspect restart/possession information and validate it against recorded transitions.

Retain the original routine choices:
- Attacking corners: short combination or near-post delivery with designated receiver/runners.
- Defensive free kicks: compact block/marking or a legal coordinated offside-trap attempt.
- Goal kicks: short buildup to ball-playing defenders or direct delivery toward a target man.
- Throw-ins: safe support combination or progressive option when a lane exists.

Routine state machine: selected -> setup -> restart taken -> continuation -> complete/abort. Completion follows the ball/event sequence rather than a universal three-second timer. Add explicit setup and continuation timeouts to prevent deadlocks. Exit on opposing controlled possession, incompatible restart, or timeout; preserve normal emergency defending throughout.

The manager assigns taker, receiver, runners, blockers/markers where legal, and defensive cover. Policy inputs include assignment and phase. Never move players/ball by teleportation or force opponent actions.

A trap is a high-risk option, not mandatory behavior rewarded for synchronized acceleration. Validate engine offside behavior and tactical legality before enabling it. Successful routines earn only their downstream shared competitive return and potential difference. Drill training may temporarily reward legal execution/retained reception, with those bonuses removed for full-match training.

## 7. Manager learning and temporal credit

For a manager action spanning n player steps, store:

    R_macro = sum_k [(product of earlier gamma_j) * r_train(k)]
    Gamma_macro = product_k gamma_k

Bootstrap the manager critic using Gamma_macro and the next decision state. Include terminal events in the correct interval. Use the same objective as the players; do not give separate points for changes, labels, role compliance or routine selection.

First train players with fixed or randomized valid manager instructions. Then freeze a competent player checkpoint and train the manager. Finally experiment with alternating updates. Version opponent, player and manager checkpoints together; joint changes make attribution difficult.

The central critic and team rewards are a baseline, not a complete solution to credit assignment. Investigate more advanced credit mechanisms only after that baseline and the manager interface are working.

## 8. Opponent diversity and human adaptation

Reward shaping alone cannot provide human-level strength. Train against a league containing frozen historical checkpoints, current peers, built-in AI baselines, and deliberately varied tactical opponents. Built-in AI is useful early but is weaker under patched physics and cannot be the only benchmark.

An initial sampling proposal is 40% historical opponents, 30% current/near-current peers, 20% tactical specialists, and 10% built-in AI. Tune from results; these percentages are not validated. Specialists should include central short passing, wide attacks, pressing, direct counters and dribbling. Add opponents that switch style mid-match.

Maintain opponents that exploit discovered weaknesses rather than discarding them. Hold out opponent policies and human sessions from training and classifier calibration. When consented human recordings exist, use observable trajectories for tendency calibration or optional imitation warm starts; do not assume access to recordings or controller internals.

## 9. Implementation sequence for the user's PPO

1. Freeze benchmark physics/actions and verify installed engine versus source. Run build, equivalence, determinism and save/restore checks before training.
2. Build observation encoding, action/power interface and rollout storage. Check mirroring, masks, log probabilities and true-terminal versus truncation behavior.
3. Implement PPO on a small academy drill. Verify improvement and reproducibility before adding coordination or manager complexity.
4. Train shared-policy multi-player coordination with centralized value estimates. Progress through small-team drills to 11v11; introduce randomized tactical instructions.
5. Establish full-match win-objective and unshaped baseline. Tune effective discount horizon before testing potential shaping.
6. Train fixed-player manager with valid tactical packages, role swaps and set-play assignments.
7. Add observable opponent-tendency features and compare adaptive manager versus the same manager without those features.
8. Introduce league opponents and style-switching tests, then alternating manager/player updates if justified.
9. Evaluate held-out humans; address reproducible failures through opponent diversity and retraining, not by adding a reward for every mistake.

Parallel environments must use separate processes because GRF maintains game-level settings per process. Existing mid-game physical stat changes are not undone by set_state(); avoid them in reproducibility/replay tests or explicitly restore/version the stats.

## 10. Required checks and evaluation

Reward checks:
- Discounted shaping sum equals -Phi(initial) + discounted Phi(final), within numerical tolerance.
- True terminal potential is zero; rollout cuts bootstrap and do not create terminal bonuses.
- Passing loops, stationary camping, repeated sprinting and empty shots yield no persistent extra objective return.
- Equivalent trajectories at different tick rates use consistent elapsed-time discounting.
- Both team frames produce consistent rewards; manager intervals include every step exactly once.

Manager checks:
- Tactical swaps are valid and visible to players.
- Restart ownership, game modes, phases and abort paths are correct.
- Low-confidence style estimates do not trigger repeated switches.
- A central-passing opponent that changes to wide play can trigger an appropriate response; no rule assumes the counter must win.

Evaluate all checkpoints with exploration disabled under the same legal settings. Report win/draw/loss, goal difference, opponent mix, confidence intervals, and results by human/session. Repeated matches against one human are correlated; report participant-level variation rather than treating every match as independent proof.

Secondary diagnostics: shots/chance quality, turnovers by zone, transition concessions, set-play outcomes, tactical switch frequency and role execution. These explain performance but cannot select a losing checkpoint over a winning one.

Required ablations: unshaped versus shaped; fixed versus adaptive manager; no tendency inputs; no tactical swaps; no set-play selection. Keep compute and opponent distributions comparable and use multiple seeds when feasible.

Initial acceptance proposal: lower bound of a 95% win-rate interval above 50% against the defined held-out human cohort, alongside results against each opponent subgroup. Predefine draw handling and sample size/power before testing; record draw rate separately. This supports a claim about that cohort, not every possible human. Track worst-subgroup performance and adaptation after style changes to identify brittle aggregate success.

## 11. Decisions to freeze before coding

- Competitive engine switches, tick rate, match duration and legal human assists.
- Whether human control is one active player plus built-in teammates or another format.
- Physical-stat swaps enabled or disabled for each experiment.
- Match-outcome objective, discount horizon and truncation semantics.
- Observation additions, valid tactical packages and first set-play routines.
- Event attribution definitions, opponent pool and held-out evaluation protocol.

## References

Potential-based shaping foundation: Ng, Harada and Russell, Policy invariance under reward transformations (1999): https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf

Shared-policy/centralized-critic baseline inspiration: Yu et al., The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games: https://arxiv.org/abs/2103.01955

These papers support the design foundations, not a claim that this particular reward, manager, opponent sampling schedule or human benchmark has already been validated.
