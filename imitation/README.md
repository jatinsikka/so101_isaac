# Phase 2: Imitation Learning

**Goal**: train a policy from *demonstrations* instead of reward — either demos harvested from
your trained RL policy (sim), or teleoperated demos on the real SO-101 (real).

This phase is deliberately left as skeleton code (`collect_demos.py`, `train_bc.py`, `dataset.py`)
with `# TODO`s. You already know the RL side; the concepts below are the ones that are *new* here.

## Concepts / tips

- **Behavioral Cloning (BC)** is supervised learning: `policy(obs) -> action`, trained on
  `(obs, action)` pairs from a dataset of demonstrations, minimizing e.g. MSE or NLL loss.
  It's the simplest IL algorithm — get this working first before anything fancier.
- **Where do demos come from?** Two options, and you can do either or both:
  1. **RL-policy rollout** (sim only): run your trained `reach-v0` checkpoint in `play.py`-style
     inference and log `(obs, action)` pairs to disk. Fast, free, but the policy you're imitating
     is just as good/bad as your RL policy — this mainly teaches you the IL *pipeline*.
  2. **Human teleoperation** (real SO-101): drive the arm by hand (e.g. via a leader-follower
     setup, a spacemouse, or keyboard jogging) and record `(obs, action)` at each timestep.
     This is how real embodied-AI imitation datasets (e.g. LeRobot's) are built. Since you have
     the hardware, this is worth doing — it's also the dataset you'd need for anything beyond
     simple reaching (e.g. picking up objects, tasks the RL reward function can't easily express).
- **Distribution shift / compounding error** is the classic BC failure mode: small errors early
  in a rollout push the robot into states that weren't in the training data, causing errors to
  compound. Watch for this when you evaluate — a policy with low training loss can still fail
  at rollout time. (DAgger, and modern approaches like ACT/diffusion policies with action
  chunking, exist specifically to address this — worth reading about once BC works.)
- **LeRobot** (https://github.com/huggingface/lerobot) is the reference stack for SO-101 —
  it defines the dataset format (`LeRobotDataset`), has drivers for the real arm, and reference
  IL implementations (ACT, diffusion policy, etc). You don't have to use it, but it's worth
  skimming so you're not reinventing their dataset schema by accident.
- **Action space consistency**: your RL task uses `RelativeJointPositionActionCfg` (delta joint
  targets). Whatever you imitate needs the *same* action definition your policy was trained/will
  be deployed with — decide this before you start collecting demos, not after.

## Files

- `dataset.py` — demo storage format + loader (TODO: you design this)
- `collect_demos.py` — gathers `(obs, action)` pairs, either from an RL checkpoint rollout in
  sim, or from teleoperating the real arm (TODO: both collection paths)
- `train_bc.py` — supervised training loop over the collected dataset (TODO: model + loss + loop)
- `eval_bc.py` — roll out the trained BC policy in sim (`reach-v0`) to sanity-check it before
  ever touching the real robot (TODO)
