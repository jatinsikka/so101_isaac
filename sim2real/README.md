# sim2real: run the sim-trained policy on the real SO-101   [C3, 2026-10-10]

Runs on the Mac, in the **base** conda env (needs `numpy`, `onnxruntime`, `feetech-servo-sdk`).
Every script is a **dry run unless you pass `--move`**.

| file | what it does |
|---|---|
| `arm.py` | the only file that talks to the servos: hardware facts at the top, then read / send / torque |
| `move_to.py` | slowly move to a pose, one joint at a time (`zero`, `reach`, or `--q`) |
| `policy.py` | the 25-number observation (same order as training) and the ONNX network  [C6] |
| `run_policy.py` | the 50 Hz loop: read -> policy -> send; `--push` adds the dead-zone push; logs to CSV  [C6] |
| `step_test.py` | step one joint's target and record how it moves (real side of the actuator check) |
| `../scripts/step_test_sim.py` | the same test in sim, on the AWS box |
| `tools/` | one-off hardware checks: `scan_motors.py`, `read_joints.py`, `diag_servo.py` |
| `old/` | the previous version of this folder, untouched |
| `logs/` | CSVs written by the scripts (not in git) |

## Order

```bash
cd ~/so101_isaac/sim2real
python arm.py                                   # read joints (never moves)
python move_to.py zero --move                   # go to the sim's zero pose
python step_test.py --joint elbow_flex --move   # measure the real joint
python run_policy.py --move --push 0.02         # run the policy (from zero)
python arm.py --release                         # torque off (support the arm!)
```

To go back to the old version: move the files in `old/` back up one folder (see `CHANGES.md`, C3).
