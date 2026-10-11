#!/usr/bin/env bash
# Step 3: teleoperate -- move the leader, the follower copies it. Nothing is recorded.
# Use this to practise the task before recording, and to check the camera view.
#
#   conda activate lerobot
#   ./03_teleop.sh
#
# Before starting, put both arms in roughly the same pose: on start the follower jumps
# to the leader's pose. Ctrl+C to stop -- the follower then goes limp, so support it.
source "$(dirname "$0")/common.sh"

# max_relative_target: cap on how far (degrees) any joint's command may change per step,
# so a pose mismatch or a jerk on the leader can't throw the follower.
# display_data: opens the viewer with joint plots and the wrist-camera feed.
lerobot-teleoperate "${ROBOT_ARGS[@]}" "${TELEOP_ARGS[@]}" \
    --robot.max_relative_target=10 \
    --display_data=true
