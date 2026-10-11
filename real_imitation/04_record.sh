#!/usr/bin/env bash
# Step 4: record demonstrations. Each episode = one full attempt at the task, done through
# the leader, saved as follower joint positions + leader actions + wrist-camera video.
#
#   conda activate lerobot
#   ./04_record.sh 2            # a 2-episode trial run, to check everything saves
#   ./04_record.sh 50 --resume  # add 50 more episodes to the same dataset
#
# Keyboard while recording (the terminal needs macOS Accessibility permission for these):
#   Right arrow   finish this episode early (use it as soon as the task is done)
#   Left arrow    throw away this episode and redo it
#   Esc           stop recording and save everything so far
#
# Demo-quality tips (these matter more than the number of demos):
#   - start the block inside the same taped area every time, in a similar orientation
#   - move smoothly and at a steady pace; hesitations and corrections get learned too
#   - do the task the SAME way each time (same approach direction, same grasp)
#   - end each episode back at a similar rest pose
#
# No --robot.max_relative_target here, on purpose: lerobot-record saves the LEADER's command as
# the action even when that cap clamps what is actually sent (checked in lerobot_record.py),
# so with the cap on the dataset would contain actions the arm never executed. Teleop is
# verified without surprises, so record uncapped -- and move the leader smoothly instead.
source "$(dirname "$0")/common.sh"

NUM_EPISODES=${1:-2}
RESUME=false
[[ "$2" == "--resume" ]] && RESUME=true

# The task string is stored with every frame; language-conditioned policies use it, and it
# documents what the dataset is. Keep it identical across sessions.
TASK="Pick up the block and put it in the bowl"
# Datasets stay local (data/ is gitignored: video is large). repo_id is just the dataset's
# name here; it only matters if you later push it to the Hugging Face Hub.
REPO_ID=jatinsikka/so101_pick_place

lerobot-record "${ROBOT_ARGS[@]}" "${TELEOP_ARGS[@]}" \
    --dataset.repo_id="$REPO_ID" \
    --dataset.root="data/so101_pick_place" \
    --dataset.single_task="$TASK" \
    --dataset.num_episodes="$NUM_EPISODES" \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=10 \
    --dataset.push_to_hub=false \
    --resume="$RESUME" \
    --display_data=true
# episode_time_s: hard cap per attempt (press Right arrow to end sooner).
# reset_time_s:   time between episodes to put the block back; nothing is saved during it.
