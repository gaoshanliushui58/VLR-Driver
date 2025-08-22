#!/bin/bash
BASE_PORT=30005
BASE_TM_PORT=50005
IS_BENCH2DRIVE=True
BASE_ROUTES=leaderboard/data/bench2drive5
TEAM_CONFIG=leaderboard/team_code/lmb2d/lmb2d_config.py
BASE_CHECKPOINT_ENDPOINT=eval
SAVE_PATH=./output/
PLANNER_TYPE=only_traj

CARLA_GPU_RANK=1
PORT=$BASE_PORT
TM_PORT=$BASE_TM_PORT
ROUTES="${BASE_ROUTES}.xml"
CHECKPOINT_ENDPOINT="${SAVE_PATH}${BASE_CHECKPOINT_ENDPOINT}.json"
bash leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $CARLA_GPU_RANK
#                                           1     2        3               4       5           6            7                    8          9             10