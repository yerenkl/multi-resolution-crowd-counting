#!/bin/bash
# Usage: ./jobs/submit_dann_v2.sh [queue ...]
# Default queue: gpua100
# Examples:
#   ./jobs/submit_dann_v2.sh
#   ./jobs/submit_dann_v2.sh gpuv100
#   ./jobs/submit_dann_v2.sh gpua100 gpuv100 gpul40s

QUEUES=("${@:-gpua100}")

for QUEUE in "${QUEUES[@]}"; do
    echo "Submitting to $QUEUE..."
    bsub -q "$QUEUE" < jobs/run_dann_v2_2x.sh
    bsub -q "$QUEUE" < jobs/run_dann_v2_4x.sh
    bsub -q "$QUEUE" < jobs/run_dann_v2_mixed.sh
done
