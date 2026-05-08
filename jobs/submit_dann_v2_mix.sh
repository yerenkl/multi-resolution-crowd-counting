#!/bin/bash
# Submit the mix-methods DANN v2 training run.
# Usage: ./jobs/submit_dann_v2_mix.sh [queue]
# Default queue: gpua100

QUEUE="${1:-gpua100}"
echo "Submitting dann_v2_mix_methods to $QUEUE..."
bsub -q "$QUEUE" < jobs/run_dann_v2_mix_methods.sh
