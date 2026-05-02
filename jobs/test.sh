#!/bin/bash

# Define log paths
OUT_LOG="output.log"
ERR_LOG="error.log"

# Apply the redirection
exec > >(tee -a "$OUT_LOG") \
     2> >(tee -a "$ERR_LOG" >&2)

# Test messages
echo "This is a standard message."
echo "This is an error message." >&2

# Give the background 'tee' processes a moment to finish
sleep 1