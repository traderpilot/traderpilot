#!/bin/bash

echo "Running Unit tests"

pytest --random-order --cov=traderpilot --cov-config=.coveragerc tests/
