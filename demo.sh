#!/usr/bin/env bash

# Demo script to run distributed neural network training
# This script starts multiple Elixir nodes and demonstrates distributed training

echo "🧠 Distributed Neural Network Training Demo"
echo "==========================================="
echo ""

# Check if iex is available
if ! command -v iex &> /dev/null; then
    echo "❌ Elixir/iex not found. Please install Elixir first."
    exit 1
fi

echo "🚀 Starting Node A (Primary)..."
echo "This will be the main node with the web interface."
echo ""
echo "In another terminal, you can start Node B with:"
echo "  iex --name node_b@127.0.0.1 --cookie dist_train -S mix"
echo ""
echo "Then in the web interface, connect to node_b@127.0.0.1"
echo ""
echo "Starting primary node..."
echo "Navigate to http://localhost:4000/training to see the demo"
echo ""

# Start the primary node
iex --name node_a@127.0.0.1 --cookie dist_train -S mix phx.server
