# Distributed Neural Network Training

A demonstration of distributed neural network training using Elixir, Phoenix LiveView, and Nx. This project showcases how to build neural networks where different layers run on different nodes, communicating via Elixir's built-in distributed capabilities.

## 🎯 Project Goal

Build a small feed-forward neural network where:
- **Layer 1** lives on **Node A**
- **Layer 2** lives on **Node B** 
- Data flows forward from A → B
- Gradients flow backward B → A

## 🏗️ Architecture

```
┌─────────────┐    Forward Pass    ┌─────────────┐
│   Node A    │ ──────────────────► │   Node B    │
│   Layer 1   │                    │   Layer 2   │
│   (ReLU)    │ ◄────────────────── │  (Linear)   │
└─────────────┘   Backward Pass    └─────────────┘
```

## ✨ Features

- **Distributed Architecture**: Neural network layers run on different Elixir nodes
- **Real-time Visualization**: Monitor training progress with Phoenix LiveView
- **Node Management**: Dynamically connect and manage multiple nodes
- **Fault Tolerance**: Built on Elixir's OTP for robust distributed computation
- **Interactive Training**: Train on the XOR problem with live loss visualization

## 🚀 Quick Start

### Prerequisites

- Elixir 1.14+
- Phoenix Framework
- PostgreSQL (for Phoenix setup)

### Installation

1. Clone and setup:
```bash
git clone <your-repo>
cd dist_train
mix setup
```

2. Start the primary node:
```bash
./demo.sh
```

3. In another terminal, start a second node:
```bash
iex --name node_b@127.0.0.1 --cookie dist_train -S mix
```

4. Open your browser to [http://localhost:4000/training](http://localhost:4000/training)

5. In the web interface:
   - Connect to `node_b@127.0.0.1`
   - Initialize the network
   - Start training
   - Watch the distributed training in real-time!

## 🧠 How It Works

### Layer Distribution

The system automatically distributes neural network layers across available nodes:

```elixir
# Layer 1 on Node A
layer_1_config = [
  id: 1,
  input_size: 2,
  output_size: 4,
  activation: :relu,
  node: :node_a@127.0.0.1
]

# Layer 2 on Node B  
layer_2_config = [
  id: 2,
  input_size: 4,
  output_size: 1,
  activation: :linear,
  node: :node_b@127.0.0.1
]
```

### Forward Propagation

Data flows through the network across nodes:

1. Input enters Layer 1 on Node A
2. Layer 1 applies ReLU activation
3. Output is sent to Layer 2 on Node B
4. Layer 2 produces final prediction

### Backward Propagation

Gradients flow backward through the distributed network:

1. Loss gradient computed on final node
2. Gradient sent back to Layer 1 on Node A
3. Each layer updates its parameters locally
4. Process repeats for next training iteration

## 🔧 Technical Components

### Core Modules

- **`DistTrain.NeuralNetwork.Layer`**: Individual layer GenServer that can run on any node
- **`DistTrain.NeuralNetwork.Coordinator`**: Manages distributed training across nodes
- **`DistTrain.NeuralNetwork.Math`**: Mathematical operations using Nx
- **`DistTrain.NeuralNetwork.NodeManager`**: Handles node connections and distribution

### Web Interface

- **`DistTrainWeb.TrainingLive`**: Phoenix LiveView for real-time monitoring
- Real-time loss visualization
- Node connection management
- Network topology display
- Training controls

## 📊 Training Demo

The system demonstrates distributed training on the XOR problem:

| Input | Target |
|-------|--------|
| [0,0] | 0      |
| [0,1] | 1      |
| [1,0] | 1      |
| [1,1] | 0      |

Watch as the distributed network learns this non-linear pattern across multiple nodes!

## 🌐 Multi-Node Setup

### Local Development
```bash
# Terminal 1 - Primary node with web interface
iex --name node_a@127.0.0.1 --cookie dist_train -S mix phx.server

# Terminal 2 - Secondary node
iex --name node_b@127.0.0.1 --cookie dist_train -S mix
```

### Production Deployment
```bash
# Node A
iex --name node_a@192.168.1.10 --cookie your_secret -S mix phx.server

# Node B  
iex --name node_b@192.168.1.11 --cookie your_secret -S mix
```

## 🔬 Extending the System

Want to add more layers or nodes? Easy!

```elixir
# Add a third layer on a third node
layer_configs = [
  [id: 1, input_size: 2, output_size: 8, activation: :relu],
  [id: 2, input_size: 8, output_size: 4, activation: :relu], 
  [id: 3, input_size: 4, output_size: 1, activation: :linear]
]

# System automatically distributes across available nodes
DistTrain.NeuralNetwork.Coordinator.initialize_network(layer_configs)
```

## 🚨 Troubleshooting

### Nodes Can't Connect
- Ensure same cookie: `--cookie dist_train`
- Check firewall settings for Erlang distribution
- Use full hostnames: `node@hostname.local`

### Training Fails
- Verify all nodes are connected
- Check network initialization completed
- Look for error messages in logs

## 📚 Learn More

- [Elixir Distribution](https://elixir-lang.org/getting-started/mix-otp/distributed-tasks.html)
- [Phoenix LiveView](https://hexdocs.pm/phoenix_live_view/Phoenix.LiveView.html)
- [Nx Numerical Computing](https://hexdocs.pm/nx/Nx.html)
- [OTP GenServer](https://hexdocs.pm/elixir/GenServer.html)

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- More sophisticated neural network architectures
- Better gradient synchronization algorithms  
- Support for GPUs via EXLA
- Advanced visualization features
- Performance optimizations

## 📄 License

MIT License - feel free to use this for learning and experimentation!
