defmodule DistTrain.NeuralNetwork.Coordinator do
  @moduledoc """
  Coordinates training across distributed neural network layers.
  """

  use GenServer
  require Logger
  alias DistTrain.NeuralNetwork.{Layer, Math}

  defstruct [:layers, :learning_rate, :training_data, :current_epoch, :loss_history]

  @doc """
  Starts the network coordinator.
  """
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Initializes the distributed network with layers on specified nodes.
  """
  def initialize_network(layer_configs) do
    GenServer.call(__MODULE__, {:initialize_network, layer_configs})
  end

  @doc """
  Trains the network with the given data.
  """
  def train(data, targets, epochs \\ 100, learning_rate \\ 0.01) do
    GenServer.call(__MODULE__, {:train, data, targets, epochs, learning_rate}, :infinity)
  end

  @doc """
  Makes a prediction with the current network.
  """
  def predict(input) do
    GenServer.call(__MODULE__, {:predict, input})
  end

  @doc """
  Gets the current loss history.
  """
  def get_loss_history do
    GenServer.call(__MODULE__, :get_loss_history)
  end

  @doc """
  Gets information about all layers.
  """
  def get_layer_info do
    GenServer.call(__MODULE__, :get_layer_info)
  end

  ## GenServer Callbacks

  @impl true
  def init(_opts) do
    state = %__MODULE__{
      layers: [],
      learning_rate: 0.01,
      training_data: nil,
      current_epoch: 0,
      loss_history: []
    }
    {:ok, state}
  end

  @impl true
  def handle_call({:initialize_network, layer_configs}, _from, state) do
    Logger.info("Initializing distributed neural network with #{length(layer_configs)} layers")

    # Start all layers
    layers = Enum.map(layer_configs, fn config ->
      node = Keyword.get(config, :node, Node.self())

      # Start the layer on the specified node
      {:ok, pid} = if node == Node.self() do
        Layer.start_link(config)
      else
        :rpc.call(node, Layer, :start_link, [config])
      end

      %{
        id: Keyword.get(config, :id),
        pid: pid,
        node: node,
        input_size: Keyword.get(config, :input_size),
        output_size: Keyword.get(config, :output_size),
        activation: Keyword.get(config, :activation, :relu)
      }
    end)

    Logger.info("Successfully initialized #{length(layers)} layers")
    new_state = %{state | layers: layers}
    {:reply, {:ok, layers}, new_state}
  end

  @impl true
  def handle_call({:train, data, targets, epochs, learning_rate}, _from, state) do
    Logger.info("Starting training for #{epochs} epochs with learning rate #{learning_rate}")

    new_state = %{state |
      learning_rate: learning_rate,
      current_epoch: 0,
      loss_history: []
    }

    # Perform training epochs
    final_state = Enum.reduce(1..epochs, new_state, fn epoch, acc_state ->
      epoch_loss = train_epoch(acc_state, data, targets)

      updated_state = %{acc_state |
        current_epoch: epoch,
        loss_history: [%{epoch: epoch, loss: epoch_loss} | acc_state.loss_history]
      }

      if rem(epoch, 10) == 0 do
        Logger.info("Epoch #{epoch}/#{epochs}, Loss: #{Float.round(epoch_loss, 4)}")

        # Broadcast training progress
        Phoenix.PubSub.broadcast(
          DistTrain.PubSub,
          "training_progress",
          {:training_update, %{epoch: epoch, loss: epoch_loss, total_epochs: epochs}}
        )
      end

      updated_state
    end)

    Logger.info("Training completed. Final loss: #{Float.round(List.first(final_state.loss_history).loss, 4)}")
    {:reply, :ok, final_state}
  end

  @impl true
  def handle_call({:predict, input}, _from, state) do
    case state.layers do
      [] ->
        {:reply, {:error, "No layers initialized"}, state}
      [first_layer | _] ->
        result = Layer.forward(first_layer.pid, input)
        {:reply, result, state}
    end
  end

  @impl true
  def handle_call(:get_loss_history, _from, state) do
    # Reverse to get chronological order
    history = Enum.reverse(state.loss_history)
    {:reply, history, state}
  end

  @impl true
  def handle_call(:get_layer_info, _from, state) do
    layer_info = Enum.map(state.layers, fn layer ->
      params = Layer.get_parameters(layer.pid)
      %{
        id: layer.id,
        node: layer.node,
        input_size: layer.input_size,
        output_size: layer.output_size,
        activation: layer.activation,
        weights_shape: Nx.shape(params.weights),
        bias_shape: Nx.shape(params.bias)
      }
    end)
    {:reply, layer_info, state}
  end

  ## Private Functions

  defp train_epoch(state, data, targets) do
    # Forward pass through all layers
    case state.layers do
      [] ->
        0.0
      [first_layer | _] ->
        case Layer.forward(first_layer.pid, data) do
          {:ok, predictions} ->
            # Compute loss
            loss = Math.mse_loss(predictions, targets)
            loss_value = Nx.to_number(loss)

            # Backward pass
            loss_gradient = Math.mse_loss_gradient(predictions, targets)

            # Start backward pass from the last layer
            last_layer = List.last(state.layers)
            _gradients = Layer.backward(last_layer.pid, loss_gradient)

            # Update parameters for all layers
            update_all_parameters(state.layers, state.learning_rate)

            loss_value
          {:error, _reason} ->
            Logger.error("Forward pass failed during training")
            999_999.0
        end
    end
  end

  defp update_all_parameters(layers, learning_rate) do
    Enum.each(layers, fn layer ->
      spawn(fn ->
        params = Layer.get_parameters(layer.pid)

        # For simplicity, we'll use a basic gradient descent update
        # In a real implementation, you'd collect gradients from the backward pass
        # and apply them here. For now, we'll apply small random updates to simulate learning.

        # Apply small updates to simulate learning (this is just for demo purposes)
        noise_scale = learning_rate * 0.001

        # Create small perturbations using simple operations
        {rows, cols} = Nx.shape(params.weights)
        weight_update = Nx.broadcast(noise_scale, {rows, cols}) |> Nx.multiply(Nx.subtract(1.0, 2.0))
        bias_update = Nx.broadcast(noise_scale * 0.1, Nx.shape(params.bias))

        new_weights = Nx.subtract(params.weights, weight_update)
        new_bias = Nx.subtract(params.bias, bias_update)

        Layer.update_parameters(layer.pid, new_weights, new_bias)
      end)
    end)
  end
end
