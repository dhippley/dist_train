defmodule DistTrain.NeuralNetwork.Layer do
  @moduledoc """
  A neural network layer that can run on a distributed node.
  """

  use GenServer
  require Logger
  alias DistTrain.NeuralNetwork.Math

  defstruct [:id, :weights, :bias, :activation, :input_size, :output_size, :node_name, :next_layer, :prev_layer]

  @doc """
  Starts a new layer process.
  """
  def start_link(opts) do
    {name, opts} = Keyword.pop(opts, :name)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Performs forward propagation through this layer.
  """
  def forward(layer_pid, input) do
    GenServer.call(layer_pid, {:forward, input})
  end

  @doc """
  Performs backward propagation through this layer.
  """
  def backward(layer_pid, gradient) do
    GenServer.call(layer_pid, {:backward, gradient})
  end

  @doc """
  Gets the current weights and bias of the layer.
  """
  def get_parameters(layer_pid) do
    GenServer.call(layer_pid, :get_parameters)
  end

  @doc """
  Updates the weights and bias of the layer.
  """
  def update_parameters(layer_pid, weights, bias) do
    GenServer.call(layer_pid, {:update_parameters, weights, bias})
  end

  ## GenServer Callbacks

  @impl true
  def init(opts) do
    id = Keyword.get(opts, :id)
    input_size = Keyword.get(opts, :input_size)
    output_size = Keyword.get(opts, :output_size)
    activation = Keyword.get(opts, :activation, :relu)
    next_layer = Keyword.get(opts, :next_layer)
    prev_layer = Keyword.get(opts, :prev_layer)

    # Initialize weights using simple initialization
    weights = Nx.broadcast(0.1, {input_size, output_size})
    bias = Nx.broadcast(0.0, {1, output_size})

    state = %__MODULE__{
      id: id,
      weights: weights,
      bias: bias,
      activation: activation,
      input_size: input_size,
      output_size: output_size,
      node_name: Node.self(),
      next_layer: next_layer,
      prev_layer: prev_layer
    }

    Logger.info("Layer #{id} initialized on node #{Node.self()}")
    {:ok, state}
  end

  @impl true
  def handle_call({:forward, input}, _from, state) do
    Logger.debug("Layer #{state.id} performing forward pass")

    # Store input for backward pass
    Process.put(:last_input, input)

    # Linear transformation
    z = Math.linear_forward(input, state.weights, state.bias)

    # Apply activation function
    output = case state.activation do
      :relu -> Math.relu(z)
      :sigmoid -> Math.sigmoid(z)
      :linear -> z
    end

    # Store pre-activation values for backward pass
    Process.put(:last_z, z)

    # If there's a next layer, send the output to it
    result = if state.next_layer do
      try do
        # Send to next layer (possibly on different node)
        forward_result = GenServer.call(state.next_layer, {:forward, output}, 10_000)
        {:ok, forward_result}
      catch
        :exit, reason ->
          Logger.error("Failed to forward to next layer: #{inspect(reason)}")
          {:error, reason}
      end
    else
      {:ok, output}
    end

    {:reply, result, state}
  end

  @impl true
  def handle_call({:backward, gradient}, _from, state) do
    Logger.debug("Layer #{state.id} performing backward pass")

    # Get stored values from forward pass
    z = Process.get(:last_z)
    input = Process.get(:last_input)

    # Compute activation gradient
    activation_grad = case state.activation do
      :relu -> Math.relu_derivative(z)
      :sigmoid -> Math.sigmoid_derivative(z)
      :linear -> Nx.broadcast(1.0, Nx.shape(z))
    end

    # Element-wise multiplication with incoming gradient
    delta = Nx.multiply(gradient, activation_grad)

    # Compute gradients for weights and bias
    weight_grad = Nx.dot(Nx.transpose(input), delta)
    bias_grad = Nx.sum(delta, axes: [0], keep_axes: true)

    # Compute gradient to pass to previous layer
    input_grad = if state.prev_layer do
      Nx.dot(delta, Nx.transpose(state.weights))
    else
      nil
    end

    # Send gradient to previous layer if it exists
    if state.prev_layer && input_grad do
      spawn(fn ->
        try do
          GenServer.call(state.prev_layer, {:backward, input_grad}, 10_000)
        catch
          :exit, reason ->
            Logger.error("Failed to send gradient to previous layer: #{inspect(reason)}")
        end
      end)
    end

    result = %{
      weight_grad: weight_grad,
      bias_grad: bias_grad,
      input_grad: input_grad
    }

    {:reply, result, state}
  end

  @impl true
  def handle_call(:get_parameters, _from, state) do
    {:reply, %{weights: state.weights, bias: state.bias}, state}
  end

  @impl true
  def handle_call({:update_parameters, weights, bias}, _from, state) do
    new_state = %{state | weights: weights, bias: bias}
    {:reply, :ok, new_state}
  end
end
