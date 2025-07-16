defmodule DistTrain.NeuralNetwork.NodeManager do
  @moduledoc """
  Manages distributed nodes for the neural network.
  """

  use GenServer
  require Logger

  defstruct [:nodes, :node_layers]

  @doc """
  Starts the node manager.
  """
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Connects to a list of nodes.
  """
  def connect_nodes(nodes) do
    GenServer.call(__MODULE__, {:connect_nodes, nodes})
  end

  @doc """
  Gets the list of connected nodes.
  """
  def get_connected_nodes do
    GenServer.call(__MODULE__, :get_connected_nodes)
  end

  @doc """
  Distributes layers across available nodes.
  """
  def distribute_layers(layer_configs) do
    GenServer.call(__MODULE__, {:distribute_layers, layer_configs})
  end

  ## GenServer Callbacks

  @impl true
  def init(_opts) do
    state = %__MODULE__{
      nodes: [Node.self()],
      node_layers: %{}
    }
    {:ok, state}
  end

  @impl true
  def handle_call({:connect_nodes, nodes}, _from, state) do
    Logger.info("Attempting to connect to nodes: #{inspect(nodes)}")

    connected_nodes = Enum.filter(nodes, fn node ->
      case Node.connect(node) do
        true ->
          Logger.info("Successfully connected to #{node}")
          true
        false ->
          Logger.warning("Failed to connect to #{node}")
          false
        :ignored ->
          Logger.info("Already connected to #{node}")
          true
      end
    end)

    all_nodes = [Node.self() | connected_nodes] |> Enum.uniq()
    new_state = %{state | nodes: all_nodes}

    {:reply, {:ok, connected_nodes}, new_state}
  end

  @impl true
  def handle_call(:get_connected_nodes, _from, state) do
    {:reply, state.nodes, state}
  end

  @impl true
  def handle_call({:distribute_layers, layer_configs}, _from, state) do
    # Distribute layers across available nodes
    distributed_configs = layer_configs
    |> Enum.with_index()
    |> Enum.map(fn {config, index} ->
      node = Enum.at(state.nodes, rem(index, length(state.nodes)))
      Keyword.put(config, :node, node)
    end)

    Logger.info("Distributed #{length(layer_configs)} layers across #{length(state.nodes)} nodes")

    {:reply, distributed_configs, state}
  end
end
