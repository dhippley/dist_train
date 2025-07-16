defmodule DistTrainWeb.TrainingLive do
  @moduledoc """
  LiveView for monitoring distributed neural network training.
  """

  use DistTrainWeb, :live_view
  alias DistTrain.NeuralNetwork.{Coordinator, NodeManager}

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket) do
      Phoenix.PubSub.subscribe(DistTrain.PubSub, "training_progress")
    end

    socket =
      socket
      |> assign(:training_active, false)
      |> assign(:current_epoch, 0)
      |> assign(:total_epochs, 0)
      |> assign(:current_loss, 0.0)
      |> assign(:loss_history, [])
      |> assign(:connected_nodes, [Node.self()])
      |> assign(:layer_info, [])
      |> assign(:network_initialized, false)

    {:ok, socket}
  end

  @impl true
  def handle_event("initialize_network", _params, socket) do
    # Define a simple 2-layer network
    layer_configs = [
      [
        id: 1,
        name: {:global, :layer_1},
        input_size: 2,
        output_size: 4,
        activation: :relu,
        next_layer: {:global, :layer_2},
        prev_layer: nil
      ],
      [
        id: 2,
        name: {:global, :layer_2},
        input_size: 4,
        output_size: 1,
        activation: :linear,
        next_layer: nil,
        prev_layer: {:global, :layer_1}
      ]
    ]

    # Distribute layers across nodes
    distributed_configs = NodeManager.distribute_layers(layer_configs)

    case Coordinator.initialize_network(distributed_configs) do
      {:ok, _layers} ->
        layer_info = Coordinator.get_layer_info()

        socket =
          socket
          |> assign(:network_initialized, true)
          |> assign(:layer_info, layer_info)
          |> put_flash(:info, "Network initialized successfully!")

        {:noreply, socket}

      {:error, reason} ->
        socket = put_flash(socket, :error, "Failed to initialize network: #{inspect(reason)}")
        {:noreply, socket}
    end
  end

  @impl true
  def handle_event("start_training", _params, socket) do
    if socket.assigns.network_initialized do
      # Generate some sample training data (XOR problem)
      data = Nx.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
      targets = Nx.tensor([[0.0], [1.0], [1.0], [0.0]])

      epochs = 100
      learning_rate = 0.01

      # Start training in a separate process
      spawn(fn ->
        Coordinator.train(data, targets, epochs, learning_rate)
      end)

      socket =
        socket
        |> assign(:training_active, true)
        |> assign(:total_epochs, epochs)
        |> put_flash(:info, "Training started!")

      {:noreply, socket}
    else
      socket = put_flash(socket, :error, "Please initialize the network first!")
      {:noreply, socket}
    end
  end

  @impl true
  def handle_event("connect_node", %{"node" => node_name}, socket) do
    node_atom = String.to_atom(node_name)

    case NodeManager.connect_nodes([node_atom]) do
      {:ok, connected} ->
        nodes = NodeManager.get_connected_nodes()

        socket =
          socket
          |> assign(:connected_nodes, nodes)
          |> put_flash(:info, "Connected to #{length(connected)} nodes")

        {:noreply, socket}

      {:error, reason} ->
        socket = put_flash(socket, :error, "Failed to connect: #{inspect(reason)}")
        {:noreply, socket}
    end
  end

  @impl true
  def handle_event("test_prediction", _params, socket) do
    if socket.assigns.network_initialized do
      # Test with XOR inputs
      test_input = Nx.tensor([[0.0, 1.0]])

      case Coordinator.predict(test_input) do
        {:ok, prediction} ->
          pred_value = prediction |> Nx.to_flat_list() |> hd() |> Float.round(4)

          socket = put_flash(socket, :info, "Prediction for [0, 1]: #{pred_value}")
          {:noreply, socket}

        {:error, reason} ->
          socket = put_flash(socket, :error, "Prediction failed: #{inspect(reason)}")
          {:noreply, socket}
      end
    else
      socket = put_flash(socket, :error, "Please initialize the network first!")
      {:noreply, socket}
    end
  end

  @impl true
  def handle_info({:training_update, %{epoch: epoch, loss: loss, total_epochs: total_epochs}}, socket) do
    loss_history = [%{epoch: epoch, loss: loss} | socket.assigns.loss_history]
    |> Enum.take(50)  # Keep only last 50 points for performance

    socket =
      socket
      |> assign(:current_epoch, epoch)
      |> assign(:total_epochs, total_epochs)
      |> assign(:current_loss, loss)
      |> assign(:loss_history, loss_history)
      |> assign(:training_active, epoch < total_epochs)

    {:noreply, socket}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <div class="p-6 max-w-6xl mx-auto">
      <h1 class="text-3xl font-bold text-gray-900 mb-8">Distributed Neural Network Training</h1>

      <!-- Network Status -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div class="bg-white rounded-lg shadow p-6">
          <h3 class="text-lg font-medium text-gray-900 mb-4">Network Status</h3>
          <div class="space-y-2">
            <div class="flex justify-between">
              <span class="text-gray-600">Initialized:</span>
              <span class={if @network_initialized, do: "text-green-600", else: "text-red-600"}>
                <%= if @network_initialized, do: "✓ Yes", else: "✗ No" %>
              </span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-600">Training:</span>
              <span class={if @training_active, do: "text-blue-600", else: "text-gray-600"}>
                <%= if @training_active, do: "🔄 Active", else: "⏸ Inactive" %>
              </span>
            </div>
          </div>
        </div>

        <div class="bg-white rounded-lg shadow p-6">
          <h3 class="text-lg font-medium text-gray-900 mb-4">Training Progress</h3>
          <div class="space-y-2">
            <div class="flex justify-between">
              <span class="text-gray-600">Epoch:</span>
              <span class="font-mono"><%= @current_epoch %>/<%= @total_epochs %></span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-600">Loss:</span>
              <span class="font-mono"><%= Float.round(@current_loss, 6) %></span>
            </div>
          </div>
          <div class="mt-4">
            <div class="bg-gray-200 rounded-full h-2">
              <div
                class="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={"width: #{if @total_epochs > 0, do: (@current_epoch / @total_epochs * 100), else: 0}%"}
              >
              </div>
            </div>
          </div>
        </div>

        <div class="bg-white rounded-lg shadow p-6">
          <h3 class="text-lg font-medium text-gray-900 mb-4">Connected Nodes</h3>
          <div class="space-y-2">
            <%= for node <- @connected_nodes do %>
              <div class="flex items-center space-x-2">
                <div class="w-2 h-2 bg-green-500 rounded-full"></div>
                <span class="text-sm font-mono text-gray-700"><%= node %></span>
              </div>
            <% end %>
          </div>
        </div>
      </div>

      <!-- Controls -->
      <div class="bg-white rounded-lg shadow p-6 mb-8">
        <h3 class="text-lg font-medium text-gray-900 mb-4">Controls</h3>
        <div class="flex flex-wrap gap-4">
          <button
            phx-click="initialize_network"
            disabled={@network_initialized}
            class="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            Initialize Network
          </button>

          <button
            phx-click="start_training"
            disabled={@training_active or not @network_initialized}
            class="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            Start Training
          </button>

          <button
            phx-click="test_prediction"
            disabled={not @network_initialized}
            class="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            Test Prediction
          </button>
        </div>

        <div class="mt-4">
          <form phx-submit="connect_node" class="flex gap-2">
            <input
              type="text"
              name="node"
              placeholder="node@hostname"
              class="flex-1 px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
              type="submit"
              class="px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700"
            >
              Connect Node
            </button>
          </form>
        </div>
      </div>

      <!-- Layer Information -->
      <%= if @network_initialized do %>
        <div class="bg-white rounded-lg shadow p-6 mb-8">
          <h3 class="text-lg font-medium text-gray-900 mb-4">Layer Information</h3>
          <div class="overflow-x-auto">
            <table class="min-w-full divide-y divide-gray-200">
              <thead class="bg-gray-50">
                <tr>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Layer ID</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Node</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Input Size</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Output Size</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Activation</th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Weights Shape</th>
                </tr>
              </thead>
              <tbody class="bg-white divide-y divide-gray-200">
                <%= for layer <- @layer_info do %>
                  <tr>
                    <td class="px-6 py-4 text-sm font-mono text-gray-900"><%= layer.id %></td>
                    <td class="px-6 py-4 text-sm font-mono text-gray-900"><%= layer.node %></td>
                    <td class="px-6 py-4 text-sm text-gray-900"><%= layer.input_size %></td>
                    <td class="px-6 py-4 text-sm text-gray-900"><%= layer.output_size %></td>
                    <td class="px-6 py-4 text-sm text-gray-900"><%= layer.activation %></td>
                    <td class="px-6 py-4 text-sm font-mono text-gray-900"><%= inspect(layer.weights_shape) %></td>
                  </tr>
                <% end %>
              </tbody>
            </table>
          </div>
        </div>
      <% end %>

      <!-- Loss Chart -->
      <%= if length(@loss_history) > 0 do %>
        <div class="bg-white rounded-lg shadow p-6">
          <h3 class="text-lg font-medium text-gray-900 mb-4">Training Loss</h3>
          <div class="h-64 flex items-end space-x-1">
            <%= for {point, _index} <- Enum.with_index(@loss_history |> Enum.reverse()) do %>
              <div
                class="bg-blue-500 rounded-t"
                style={"
                  height: #{min(point.loss * 100, 100)}%;
                  width: #{100 / length(@loss_history)}%;
                "}
                title={"Epoch #{point.epoch}: #{Float.round(point.loss, 6)}"}
              >
              </div>
            <% end %>
          </div>
          <div class="mt-2 text-sm text-gray-600 text-center">
            Loss over time (last <%= length(@loss_history) %> epochs)
          </div>
        </div>
      <% end %>
    </div>
    """
  end
end
