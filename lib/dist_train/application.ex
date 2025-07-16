defmodule DistTrain.Application do
  # See https://hexdocs.pm/elixir/Application.html
  # for more information on OTP Applications
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      DistTrainWeb.Telemetry,
      # DistTrain.Repo,  # Commented out for demo - we don't need database
      {DNSCluster, query: Application.get_env(:dist_train, :dns_cluster_query) || :ignore},
      {Phoenix.PubSub, name: DistTrain.PubSub},
      # Start the Finch HTTP client for sending emails
      {Finch, name: DistTrain.Finch},
      # Start neural network components
      DistTrain.NeuralNetwork.NodeManager,
      DistTrain.NeuralNetwork.Coordinator,
      # Start to serve requests, typically the last entry
      DistTrainWeb.Endpoint
    ]

    # See https://hexdocs.pm/elixir/Supervisor.html
    # for other strategies and supported options
    opts = [strategy: :one_for_one, name: DistTrain.Supervisor]
    Supervisor.start_link(children, opts)
  end

  # Tell Phoenix to update the endpoint configuration
  # whenever the application is updated.
  @impl true
  def config_change(changed, _new, removed) do
    DistTrainWeb.Endpoint.config_change(changed, removed)
    :ok
  end
end
