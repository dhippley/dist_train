defmodule DistTrain.NeuralNetwork.Math do
  @moduledoc """
  Mathematical operations for neural network computations.
  """

  @doc """
  Applies ReLU activation function
  """
  def relu(x) do
    Nx.max(0, x)
  end

  @doc """
  Applies sigmoid activation function
  """
  def sigmoid(x) do
    Nx.divide(1, Nx.add(1, Nx.exp(Nx.negate(x))))
  end

  @doc """
  Computes the derivative of ReLU
  """
  def relu_derivative(x) do
    Nx.select(Nx.greater(x, 0), 1.0, 0.0)
  end

  @doc """
  Computes the derivative of sigmoid
  """
  def sigmoid_derivative(x) do
    s = sigmoid(x)
    Nx.multiply(s, Nx.subtract(1, s))
  end

  @doc """
  Performs matrix multiplication and adds bias
  """
  def linear_forward(input, weights, bias) do
    Nx.add(Nx.dot(input, weights), bias)
  end

  @doc """
  Computes mean squared error loss
  """
  def mse_loss(predictions, targets) do
    diff = Nx.subtract(predictions, targets)
    Nx.mean(Nx.pow(diff, 2))
  end

  @doc """
  Computes gradient of MSE loss
  """
  def mse_loss_gradient(predictions, targets) do
    n = elem(Nx.shape(predictions), 0)
    Nx.divide(Nx.subtract(predictions, targets), n)
  end
end
