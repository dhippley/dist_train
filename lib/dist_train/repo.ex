defmodule DistTrain.Repo do
  use Ecto.Repo,
    otp_app: :dist_train,
    adapter: Ecto.Adapters.Postgres
end
