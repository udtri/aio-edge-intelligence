# Google TimesFM provider

The `timesfm` provider adds zero-shot forecasting without changing the service API. It accepts the repository-standard input shape—`[batch, channels, sequence]`—and returns `[batch, channels, horizon]` point forecasts.

## Choose a checkpoint

| Checkpoint | Behavior | Weight license | Use it for |
|---|---|---|---|
| `google/timesfm-2.5-200m-pytorch` | Forecasts each channel independently | Apache-2.0 | The default, including commercial evaluation |
| `google/timesfm-3.0-pytorch` | Joint multivariate forecasting with optional covariates | Non-commercial | Research and non-production experiments |

TimesFM 3.0 is newer, but its current checkpoint license does not permit commercial or production use. This project therefore defaults to TimesFM 2.5 when the `timesfm` provider is constructed directly.

## Run TimesFM 2.5

```bash
MODEL_PROVIDER=timesfm \
MODEL_NAME=google/timesfm-2.5-200m-pytorch \
MODEL_DEVICE=cpu \
DEFAULT_TASK=forecasting \
docker compose -f deploy/standalone/docker-compose.yaml up --build
```

Then call the existing forecast endpoint:

```bash
curl http://localhost:8080/infer/forecast \
  --header 'content-type: application/json' \
  --data '{
    "data": {
      "sensor_id": "motor-7/vibration",
      "values": [0.11, 0.13, 0.12, 0.16, 0.19, 0.18]
    },
    "forecast_horizon": 3
  }'
```

The checkpoint downloads from Hugging Face on first start. Cache the weights in the container image or a persistent volume before using disconnected edge sites.

## Run TimesFM 3.0

Set `MODEL_NAME=google/timesfm-3.0-pytorch`. The provider preserves channel relationships and supports `past_only_covariates` and `past_future_covariates` when called directly from Python. The current HTTP schema remains intentionally univariate; exposing covariates is tracked in the roadmap.

## Operational notes

- TimesFM implements forecasting only. The anomaly endpoint returns `422` while this provider is active.
- TimesFM 2.5 compiles a maximum context of 1,024 points and horizon of 256 points by default.
- Model quantiles are retained in `ModelResult.metadata`; the current HTTP response returns the point forecast.
- Evaluate accuracy, latency, memory, and failure behavior on your own sensor distribution before deployment.

Sources: [Google Research TimesFM](https://github.com/google-research/timesfm) and the [TimesFM 3 announcement](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/).
