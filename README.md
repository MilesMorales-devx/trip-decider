# TRIP DECIDER — AI-Powered Travel Destination Recommendation System

## Abstract
Trip Decider is an educational full-stack recommendation system that ranks Indian travel destinations using a trained TensorFlow/Keras deep neural network. Visitors describe their budget, trip length, interests, weather, travel companions, and month, then receive an explainable shortlist with estimated costs and activity information. The current destination catalogue contains 64 records. Training labels are synthetic and generated from documented, transparent attribute-overlap rules; scores are therefore educational model outputs, not validated enjoyment probabilities or travel advice.

## Introduction and problem statement
Travelers must balance limited time and money with different interests and seasonal preferences. Broad popularity lists do not account for those differences. This project demonstrates how a supervised neural network can combine user and destination features and rank candidate places.

## Objectives and scope
- Demonstrate an end-to-end DNN training and inference pipeline.
- Provide a responsive questionnaire and recommendations for destinations in India.
- Explain recommendations from destination attributes and user preferences.
- Keep core features independent of external APIs. Images are illustrative Unsplash images and require internet access; the app works without them using image placeholders only if the browser blocks remote images.
- International destinations, live transit, booking, verified prices, and real user feedback are outside the current scope.

## Existing and proposed systems
Generic travel listings tend to sort by popularity and often require many filters. Trip Decider first captures user preferences, encodes them with destination attributes, predicts a suitability score using the trained DNN, and explains key attribute matches. Budget and trip-duration mismatches are clearly identified.

## Technologies
Python, TensorFlow/Keras, NumPy, pandas, scikit-learn, FastAPI, Pydantic, React, TypeScript, Vite, Lucide React, CSS, pytest, HTTPX.

## Requirements
Python 3.9–3.11 (TensorFlow 2.13.1), Node.js 20+ and npm, 4 GB RAM minimum (8 GB recommended for model training), and an internet connection for dependency installation / remote photos. The project was authored against Python 3.9.6 and Node 26.5.0. Model training and API execution require the dependencies below.

## System architecture
1. `backend/data/destinations.csv` stores 64 Indian destinations and their attributes.
2. `backend/scripts/train_model.py` creates reproducible synthetic preference/destination examples, fits the Keras network, computes holdout metrics, and saves the model plus feature schema.
3. FastAPI loads that artifact, runs inference over all destinations, ranks scores, and generates explanations from recorded attributes.
4. The React questionnaire calls the API. Browser local storage keeps saved destinations.

See [docs/architecture.mmd](docs/architecture.mmd).

## Dataset and provenance
The CSV is a small hand-curated educational catalogue assembled for this project from general destination knowledge. Cost ranges, durations, seasons, descriptions, and crowd levels are approximate illustrative values, not sourced live quotations. It does not contain personal information. See [docs/dataset_documentation.md](docs/dataset_documentation.md). Validate every price, permit, season, and accessibility detail before relying on it.

## Preprocessing and feature design
Training and inference call the same deterministic `encode` function in `backend/app/engine.py`. It normalizes budget, days, month, destination cost/duration/crowd; adds fixed-order interest/activity overlap and destination-attribute indicators; then encodes weather, style, companions, season, and optional preference flags. The input vector has a fixed dimension and schema metadata is saved beside the model. This avoids train/serve transformation drift.

## DNN architecture and training
The network uses Dense(128, ReLU), Batch Normalization, Dropout(0.25), Dense(64, ReLU), Dropout(0.20), Dense(32, ReLU), and Dense(1, sigmoid). Adam optimizes mean squared error because the synthetic target is a continuous relevance score in [0,1]. The script seeds NumPy and TensorFlow, creates 5,000 preference profiles paired with all destinations, and uses a held-out 20% split. Early stopping monitors validation loss. MAE, RMSE, and R² are calculated and written to `backend/models/metrics.json`. The executed run yielded MAE 0.0570, RMSE 0.0758, and R² 0.7599 on 64,000 held-out synthetic examples. These metrics describe agreement with synthetic labels, not traveler satisfaction.

## Recommendation and evaluation
The model score is shown as a percentage and called a model-generated suitability score. It is not a probability that the traveler will enjoy a trip. The API ranks strict budget/duration matches when at least five are available; otherwise it returns broader ranked results with explicit cost and duration mismatch explanations. Metrics quantify agreement with synthetic labels, not real traveler satisfaction. A future evaluation should use consented explicit feedback, a baseline comparison, and a temporal or user-separated evaluation split.

## API
Base URL: `http://localhost:8000`

- `GET /api/health` — API and dataset status.
- `GET /api/destinations` — full destination catalogue.
- `GET /api/destinations/{name}` — one destination, or 404.
- `POST /api/recommendations` — validated questionnaire JSON; returns ten sorted destinations and the synthetic-label notice. Missing model artifact returns 503.
- `POST /api/budget-estimate` — simple arithmetic estimate from `days`, `daily_budget`, `stay_cost`, `travel_cost`, `activities_cost`.
- `POST /api/compare` — two or three destination names.

Interactive API docs are available at `/docs` when the backend is running.

## Frontend features
Responsive landing page, five-step preference form, input validation and progress, preference-aware scorecards, destination detail modal, budget snapshot, compare selection, search, and saved places in browser local storage. The UI states that training labels are synthetic. Remote destination photography is decorative.

## Installation
Use Python 3.9–3.11. From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

Train and save the network:

```bash
cd backend
python scripts/train_model.py
cd ..
```

Start the backend in one terminal:

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --app-dir backend
```

Or from inside `backend`, use `uvicorn app.main:app --reload`.

Start the frontend in another terminal:

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

Open `http://localhost:5173`. The API defaults to `http://localhost:8000`; set `VITE_API_URL` in `frontend/.env` to change it. CORS allows the local Vite origins.

## Run from VS Code

Open `Trip Decider.code-workspace` in VS Code. The project includes VS Code tasks for starting both app servers together, training the model, and running backend tests. Choose **Terminal → Run Task → Trip Decider: Run App**, then open `http://localhost:5173`. To stop the servers, terminate the two Trip Decider task terminals.

The workspace file points VS Code to `.venv/bin/python`; create the virtual environment and install backend/frontend dependencies first using the setup steps above.

## Testing
From the project root with backend dependencies installed:

```bash
PYTHONPATH=backend pytest backend/tests -q
cd frontend && npm run build
```

The API recommendation test checks that missing model artifacts are reported as 503, or that recommendations work when the model exists. The frontend production build passed. Backend tests passed after the model was trained; see the final run status in this project workspace. See `backend/tests/test_api.py` for included automated checks.

## Results and screenshot checklist
The executed metrics are in `backend/models/metrics.json`. For a college report, capture: landing page, each questionnaire stage, one successful ranked list, destination detail, saved list after refresh, comparison selection, and API docs. Do not enter invented model metrics.

## Limitations
- Synthetic targets are a designed proxy and can encode bias in their weights.
- Catalogue costs, seasons, activity access, and descriptions are approximate and not continuously maintained.
- The starting city and optional maximum travel time are collected, but route time is not modeled yet; the UI discloses that this preference does not currently affect ranking.
- Accessibility requirements are collected but the catalogue lacks verified accessibility data.
- Some questionnaire flags currently affect model input but do not have independently validated semantics.
- No real user satisfaction or booking data has been collected.

## Future scope
Gather consented feedback; add verified, dated destination and accessibility data; evaluate against content-based baselines; include origin-specific transit estimates; build offline image assets; add itinerary export and an optional map; deploy with appropriate monitoring and data stewardship.

## References
- TensorFlow Keras Sequential model documentation: https://www.tensorflow.org/guide/keras/sequential_model
- Keras Dense layer documentation: https://keras.io/api/layers/core_layers/dense/
- Keras BatchNormalization documentation: https://keras.io/api/layers/normalization_layers/batch_normalization/
- Keras Dropout documentation: https://keras.io/api/layers/regularization_layers/dropout/
- FastAPI documentation: https://fastapi.tiangolo.com/
- scikit-learn model evaluation documentation: https://scikit-learn.org/stable/modules/model_evaluation.html
- Dataset: project-created illustrative catalogue, documented in `docs/dataset_documentation.md`; it is not a published benchmark dataset.
