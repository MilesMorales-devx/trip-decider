# Trip Decider: AI-Powered Travel Destination Recommendation System

## Abstract
Trip Decider is an educational full-stack system that ranks 64 Indian destinations against a traveler's budget, duration, interests, climate preference, activities, and travel month. A TensorFlow/Keras dense neural network predicts a continuous suitability score from a shared deterministic feature representation. FastAPI serves recommendations and React provides a responsive questionnaire and destination interface. Because no labeled traveler feedback dataset is included, the training targets are reproducibly generated from a documented attribute-matching formula. The work demonstrates an end-to-end deep learning pipeline; it does not claim validated real-world recommendation accuracy.

## 1. Introduction
Choosing a destination requires balancing constraints and preferences. Static lists and keyword filters can be difficult to personalize. A recommender can represent users and candidate destinations in a common feature space, estimate relevance, and sort alternatives. This project is designed as a third-year undergraduate demonstration of data preparation, model training, API serving, and a user-facing application.

## 2. Background overview
Recommendation systems commonly use collaborative filtering, content-based methods, or hybrid approaches. Collaborative methods use interaction histories; content-based methods compare user preferences with item attributes. Trip Decider uses destination and preference attributes because a trustworthy interaction dataset is unavailable. A feed-forward neural network can learn nonlinear combinations of these attributes. TensorFlow/Keras provides model composition, training, and serialization. The project cites the official framework documentation in the README rather than claiming a novel method.

## 3. Problem formulation
For user profile `u` and destination `d`, construct feature vector `x(u,d)` and estimate `s = fθ(x)`, where sigmoid output `s ∈ (0,1)`. Sort destinations by `s` and provide attribute-grounded explanations. Suitability is a ranking score, not a calibrated probability. The training target is synthetic relevance `y(u,d)` calculated from attribute overlaps and simple constraint-fit terms.

## 4. Methodology
The CSV catalog stores destination types, activities, climate, cost range, duration range, seasons, and contextual metadata. Seeded random user profiles are paired with every catalog item. The shared `encode` method applies fixed vocabularies and numeric scaling in both training and inference. A seeded holdout split separates training and evaluation examples. Early stopping uses validation loss. Training computes MAE, RMSE, and R² on held-out synthetic examples.

## 5. Architecture
The model input combines user and destination features. Layers are Dense 128/ReLU, batch normalization, dropout 0.25, Dense 64/ReLU, dropout 0.20, Dense 32/ReLU, then a one-unit sigmoid output. Adam is the optimizer. Mean squared error is used because the proxy target is continuous. The saved Keras artifact and JSON feature schema are placed in `backend/models/`.

## 6. Implementation
FastAPI validates request data, loads the saved model, scores every destination, sorts the candidates, and returns the top ten with cost fit, duration fit, metadata, and explanations. It restricts to strict budget/duration matches when at least five exist, and otherwise returns broader results with mismatch notes. React/Vite presents a five-step form, ranked cards, detail modal, comparison controls, and saved destinations. Saved places use browser local storage. CORS permits the local Vite origin.

## 7. Results
The executed holdout evaluation used 64,000 synthetic examples and produced MAE 0.0570, RMSE 0.0758, and R² 0.7599. Full run metrics are in `backend/models/metrics.json`. These figures measure agreement with the synthetic relevance formula only; they do not prove traveler satisfaction. The backend test suite completed with 10 passing tests.

## 8. Discussion
The project demonstrates the full flow from dataset through model to API and UI. A DNN is used to teach model training and serving, even though a transparent content-based scoring baseline may be more appropriate for the current small synthetic dataset. The main value is educational pipeline integration. The reported score is only as meaningful as its target labels.

## 9. Limitations
The catalog is illustrative and manually assembled; costs and seasons may be stale or incorrect. Synthetic labels reflect hand-chosen weights and cannot establish user preference quality. The model does not compute routes from the origin city or apply the optional maximum-travel-time preference. Accessibility information is not verified. No real feedback, calibration, or external validity study exists. Remote Unsplash photography requires connectivity.

## 10. Conclusion
Trip Decider implements the structure of a DNN-based travel recommendation application with reproducible synthetic supervision, reusable preprocessing, a REST API, and a responsive UI. It should be treated as a college demonstration until real data, testing, and user evaluation are completed.

## 11. Future scope
Collect explicit consented ratings; validate cost/season/accessibility data with dated authoritative sources; compare against content-based baselines; perform user-separated evaluation; include distance and transport estimates; and deploy with model versioning, privacy controls, and monitoring.

## References
See the dated framework and API references in the project README. The destination catalogue is project-created, not a published dataset.
