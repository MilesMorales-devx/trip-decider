# Dataset documentation

`backend/data/destinations.csv` contains 64 unique Indian destinations. Each row records name, state/administrative area, region, pipe-delimited destination types and activities, a broad climate category, approximate per-person trip cost bounds in INR, suggested day bounds, best-month numbers, a coarse crowd score (1–5), coordinates, and a short description.

## Provenance and intended use
This is a project-created educational dataset. Entries were manually assembled from general knowledge for the prototype; the catalogue has not been fact-checked against a single authoritative source or live travel provider. Do not treat prices, seasons, coordinates, access, or descriptions as current travel guidance. It contains no personal or licensed user data. Before operational use, replace or verify records using authoritative tourism boards, protected-area authorities, and dated price sources, and record URLs, retrieval dates, and licenses per row.

## Synthetic labels
The training script synthesizes profiles (budget, trip duration, month, interests, activities, weather, style, companions) and labels each profile/destination pair using an explicit weighted relevance formula in `score_label`. Its components are category overlap, activity overlap, affordability, duration fit, climate match, and seasonal suitability. The DNN learns this proxy mapping. This makes the pipeline reproducible and demonstrates supervised training, but does not establish real-world user preference accuracy.

## Validation
The API checks for a non-empty file and unique non-null destination names. Tests cover count and feature repeatability. Before extending the CSV, keep categorical values aligned with the fixed vocabularies in `backend/app/engine.py` and preserve numeric ranges and pipe delimiter conventions.
