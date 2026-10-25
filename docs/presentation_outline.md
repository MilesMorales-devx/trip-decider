# Presentation outline (11 slides)

1. **Title** — Trip Decider; student/team, department, guide.
2. **Problem** — Too many destinations; budget, time, and interests differ.
3. **Objectives and scope** — Personalized India-focused shortlist; educational DNN; no booking or live prices.
4. **Existing approach and gap** — Popularity lists and manual filters; limited combined preference ranking.
5. **Proposed workflow** — Questionnaire → encoder → trained model → ranked results and explanations.
6. **Dataset** — 64 destinations, fields, approximate hand-curated provenance, synthetic label rationale.
7. **Feature engineering** — Fixed vectors for budget, days, categories, activities, weather, month, and destination attributes.
8. **DNN architecture** — 128/64/32 dense layers, batch normalization, dropout, sigmoid score; Adam and MSE.
9. **Application and API** — React/Vite UI, FastAPI endpoints, saved places, comparison.
10. **Evaluation and limitations** — Show only actual generated MAE/RMSE/R²; metrics are against synthetic labels; explain data and travel-data limitations.
11. **Conclusion and future work** — Full-stack learning outcome, real feedback and verified data needed; questions.
