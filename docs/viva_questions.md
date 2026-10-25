# Viva questions and concise answers

1. **What is a DNN?** A neural network with multiple learned layers between input and output.
2. **How does a DNN differ from a basic neural network?** “Deep” generally means multiple hidden layers that can learn successive representations.
3. **What is a neuron?** A computation that combines weighted inputs and a bias, then applies an activation.
4. **What are weights and biases?** Trainable parameters controlling input influence and activation offset.
5. **What is an activation function?** A nonlinear transformation that lets stacked layers model nonlinear relationships.
6. **Why use ReLU?** It is simple, computationally efficient, and commonly effective in hidden layers.
7. **Why sigmoid here?** It maps the single output into a bounded 0–1 suitability score.
8. **What is forward propagation?** Passing input through layers to calculate a prediction.
9. **What is backpropagation?** Computing gradients of loss with respect to parameters using the chain rule.
10. **What is gradient descent?** Updating parameters in a direction that reduces the loss.
11. **What is Adam?** An adaptive optimizer using estimates of first and second gradient moments.
12. **What is a loss function?** A scalar measure of prediction error used to train parameters.
13. **Why MSE instead of binary cross-entropy?** The target is a continuous proxy score, rather than a binary label.
14. **What is overfitting?** Learning training examples too closely and performing worse on unseen examples.
15. **What does dropout do?** Randomly disables a fraction of activations during training to discourage co-adaptation.
16. **Why batch normalization?** It normalizes intermediate activations during training and can stabilize optimization.
17. **Why is preprocessing required?** Features have different scales and categories; consistent encoding makes model inputs usable.
18. **What is train-test splitting?** Holding out data to estimate performance on examples not used for fitting.
19. **How does this recommender work?** It encodes each user-destination pair, predicts scores, sorts candidates, and returns explanations.
20. **Why use synthetic data?** No suitable labeled traveler feedback dataset was available; synthesis enables a reproducible learning demo.
21. **What is the key synthetic-data limitation?** Metrics measure agreement with the designed formula, not actual traveler satisfaction.
22. **How can the model improve?** Use consented real feedback, stronger baselines, unbiased evaluation splits, and updated verified data.
23. **What is TensorFlow?** An open-source machine-learning framework; Keras is its high-level model API.
24. **What is FastAPI?** A Python web framework for typed HTTP APIs, validation, and generated OpenAPI docs.
25. **How is the model deployed?** The API loads the serialized Keras model and uses it for request-time inference.
26. **Which evaluation metrics are reported?** MAE, RMSE, and R² against held-out synthetic targets.
27. **What is a recommendation system?** Software that ranks items likely to be relevant to a user.
28. **What is a content-based recommender?** A recommender that uses item attributes and user preferences.
29. **What does the suitability percentage mean?** A model-generated score rescaled for display, not a calibrated enjoyment probability.
30. **What are current limitations?** Approximate catalogue data, synthetic labels, no route-time calculation, and unverified accessibility data.
