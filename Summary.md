# Results
In this competition there does not appear to be very much separation between the best submissions and the top half. This is not especially surprising given there are not many features present in the dataset that appear to show a strong relationship to the dependent variable.


## Model - LightGbm
Tabular datasets are typically very well suited for `GBDT` models and the prompt specifically requested one. I've chosen to use lightgbm because it is usually faster to train, requires less memory, often performs slightly better and because it is the api I'm most familar with.


## Baseline - example kernel shared
I've included the baseline from the sample kernel sent as well as compared against an extremely naive model that always predicts zero (achieving 96% accuracy and demonstrating why accuracy is a useless metric for this task).


## Evaluation
 - Gini
 - ROC Curve


## Summary
In Kaggle success for this tournament involves optimizing models to the 4th decimal place of precision on a dataset dependent metric (GINI). This is very rarely a useful excercise in the real world and not something I chose to spend a significant portion of time on. Instead, after verifying that my model performs adequately on my cross-validation. I chose to package up and document the process, and focus on feature elimination since many features here are clearly worthless. In the real world eliminating unimportant features results in a more robust model and interpretable model. Next steps may look like creating a REST API to serve real time responses for requests for insurance. I might include something like a shap endpoint that will return not only the probability of the user filing a claim but the marginal contribution of each feature to the final prediction.