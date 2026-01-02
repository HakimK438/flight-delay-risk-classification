# flight-delay-risk-classification

I worked on this project to understand how flight delays can be approached from a risk perspective, rather than trying to predict exact delay minutes. In real situations, knowing whether a flight is likely to be low risk or high risk before departure is often more useful than estimating how late it might arrive.

The focus here is on predicting three outcomes: **Low Risk, Moderate Risk, and High Risk**. All inputs are limited to information that would realistically be available before a flight departs, such as schedule timing, route information, weather conditions, wind speed, and congestion at the airport. Any information that directly reflects what happened after the flight, or that could leak the outcome, was intentionally removed during modeling.

While exploring the data and building models, one thing became clear very quickly: accuracy alone does not tell the full story. The dataset is imbalanced, and some mistakes are far more costly than others. Predicting a high-risk flight as low-risk is a much bigger problem than the opposite. Because of this, evaluation focused more on macro-averaged metrics and class-wise behavior rather than just overall accuracy.

I started with Logistic Regression to establish a simple reference point. Initially, its performance was weak, especially when it came to identifying high-risk flights. After addressing convergence issues and class imbalance, the model improved and became more reasonable. Even then, the confusion matrix showed that high-risk cases were still often mixed with moderate ones, which suggested that delay risk is not driven by simple linear relationships.

Tree-based models made this limitation more visible. A Decision Tree captured non-linear patterns and improved high-risk detection, but its behavior changed noticeably across different data splits. That variability made it harder to rely on as a final model.

The Random Forest model handled these interactions more consistently. It reduced the number of high-risk flights being missed and showed more stable behavior across validation folds. The improvement was visible not just in cross-validated scores, but also in how errors were distributed. In this context, reducing dangerous misclassifications mattered more than small gains in overall accuracy.

Throughout the process, confusion matrices were used to understand how models fail, not to claim performance on their own. Cross-validation helped judge robustness, while confusion matrices explained why one model was more suitable than another.



Learning reference: My understanding of machine learning concepts was strengthened through long-form YouTube lectures by CampusX (Nitish Singh).
