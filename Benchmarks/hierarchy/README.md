## Comparing hierarchy scoring with non-hierarchy scoring

Here we illustrate that the hierarchy scoring has benefit in refining models to have higher predictive power.

`generate_data` produces data for this comparison. It takes 8 sampled models and generates data for each of them: 5 randomized starting models, 5 training data, and 5 validation data. Their numberings equal to their random seeds used to generate them. The training data consists of 8\*N experiments. The validation data consists of 2\*N experiments. (It also contains 8\*N experiments from the training data so that 2\*N experiments are scored with a proper hierarchy. However, experiments of the training data have weight 0, and does not affect the total score.)

