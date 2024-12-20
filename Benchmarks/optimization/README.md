# Parameter Optimization
`Boolmore` contains multiple tunable parameters. Here we describe the process of our parameter analysis and present general conclusions. We found significant robustness to changes in these parameter values. See also the Methods section of the manuscript and Supplementary Information Text S4 ([bioRxiv](https://www.biorxiv.org/content/10.1101/2023.11.14.567002v2)).

## Models and data

### Sample models
We explored multiple values of each parameter for a compilation of 8 models from the Cell collective. These 8 models were selected to exemplify a variety of properties: small (<20) or large (>70) number of nodes, presence or absence of source nodes, single or multiple point attractors, single or multiple complex attractors.

| network size | single point attractor | many point attractors | single complex attractor | many attractors including complex | source nodes |
| -----------: | :--------------------: | :-------------------: | :----------------------: | :-------------------------------: | :----------- |
|        small | Cell Cycle Transcription by Coupled CDK and Network Oscillators |                     |                          | T-LGL Survival Network 2011 Reduced Network       | none         |
|        small |          |  Metabolic Interactions in the Gut Microbiome  | Mamallian Cell Cycle 2006  |                                 | exist       |
|        large | IL-1 Signaling | Glucose Repression Signaling 2009 | Signalling in Macrophage Activation | Influenza A Virus Replication Cycle  | exist |

We used each model to generate 5 sets of 10*N artificial experiments. We generated five starting models for each model by randomizing the binary representation of the update functions of the original model.

### ABA model and scrambled versions of the model
We used the baseline A model and the real experimental data for the ABA case study. We also generated 25 starting models by randomizing the binary representation of the update functions of the baseline model while keeping the biological constraints.


## Parameters
For all the following analyses we generated 100 models in a single run.

### Mutation probability
The mutation probability denotes the probability of changing (flipping) each bit of the binary representation of each regulatory function. A high mutation probability is an appropriate choice for starting models with a low fitness and a low mutation probability is a suitable choice for higher-fitness starting models. Probability settings with iteration-dependent probabilities (e.g. high mutation probability for the initial exploration, followed by low mutation probability for refinement) may be the best choice in certain use cases. See Supplementary information Text S4 for an illustrative description of the optimization of the mutation probability.

Setting: 20 iterations, 5 models per iteration, keep 2, mix 0

#### 8 sampled models
We tested 5 probability values, [0.01, 0.05, 0.1, 0.2, 0.5], with 200 runs for each (8 models * 5 starting models * 5 sets of artificial experiments). A constant probability of 0.1 had the highest final accuracy, closely followed by 0.05 and 0.2. An adaptive probability setting with a value of 0.5 for the first 3 iterations and a value of 0.1 afterward `{1:0.5 4:0.1}` yielded the best accuracy.

#### ABA model
We tested 6 probability values, [0.005, 0.01, 0.05, 0.1, 0.2, 0.5]. 50 independent runs per probability value with baseline A as the starting model showed that a constant probability of 0.1 led to the highest accuracy, closely followed by 0.005. Starting from 25 randomized models per probability value, the probability that yielded the best accuracy was 0.05, followed by 0.1 and 0.01.

### The number of iterations / generated models per iteration
Since we fixed the number of generated models per run to 100, the number of iterations determined the generated models per iteration. We tested 7 iteration settings, [1, 2, 5, 10, 20, 50, 100]. One iteration generates 100 models at once, hence it is equivalent to a simple random sampling.

Setting: probability of 0.2 for the sampled models, probability of 0.01 for the ABA model, keep 1, mix 0

For the sampled models, we had 200 runs (8 models * 5 starting models * 5 sets of artificial experiments) for each iteration value. 100 iterations showed the best accuracy, closely followed by 20, 50, and 10.

For the ABA model, we had 50 independent runs for each iteration value. Using 50 iterations showed the best accuracy, followed by 100, 10, then 20.

### The number of top-scoring models kept at each step (keep)
This parameter determines the number of models kept, to be chosen as the mutation targets for the next iteration. Note that 'generated models per iteration' is a separate parameter and can be specified independently. The number to keep can be higher than the number generated at each iteration. For example, the number of generated models per iteration can be 5, while we keep 15. In this case, 15 top-scoring models generated so far will be 'kept' and 5 will be chosen to mutate and generate 5 new models. The 15 top-scoring models among these 20 models will be kept again. At the beginning of the iterations, when there are fewer generated models than the 'keep' parameter, we use copies of the starting model. This prevents models worse than the starting model from being kept. We tested 7 keep settings, [1,2,3,4,5,10,15].

Setting: probability of 0.2 for sampled models, probability of 0.01 for the ABA model, 20 iterations, 5 models per iteration, mix 0

For the sampled models, we had 200 runs (8 models * 5 starting models * 5 sets of artificial experiments) for each keep value. Keeping 4 showed the best accuracy, closely followed by 2, 1, 3, then 5.

For the ABA model, we had 50 independent runs for each keep value. Keeping 1 showed the best accuracy, closely followed by 2, 5, 4, then 3.

### The number of models generated by crossing existing models (mix)
`Boolmore` can also generate a new model by crossing two existing models. This is done by randomly taking functions from either of the two models and then mutating them. Note that since the number of generated models is fixed per run, if a model is generated by crossing existing models, the tradeoff is that there is no opportunity to fine-tune the best model. We tested 5 mix settings, [0, 1, 2, 3, 4].

Setting: probability 0.2 for sampled models, probability 0.01 for the ABA model, 20 iterations

For the sampled models, we had 200 runs (8 models * 5 starting models * 5 sets of artificial experiments) for each mix value. Mixing 1 showed the best accuracy, closely followed by 0, 2.

For the ABA model, we had 50 independent runs for each keep value. Mixing none showed the best accuracy, followed by 1, 2, 3, 4, which shows that mixing was ineffective.

## Conclusions
Best settings:
1. mutation probability : 0.1 for the sampled models and 0.01 for the ABA model (0.05 for the randomized ABA models).
2. number of iterations : 100 for the sampled models and 50 for the ABA model.
3. number of top-scoring models kept at each step : 4 for the sampled models and 1 for the ABA model.
4. number of models generated by crossing existing models : 1 for the sampled models and 0 for the ABA model.

The performance of the genetic algorithm does not depend sensitively on the choice of the parameters. The most important choice is the selection of the mutation probabilities, and the other parameters have a negligible impact on the performance.

General guidelines for future uses of `boolmore`:
1. The mutation probabilities should be in the range of 0.01-0.1, smaller if the starting model has high accuracy.
2. The number of iterations should be over a certain threshold, i.e. larger than 10 when 100 models are generated.
3. The number of top-scoring models kept at each step can be any nonzero number as long as it is not larger than the number of models generated for each iteration, i.e., between 1 and 4 when 5 models are generated at each iteration.
4. The number of models generated by crossing existing models should be small, i.e., 1 when 5 models are generated for each iteration.

## File descriptions

`data` directory contains the starting models of the 8 sampled models from the Cell Collective, and their artificial experiment data. It also contains the 25 scrambled ABA models.

`scripts` directory contains scripts used to generate files in the `data` directory, as well as scripts used for running multiple runs of the genetic algorithm to compare different parameter values.

`results` directory contains the logs from running scripts in the `scripts` directory. It also contains Jupyter notebooks that generate figures to compare the different parameter values for each parameter type.
