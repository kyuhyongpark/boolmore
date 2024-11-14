# Case study: C elegans

Here we demonstrate adapting `boolmore` to multi-level models.
See supplementary information of the manuscript for details.

## File Description

- `run.py` runs the refinement process and saves the log and the final model.

- `detailed_report.py` saves tsv files containing the detailed report of agreement with the experiments. It also compares the starting model with the final model and analyzes the dynamics of the final model.

- `data` directory contains the necessary data to run the genetic algorithm.

    - `start.bnet` is the Boolean model to refine. It is constructed by converting the multi-level model.

    - `base.bnet` is a model used to simply contain all possible edges, allowing `boolmore` to add them back to the start model even if they were not present in it.

    - `data.tsv` contains the perturbation-observation pairs.  

    - `config.json` contains the parameters and the constraints for the run. Notably, it contains constraints that allows the Boolean model to be intepreted and transformed back into a multi-level model.

        ```json
        "constraints": {"fixed": [],
                        "regulate": {"lipl-3_high":["HLH-30_high"],
                                    "HLH-30_basal":["MXL-3_high"],
                                    "HLH-30_high":["MXL-3_high"],
                                    "DAF_16_high":["DAF-2"]},
                        "necessary" : {"8-nitro-cGMP":["cGMP"],
                                    "KOUT":["Depolarization"],
                                    "Malate":["PEPC", "AnionEM"],
                                    "ROS":["NADPH", "RBOH"],
                                    "DAF-16_high":["DAF-16_basal"],
                                    "mTORC1_high":["mTORC1_basal"],
                                    "MXL-3_basal":["mxl-3_basal"],
                                    "MXL-3_high":["MXL-3_basal"],
                                    "HSF-1_high":["HSF-1_basal"],
                                    "mxl-3_high":["mxl-3_basal"],
                                    "hlh-30_high":["hlh-30_basal"],
                                    "HLH-30_basal":["hlh-30_basal"],
                                    "HLH-30_high":["HLH-30_basal"],
                                    "lipl-3_high":["lipl-3_basal"]},
                        "group": {},
                        "possible_constant": ["DAF-16_basal", "lipl-3_basal"]},
        ```

- `results` directory contains the results of the run.

    - `20231121_log.txt` is the log of the run, containing the parameters of the run. It also shows the score improvement over the iterations.

    - `20231121_6870_gen68.bnet` is the final model generated with the algorithm.

    - `start_score.tsv` provides a detailed report of the agreement of the starting model with the experiments

    - `20231121_6870_gen68_score.tsv` provides a detailed report of the agreement of the final model with the experiments





