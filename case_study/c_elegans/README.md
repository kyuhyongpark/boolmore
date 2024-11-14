# Case study: C. elegans lipase regulation

Here we demonstrate how to adapt `boolmore` to multi-level models.
See the supplementary information of the manuscript for details.

## File Descriptions

- `run.py` runs the model refinement process and saves the log and the final model.

- `detailed_report.py` saves tsv files containing the detailed report of model agreement with the experiments. It also compares the starting model with the final model and analyzes the dynamics of the final model.

- The `data` directory contains the necessary data to run the genetic algorithm.

    - `start.bnet` is the Boolean model to refine. It is constructed by converting the multi-level model such that each three-level variable is represented by two Boolean variables.

    - `base.bnet` is a model that contains all possible edges among Boolean variables that refer to the same edge of the interaction network, i.e., four edges for each edge between two three-level nodes. We allow `boolmore` to add any of them to the model even if they were not present in the starting model.

    - `data.tsv` contains the perturbation-observation pairs.  

    - `config.json` contains the parameters of the run and the constraints of the regulatory functions. There are three sources of constraints. First, documented interactions are included in the `regulate` or `necessary` category. The `regulate` category includes the transcriptional regulation of lipl-3 by HLH-30, the obstruction of HLH-30 activity by MXL-3, and the known information that the insulin-bound receptor DAF-2 activates DAF-16. The relationship of a protein to the mRNA from which it is translated is included in the `necessary` category. Second, constraints among Boolean variables that correspond to the same multi-level node are included in the `necessary` category. These constraints maintain the consistency of the Boolean model with a multi-level model and allow its transformation back into a multi-level model. Third, as few experimental observations document the conditions for the basal level of DAF-16 and lipl-3, the corresponding Boolean variables are allowed to become constants. All other Boolean variables must retain at least one regulator.

        ```json
        "constraints": {"fixed": [],
                        "regulate": {"lipl-3_high":["HLH-30_high"],
                                    "HLH-30_basal":["MXL-3_high"],
                                    "HLH-30_high":["MXL-3_high"],
                                    "DAF-16_high":["DAF-2"]},
                        "necessary" : {"DAF-16_high":["DAF-16_basal"],
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

- The `results` directory contains the results of the run.

    - `20231121_log.txt` is the log of the run, containing the parameters of the run. It also shows the score improvement over the iterations.

    - `20231121_6870_gen68.bnet` is the final model generated with the algorithm.

    - `start_score.tsv` provides a detailed report of the agreement of the starting model with the experiments

    - `20231121_6870_gen68_score.tsv` provides a detailed report of the agreement of the final model with the experiments





