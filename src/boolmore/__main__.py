def main():
    import boolmore.genetic_algorithm as ga
    from sys import argv
    """
    python -m boolmore <parameters file> <starting model>

    <parameters file> is the path to the parameters json, e.g. "CAD_config.json"
    <starting model> is the path to a starting model file, e.g. "CAD_start.txt"
    if <starting mdoel> is not provided, the base model will be the starting model.

    This will run the algorithm and export the models with the top score.

    Example Usage:
    cd boolmore/demo
    python -m boolmore "CAD_config.json" "CAD_start.txt"
    """

    print("""
    This function is for demonstration. It runs the boolmore Boolean model refinement
    routine and exports the model with the top score. Default function parameters are
    used. See ReadMe for instructions regarding typical use cases.
    """)

    config_json = argv[1]
    if len(argv) == 3:
        model = argv[2]
    else:
        model = None

    ga.run_ga(config_json, model)

if __name__=='__main__':
    main()