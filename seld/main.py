import sys
from pathlib import Path

from learning import evaluate, initialize, infer, preprocess, train
from utils.cli_parser import parse_cli_overides
from utils.config import get_dataset


def main(args, cfg):
    """Execute a task based on the given command-line arguments.
    
    This function is the main entry-point of the program. It allows the
    user to extract features, train a model, infer predictions, and
    evaluate predictions using the command-line interface.

    Args:
        args: command line arguments.
    Return:
        0: successful termination
        'any nonzero value': abnormal termination
    """

    # Create workspace
    Path(cfg['workspace_dir']).mkdir(parents=True, exist_ok=True)

    # Dataset initialization
    dataset = get_dataset(dataset_name=cfg['dataset'], root_dir=cfg['dataset_dir'])

    # Preprocess
    if args.mode == 'preprocess':
        preprocessor = preprocess.Preprocessor(args, cfg, dataset)

        if args.preproc_mode == 'extract_data':
            preprocessor.extract_data()
        elif args.preproc_mode == 'extract_scalar':
            preprocessor.extract_scalar()
        elif args.preproc_mode == 'extract_meta':
            preprocessor.extract_meta()

    # Train
    if args.mode == 'train':
        train_initializer = initialize.init_train(args, cfg, dataset)
        train.train(cfg, **train_initializer)

    # Inference
    elif args.mode == 'infer':
        infer_initializer = initialize.init_infer(args, cfg, dataset)
        infer.infer(cfg, dataset, **infer_initializer)
    
    # Evaluate
    elif args.mode == 'evaluate':
        evaluate.evaluate(cfg, dataset)

    return 0


if __name__ == '__main__':
    args, cfg = parse_cli_overides()
    sys.exit(main(args, cfg))
