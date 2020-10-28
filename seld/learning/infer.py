import torch
from utils.config import get_afextractor, get_inferer, get_models


def infer(cfg, dataset, **infer_initializer):
    """ Infer, only save the testset predictions

    """
    submissions_dir = infer_initializer['submissions_dir']
    ckpts_paths_list = infer_initializer['ckpts_paths_list']
    ckpts_models_list = infer_initializer['ckpts_models_list']
    test_generator = infer_initializer['test_generator']
    cuda = infer_initializer['cuda']
    preds = []
    for ckpt_path, model_name in zip(ckpts_paths_list, ckpts_models_list):
        print('=====>> Resuming from the checkpoint: {}\n'.format(ckpt_path))
        af_extractor = get_afextractor(cfg, cuda)
        model = get_models(cfg, dataset, cuda, model_name=model_name)
        state_dict = torch.load(ckpt_path)
        model.module.load_state_dict(state_dict['model'])
        print('  Resuming complete\n')
        inferer = get_inferer(cfg, dataset, af_extractor, model, cuda)
        pred = inferer.infer(test_generator)
        preds.append(pred)
        print('\n  Inference finished for {}\n'.format(ckpt_path))
    inferer.fusion(submissions_dir, preds)


