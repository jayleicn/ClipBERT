from torch.optim import Adam, Adamax, SGD
from src.optimization.adamw import AdamW


def setup_optimizer(model, opts, model_type="transformer"):
    """model_type: str, one of [transformer, cnn]"""

    if model_type == "transformer":
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = build_optimizer_w_lr_mul(
            param_optimizer, opts.learning_rate,
            opts.weight_decay, no_decay=no_decay,
            lr_mul=opts.transformer_lr_mul,
            lr_mul_prefix=opts.transformer_lr_mul_prefix)

        if opts.optim == 'adam':
            OptimCls = Adam
        elif opts.optim == 'adamax':
            OptimCls = Adamax
        elif opts.optim == 'adamw':
            OptimCls = AdamW
        else:
            raise ValueError('invalid optimizer')
        optimizer = OptimCls(optimizer_grouped_parameters,
                             lr=opts.learning_rate, betas=opts.betas)
    else:
        assert model_type == "cnn"
        parameters = list(model.named_parameters())
        if opts.cnn_optim == "sgd":
            optimizer_grouped_parameters = build_optimizer_w_lr_mul(
                parameters, opts.cnn_learning_rate,
                opts.cnn_weight_decay,
                lr_mul=opts.cnn_lr_mul,
                lr_mul_prefix=opts.cnn_lr_mul_prefix)
            optimizer = SGD(optimizer_grouped_parameters,
                            lr=opts.cnn_learning_rate,
                            momentum=opts.cnn_sgd_momentum,
                            weight_decay=opts.cnn_weight_decay)
        elif opts.cnn_optim == "adamw":
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = build_optimizer_w_lr_mul(
                parameters, opts.cnn_learning_rate,
                opts.cnn_weight_decay, no_decay=no_decay,
                lr_mul=opts.cnn_lr_mul,
                lr_mul_prefix=opts.cnn_lr_mul_prefix)
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=opts.cnn_learning_rate, betas=opts.betas)
        else:
            raise ValueError("Only support SGD/adamW for cnn.")
    return optimizer


def build_optimizer_w_lr_mul(model_param_optimizer, learning_rate,
                             weight_decay, no_decay=[], lr_mul=1,
                             lr_mul_prefix=""):
    # Prepare optimizer
    if lr_mul_prefix == "":
        param_optimizer = model_param_optimizer
        param_top = []
    else:
        # top layer has larger learning rate
        param_top = [(n, p) for n, p in model_param_optimizer
                     if lr_mul_prefix in n and p.requires_grad]
        param_optimizer = [(n, p) for n, p in model_param_optimizer
                           if lr_mul_prefix not in n and p.requires_grad]

    optimizer_grouped_parameters = []
    if len(param_top):
        optimizer_grouped_parameters.append(
            {'params': [p for n, p in param_top
                        if not any(nd in n for nd in no_decay)],
             'lr': lr_mul*learning_rate,
             'weight_decay': weight_decay})
        if len(no_decay):
            optimizer_grouped_parameters.append(
                {'params': [p for n, p in param_top
                            if any(nd in n for nd in no_decay)],
                 'lr': lr_mul*learning_rate,
                 'weight_decay': 0.0})
    if len(param_optimizer):
        optimizer_grouped_parameters.append(
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay})
        if len(no_decay):
            optimizer_grouped_parameters.append(
                {'params': [p for n, p in param_optimizer
                            if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0})

    return optimizer_grouped_parameters


def setup_e2e_optimizer(model, opts):
    """model_type: str, one of [transformer, cnn]"""

    transformer_param_optimizer = [
        (n, p) for n, p in list(model.named_parameters())
        if "transformer" in n and p.requires_grad]
    cnn_param_optimizer = [
        (n, p) for n, p in list(model.named_parameters())
        if "cnn" in n and p.requires_grad]
    trasformer_grouped_parameters = build_e2e_optimizer_w_lr_mul(
        transformer_param_optimizer,
        opts.learning_rate, opts.weight_decay,
        lr_mul=opts.transformer_lr_mul,
        lr_mul_prefix=opts.transformer_lr_mul_prefix)
    cnn_grouped_parameters = build_e2e_optimizer_w_lr_mul(
        cnn_param_optimizer,
        opts.cnn_learning_rate, opts.cnn_weight_decay,
        lr_mul=opts.cnn_lr_mul, lr_mul_prefix=opts.cnn_lr_mul_prefix)

    optimizer_grouped_parameters = []
    optimizer_grouped_parameters.extend(trasformer_grouped_parameters)
    optimizer_grouped_parameters.extend(cnn_grouped_parameters)
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(optimizer_grouped_parameters,
                         lr=opts.learning_rate, betas=opts.betas)
    return optimizer


def build_e2e_optimizer_w_lr_mul(
        model_param_optimizer, learning_rate, weight_decay,
        lr_mul=1, lr_mul_prefix=""):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # Prepare optimizer
    if lr_mul_prefix == "":
        param_optimizer = model_param_optimizer
        param_top = []
    else:
        # top layer has larger learning rate
        param_top = [(n, p) for n, p in model_param_optimizer
                     if lr_mul_prefix in n and p.requires_grad]
        param_optimizer = [(n, p) for n, p in model_param_optimizer
                           if lr_mul_prefix not in n and p.requires_grad]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_top
                    if not any(nd in n for nd in no_decay)],
         'lr': lr_mul*learning_rate,
         'weight_decay': weight_decay},
        {'params': [p for n, p in param_top
                    if any(nd in n for nd in no_decay)],
         'lr': lr_mul*learning_rate,
         'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]
    return optimizer_grouped_parameters
