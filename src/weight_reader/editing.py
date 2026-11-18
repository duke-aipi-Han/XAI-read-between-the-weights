"""Explicit, copy-on-write checkpoint edits for exploratory experiments."""
import copy
import math

import torch


def edit_weights(model, parameter, index, scale):
    """Multiply an indexed weight or bias slice, leaving the original model intact.

    An index is a list of integers or nulls; null selects the full dimension.
    """
    parameters = dict(model.named_parameters())
    if parameter not in parameters:
        raise ValueError('unknown parameter')
    tensor = parameters[parameter]
    if not isinstance(index, list) or len(index) != tensor.ndim:
        raise ValueError('index must specify each parameter dimension')
    for item, size in zip(index, tensor.shape):
        if item is not None and (isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < size):
            raise ValueError('indices must be null or valid nonnegative integers')
    if isinstance(scale, bool) or not isinstance(scale, (float, int)) or not math.isfinite(scale):
        raise ValueError('scale must be finite')
    edited = copy.deepcopy(model)
    selection = tuple(slice(None) if i is None else i for i in index)
    with torch.no_grad():
        target = dict(edited.named_parameters())[parameter]
        target[selection] *= scale
        if not torch.isfinite(target).all():
            raise ValueError('edit produced non-finite weights')
    return edited


def validation_metrics(model, split):
    """Compute accuracy only on the explicitly provided development split."""
    x, y = split
    with torch.no_grad():
        correct = (model(x) > 0) == y.bool()
    result = {'accuracy': correct.float().mean().item(), 'count': len(x), 'quadrants': {}}
    for name, s0, s1 in [('Q++', 1, 1), ('Q+-', 1, -1), ('Q-+', -1, 1), ('Q--', -1, -1)]:
        mask = (x[:, 0] * s0 > 0) & (x[:, 1] * s1 > 0)
        result['quadrants'][name] = {'count': int(mask.sum()), 'accuracy': correct[mask].float().mean().item() if mask.any() else None}
    return result
