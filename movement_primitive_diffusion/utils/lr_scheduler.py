from typing import Union, Optional

from transformers import SchedulerType
from transformers import get_scheduler as huggingface_get_scheduler
from torch.optim import Optimizer


def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    warmup_proportion: Optional[float] = None,
    use_proportional_warmup: Optional[bool] = None,
    **kwargs,
):
    """Added kwargs vs diffuser's original implementation

    From: https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/model/common/lr_scheduler.py

    Unified API to get any scheduler from its name.

    Args:
        name (Union[str, SchedulerType]): The name of the scheduler to use.
        optimizer (torch.optim.Optimizer): The optimizer that will be used during training.
        num_warmup_steps (Optional[int]): The number of warmup steps to do. This is not required by all schedulers (hence the argument being optional),
            the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (Optional[int]): The number of training steps to do. This is not required by all
            schedulers (hence the argument being optional), the function will raise an error if it's unset and the scheduler type requires it.
    """
    name = SchedulerType(name)
    if name == SchedulerType.CONSTANT:
        return huggingface_get_scheduler(name=name, optimizer=optimizer, **kwargs)

    if use_proportional_warmup:
        if warmup_proportion is None:
            raise ValueError(
                "Can't use proportional warmup without specifying `warmup_proportion`."
            )
        num_warmup_steps = int(warmup_proportion * num_training_steps)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return huggingface_get_scheduler(name=name, optimizer=optimizer, num_warmup_steps=num_warmup_steps, **kwargs)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return huggingface_get_scheduler(name=name, optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, **kwargs)
