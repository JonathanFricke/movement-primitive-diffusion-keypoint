from typing import Dict
import torch

from movement_primitive_diffusion.aggregators.base_aggregator import Aggregator


class ConcatenateAggregator(Aggregator):
    def __call__(self, encodings: Dict[str, torch.Tensor]):
        """Concatenate the values of the given data dictionary along the last dimension.

        Args:
            data (Dict[str, torch.Tensor]): The data to aggregate.

        Examples:
            >>> encodings = {"x": torch.tensor([[1, 2, 3]]), "y": torch.tensor([[4, 5, 6]])}
            >>> ConcatenateAggregator().aggregate(encodings)
            tensor([[1, 2, 3, 4, 5, 6]])
        """

        return torch.cat(list(encodings.values()), dim=-1)

class ConcatenateAggregator_(Aggregator):
    def __call__(self, encodings: Dict[str, torch.Tensor]):
        """Concatenate the values of the given data dictionary along the last dimension.

        Args:
            data (Dict[str, torch.Tensor]): The data to aggregate.

        Examples:
            >>> encodings = {"x": torch.tensor([[1, 2, 3]]), "y": torch.tensor([[4, 5, 6]])}
            >>> ConcatenateAggregator_().aggregate(encodings)
            tensor([[1, 2, 3], [4, 5, 6]])
        """

        # for encoding in encodings.values():
        #     # B T emb
        #     print(f"encodings: {encoding.shape}")

        encodings_padded = [pad_to_max(t, 4) for t in list(encodings.values())]
        encodings_cat = torch.cat(encodings_padded, dim=-2)

        return encodings_cat

class PassThroughAggregator(Aggregator):
    def __call__(self, encodings: Dict[str, torch.Tensor]):
        return encodings
    

def pad_to_max(tensor, max_dim):
    pad_size = max_dim - tensor.shape[-1]
    return torch.nn.functional.pad(tensor, (0, pad_size), value=0)  # Pad last dimension