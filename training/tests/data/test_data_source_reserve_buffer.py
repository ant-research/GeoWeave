from collections import deque

import torch

from unirl.data.data_source import MultimodalRLDataSource
from unirl.types.primitives import Image, Images, Texts
from unirl.types.prompts import RolloutInputs


def _inputs(names):
    return RolloutInputs(
        primitives={"text": Texts(texts=list(names))},
        sample_ids=[f"sample-{name}" for name in names],
        group_ids=[f"group-{name}" for name in names],
        metadata=[{"name": name} for name in names],
    )


def test_unused_reserve_rows_are_returned_to_front():
    source = MultimodalRLDataSource.__new__(MultimodalRLDataSource)
    source._sample_buffer = deque()
    loader_batches = deque([_inputs(["a", "b"]), _inputs(["c", "d"])])
    source._next_loader_batch = loader_batches.popleft

    pool = source.get_samples(3)
    assert pool.primitives["text"].texts == ["a", "b", "c"]

    source.return_samples(pool.select(torch.tensor([2])))
    following = source.get_samples(2)
    assert following.primitives["text"].texts == ["c", "d"]


def test_reserve_pool_pads_images_across_loader_batches():
    source = MultimodalRLDataSource.__new__(MultimodalRLDataSource)
    source._sample_buffer = deque()
    first = _inputs(["a", "b"])
    first.primitives["image"] = Images.from_list(
        [Image(torch.ones(3, 608, 544)), Image(torch.ones(3, 576, 608))]
    )
    second = _inputs(["c", "d"])
    second.primitives["image"] = Images.from_list(
        [Image(torch.ones(3, 576, 576)), Image(torch.ones(3, 512, 544))]
    )
    loader_batches = deque([first, second])
    source._next_loader_batch = loader_batches.popleft

    pool = source.get_samples(3)

    images = pool.primitives["image"]
    assert images.pixels.shape == (3, 3, 608, 608)
    assert images.original_sizes == [(608, 544), (576, 608), (576, 576)]
    assert [tuple(image.pixels.shape) for image in images.to_list()] == [
        (3, 608, 544),
        (3, 576, 608),
        (3, 576, 576),
    ]
