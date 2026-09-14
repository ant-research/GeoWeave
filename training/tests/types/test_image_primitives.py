import torch

from unirl.types.primitives import Image, Images


def test_images_round_trip_restores_sizes_before_batch_padding():
    first = Image(torch.ones(3, 5, 7))
    second = Image(torch.ones(3, 9, 4))

    batch = Images.from_list([first, second])
    restored = batch.to_list()

    assert batch.pixels.shape == (2, 3, 9, 7)
    assert batch.sizes == [(5, 7), (9, 4)]
    assert [tuple(image.pixels.shape) for image in restored] == [(3, 5, 7), (3, 9, 4)]
