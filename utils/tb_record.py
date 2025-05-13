"""This script provides a TensorboardRecord class to read tensorboard records.
And a command line tool to convert tensorboard scalar records to images.

Modified from https://github.com/lanpa/tensorboard-dumper/blob/master/dump.py
"""

from __future__ import annotations

import math
import struct
from io import BytesIO
from pathlib import Path
from typing import Generator

import numpy as np
from PIL import Image
from tensorboard.compat.proto import event_pb2
from tqdm import tqdm

TAG_T = str
STEP_T = int


class TBImage:
    """Fake stub for type checking."""

    encoded_image_string: bytes


class Value:
    """Fake stub for type checking."""

    image: TBImage
    simple_value: float
    tag: str

    def HasField(self, field: str) -> bool:
        ...


class Summary:
    """Fake stub for type checking."""

    value: list[Value]


class Event:
    """Fake stub for type checking."""

    step: int
    summary: Summary

    def HasField(self, field: str) -> bool:
        ...

    def ParseFromString(self, data: bytes):
        ...


class TensorboardRecord:
    """Tensorboard record parser.

    Attributes:
        scalars: dict of scalars. {tag: {step: value}}
        images: dict of images. {tag: {step: image}}

    Class Methods:
        from_file: read tensorboard record from file.
        from_dir: read tensorboard record from directory recursively.
            records in subdirectories will be added with relative path.

    Examples:
        # create a summary writer and write some data
        >>> import torch
        >>> from torch.utils.tensorboard import SummaryWriter
        >>> writer = SummaryWriter(log_dir="runs/example")
        >>> for n_iter in range(100):
        ...     writer.add_scalars("loss", {"train": n_iter, "test": n_iter + 1}, n_iter)
        ...     writer.add_scalar("accuracy", n_iter / 100, n_iter)
        ...     writer.add_image("image", torch.rand(3, 64, 64), n_iter)
        ...     writer.add_images("images", torch.rand(4, 3, 16, 16), n_iter)
        >>> writer.close()

        # read tensorboard record
        >>> record = TensorboardRecord.from_dir('runs/example')
        >>> record.scalar_tags
        ['accuracy', 'loss_test/loss', 'loss_train/loss']
        >>> record.image_tags
        ['image', 'images']
    """

    def __init__(self) -> None:
        self.scalars: dict[TAG_T, dict[STEP_T, float]] = {}
        self.images: dict[TAG_T, dict[STEP_T, Image.Image]] = {}

    def relative_to(self, dir: Path) -> TensorboardRecord:
        """Add relative path to tag."""
        if dir == Path('.'):
            return self
        posix = dir.as_posix()
        record = TensorboardRecord()
        for tag, steps in self.scalars.items():
            record.scalars[f'{posix}/{tag}'] = steps
        for tag, steps in self.images.items():
            record.images[f'{posix}/{tag}'] = steps
        return record

    @classmethod
    def read_events(cls, data: bytes) -> Generator[bytes, None, None]:
        offset = 0
        while offset < len(data):
            header = struct.unpack_from('Q', data, offset)
            event_str = data[offset + 12:offset + 12 + int(header[0])]
            offset += 12 + int(header[0]) + 4
            yield event_str

    @classmethod
    def parse(
        cls,
        data: bytes,
        tqdm: tqdm | None = None,
    ) -> Generator[tuple[TAG_T, STEP_T, float | Image.Image], None, None]:
        for event_str in cls.read_events(data):
            event: Event = event_pb2.Event()  # type: ignore
            event.ParseFromString(event_str)
            if tqdm is not None:
                tqdm.update(len(event_str))
            if event.HasField('summary'):
                for value in event.summary.value:
                    if value.HasField('simple_value'):
                        yield value.tag, event.step, value.simple_value
                    if value.HasField('image'):
                        image = value.image
                        image_pil = Image.open(
                            BytesIO(image.encoded_image_string))
                        yield value.tag, event.step, image_pil
        if tqdm is not None:
            tqdm.update(tqdm.total - tqdm.n)

    @classmethod
    def from_file(cls, path: str | Path) -> TensorboardRecord:
        path = Path(path)
        record = cls()
        with path.open('rb') as f:
            data = f.read()
        pbar = tqdm(total=len(data), desc=path.name, unit='B', unit_scale=True)
        for tag, step, value in cls.parse(data, tqdm=pbar):
            if isinstance(value, Image.Image):
                record.images.setdefault(tag, {})[step] = value
            else:
                record.scalars.setdefault(tag, {})[step] = value
        pbar.close()
        return record

    @classmethod
    def from_dir(cls, dir: str | Path) -> TensorboardRecord:
        dir = Path(dir)
        record = cls()
        for path in dir.rglob('events.out.tfevents.*'):
            sub_record = cls.from_file(path)
            sub_dir = path.parent.relative_to(dir)
            sub_record = sub_record.relative_to(sub_dir)
            # detect tag conflict
            for tag in sub_record.scalar_tags:
                if tag in record.scalar_tags:
                    raise ValueError(f'Tag conflict: {tag} in {path}')
            for tag in sub_record.image_tags:
                if tag in record.image_tags:
                    raise ValueError(f'Tag conflict: {tag} in {path}')
            record.scalars.update(sub_record.scalars)
            record.images.update(sub_record.images)
        return record

    @property
    def scalar_tags(self) -> list[TAG_T]:
        return list(self.scalars.keys())

    @property
    def image_tags(self) -> list[TAG_T]:
        return list(self.images.keys())

    def scalar(self, tag: TAG_T) -> dict[STEP_T, float]:
        return self.scalars[tag]

    def image(self, tag: TAG_T) -> dict[STEP_T, Image.Image]:
        return self.images[tag]

    def group_scalars(self, *tag: TAG_T) -> dict[STEP_T, dict[TAG_T, float]]:
        """Group scalars with same step."""
        if not tag:
            return {}

        scalars = [self.scalar(t) for t in tag]
        # make sure all scalars have same steps
        steps = set(scalars[0].keys())
        for scalar in scalars[1:]:
            if set(scalar.keys()) != steps:
                raise ValueError('Scalas have different steps.')

        grouped = {}
        for step in steps:
            grouped[step] = {
                tag: scalar[step]
                for tag, scalar in zip(tag, scalars)
            }
        return grouped

    def group_images(self,
                     *tag: TAG_T) -> dict[STEP_T, dict[TAG_T, Image.Image]]:
        """Group images with same step."""
        if not tag:
            return {}

        images = [self.image(t) for t in tag]
        # make sure all images have same steps
        steps = set(images[0].keys())
        for image in images[1:]:
            if set(image.keys()) != steps:
                raise ValueError('Images have different steps.')

        grouped = {}
        for step in steps:
            grouped[step] = {
                tag: image[step]
                for tag, image in zip(tag, images)
            }
        return grouped


def show_basic_info(record: TensorboardRecord):
    for tag in record.scalar_tags:
        print(tag, len(record.scalar(tag)))
    for tag in record.image_tags:
        print(tag, len(record.image(tag)))


def smoothing(
    x: list[float],
    weight: float,
    ignore_outlier: bool = True,
) -> list[float]:
    if ignore_outlier:
        arr = np.array(x)
        mean = np.mean(arr)
        std = np.std(arr)
        lo, hi = mean - 3 * std, mean + 3 * std
        arr = np.clip(arr, lo, hi)
        x = arr.tolist()

    last = 0
    smoothed = []
    num_acc = 0
    for next_val in x:
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - math.pow(weight, num_acc)
        smoothed_val = last / debias_weight
        smoothed.append(smoothed_val)

    return smoothed


def plot_scalar(
    record: TensorboardRecord,
    tags: list[TAG_T],
    smooth: float = 0.0,
):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for tag in tags:
        if tag not in record.scalar_tags:
            print(f'No scalar tag {tag}')
        steps = record.scalar(tag)
        tag_name = tag.replace('/', '_').replace('-', '_')
        keys = list(steps.keys())
        values = list(steps.values())
        values = smoothing(values, smooth)
        ax.plot(keys, values, label=tag_name)
    ax.legend()
    return fig


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('path', type=Path)
    parser.add_argument('--scalar', nargs='*', help='scalar to plot')
    parser.add_argument('--smooth', type=float, default=0.0)
    args = parser.parse_args()

    if args.path.is_dir():
        record = TensorboardRecord.from_dir(args.path)
    else:
        record = TensorboardRecord.from_file(args.path)

    # fig_dir = args.path.parent / f'{args.path.stem}_figs'
    # fig_dir.mkdir(exist_ok=True)

    show_basic_info(record)
    # scalars = args.scalar or record.scalar_tags
    # if scalars:
    #     for tag in scalars:
    #         fig = plot_scalar(record, [tag], args.smooth)
    #         tagname = tag.replace('/', '_').replace('-', '_')
    #         fig.savefig(str(fig_dir / f'{tagname}.png'))


if __name__ == "__main__":
    main()
