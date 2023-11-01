from webdataset import utils, Composable, Shorthands
from torch.utils.data  import  DataLoader, IterableDataset
from tarfile import ReadError
import PIL


def WebLoader(*args, **kw):
    """Return a small wrapper around torch.utils.data.DataLoader.
    This wrapper works identically to the original `DataLoader`, but adds
    alls the convenience functions and filters for WebDataset.
    You can use all the methods from `Composable` (`then`, `compose`) and
    from `Shorthands` (`batched`, `unbatched`, `decode`, `shuffle`, etc.)
    on the result.
    :param args: forwarded to `DataLoader`
    :param kw: forwarded to `DataLoader`
    """
    return Processor(DataLoader(*args, **kw), utils.identity)


class Processor(IterableDataset, Composable, Shorthands):
    """A class that turns a function into an IterableDataset."""

    def __init__(self, source, f, *args, _kwa={}, **kw):
        """Create a processor.
        The function should take an iterator as an argument and yield
        processed samples. The function is invoked as `f(source, *args, **kw)`.
        :param source: source dataset, an IterableDataset
        :param f: function implementing the processor
        :param args: extra arguments to the processor after the source iterator
        :param _kwa: keyword arguments
        :param kw: extra keyword arguments
        """
        super().__init__()
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = dict(_kwa)
        self.kw.update(kw)

        """Return an iterator over the source dataset processed by the given function."""
        assert self.source is not None, f"must set source before calling iter {self.f} {self.args} {self.kw}"
        assert callable(self.f), self.f
        self._iter = self.f(iter(self.source), *self.args, **self.kw)

    def source_(self, source):
        """Set the source dataset.
        :param source: source dataset
        """
        self.source = source
        return self

    def __iter__(self):
        return self._iter

    def next(self):
        try:
            return self._iter.next()
        except StopIteration as e:
            raise e
        except (PIL.UnidentifiedImageError, OSError, ReadError, IndexError) as e:
            print(e)
            return self.next()