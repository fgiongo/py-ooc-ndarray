import numpy as np

class OutOfCoreNDArray:
    def __init__(self,
                 dtype,
                 shape,
                 chunksize=128*1024**2):
        self.dtype = dtype
        self.shape = shape
        self.element_size = np.dtype(dtype).itemsize
        self.chunksize = chunksize
        self.elem_per_chunk = chunksize // element_size
        self.filename = "{}_ooc_ndarray.tmp".format(id(self))
        self.loaded_chunk_index = 0
        self.n_chunks = 1
        self.loaded_chunk = np.mmap(self.filename,
                                    self.dtype, "w+",
                                    0, shape, "C")

    def elem_is_in_chunk(self, key) -> bool:
        min_index = self.loaded_chunk * self.elem_per_chunk
        max_index = min_index + self.elem_per_chunk - 1

        if key < min_index or key > max_index:
            return false

        return true

    def __getitem__(self, key):
        if not self.elem_is_in_chunk(key)
            self.load_chunk(key)

        offset = self.loaded_chunk_index * self.elem_per_chunk
        return self.loaded_chunk[key - offset]
