import matplotlib.pyplot as plt
import numpy as np
from abc import *
from enum import Enum
from typing import Any, List, Optional, Tuple
from warnings import warn

from SW.DL.util import split_data_set


class DataSet(metaclass=ABCMeta):
    class IO(Enum):
        x = 0
        y = 1

    class Cat(Enum):
        train = 0
        val = 1
        test = 2

    def __init__(self, factors: List, input_size, output_size):
        self.factors: List = factors
        self.input_size: int = input_size
        self.output_size: int = output_size

        self._raw: Any = self._load_raw_data()
        self.data_sets: List[Tuple[np.ndarray, np.ndarray]] = []

    def __get_property(self, io: IO, cat: Cat) -> Optional[np.ndarray]:
        if not self.data_sets:
            raise RuntimeError('You must generate data set first. Use `data.generate(num, ratio)`')

        if len(self.data_sets) > cat.value:
            return self.data_sets[cat.value][io.value]
        else:
            warn('No ' + cat.name + ' data set. Check the `ratio` parameter of the `generate()` function')
            return None

    @property
    def x_train(self) -> Optional[np.ndarray]:
        return self.__get_property(self.IO.x, self.Cat.train)

    @property
    def y_train(self) -> Optional[np.ndarray]:
        return self.__get_property(self.IO.y, self.Cat.train)

    @property
    def x_val(self) -> Optional[np.ndarray]:
        return self.__get_property(self.IO.x, self.Cat.val)

    @property
    def y_val(self) -> Optional[np.ndarray]:
        return self.__get_property(self.IO.y, self.Cat.val)

    @property
    def x_test(self) -> Optional[np.ndarray]:
        return self.__get_property(self.IO.x, self.Cat.test)

    @property
    def y_test(self) -> Optional[np.ndarray]:
        return self.__get_property(self.IO.y, self.Cat.test)

    def generate(self, num: int, ratio: List[float]):
        if not ratio:
            raise AttributeError('Nothing in the `ratio`')

        self.data_sets.clear()
        data_sets = [[[], []] for _ in range(len(ratio) + 1)]

        for factor in self.factors:
            x = []
            y = []
            for i in range(num):
                x.append(self.single_x(factor))
                y.append(self.single_y(factor))

            split_sets = split_data_set(x, y, ratio)
            for data_set, split_set in zip(data_sets, split_sets):
                data_set[self.IO.x.value] += split_set[self.IO.x.value]
                data_set[self.IO.y.value] += split_set[self.IO.y.value]

        for data_set in data_sets:
            self.data_sets.append(
                (self.reshape_x(data_set[self.IO.x.value]), self.reshape_y(data_set[self.IO.y.value]))
            )

        return self

    def plot_x(self, factor):
        plt.plot(self.single_x(factor))
        plt.tight_layout()
        plt.show()

    @abstractmethod
    def _load_raw_data(self) -> Any:
        pass

    @abstractmethod
    def single_x(self, factor):
        pass

    @abstractmethod
    def single_y(self, factor):
        pass

    @abstractmethod
    def reshape_x(self, x) -> np.ndarray:
        pass

    @abstractmethod
    def reshape_y(self, y) -> np.ndarray:
        pass
