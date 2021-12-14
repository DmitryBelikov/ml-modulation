from abc import ABCMeta, abstractmethod


class Modulator(metaclass=ABCMeta):
    @abstractmethod
    def encode(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def decode(self) -> None:
        raise NotImplementedError
