from abc import ABCMeta, abstractmethod


class Modulator(metaclass=ABCMeta):
    @abstractmethod
    def encode(self, raw):
        raise NotImplementedError

    @abstractmethod
    def decode(self, encoded):
        raise NotImplementedError
