from enum import Enum


class StreamSequence(str, Enum):
    START = "start"
    CHUNK = "chunk"
    END = "end"