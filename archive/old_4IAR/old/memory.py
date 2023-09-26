from collections import deque
from random import sample

class Memory():
    def __init__(self, initial_mem_size):
        self.memory_size = initial_mem_size
        self.memory_bank = deque(maxlen = self.memory_size)

    def get_memory_usage(self):
        return len(self.memory_bank), self.memory_size

    def clear_memory_bank(self):
        self.memory_bank = deque(maxlen = self.memory_size)

    def change_memory_size(self, new_memory_size):
        self.memory_size = new_memory_size
        new_memory_bank = deque(maxlen = new_memory_size)
        for mem in self.memory_bank:
            new_memory_bank.append(mem)
        self.memory_bank = new_memory_bank
    
    def store_memories(self, memories):
        for mem in memories:
            self.memory_bank.append(mem)

    def sample_memories(self, batch_size):
        return sample(self.memory_bank, min(batch_size, len(self.memory_bank)))

    # @property
    # def size(self):
    #     return len(self.memory_bank)