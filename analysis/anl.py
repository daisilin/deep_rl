import numpy as np



def decode_board(x):
    return "{0:036b}".format(x)

def decode_move(x):
    return 36-int(x).bit_length()

