
import numpy as np


def get_children(board_arr,tree,verbose=True):
    game = g  = tree.game
    children = []
    for (bstr,action),val in tree.Qsa.items():
        b_array = g.str_rep_to_array(bstr)
        if len(b_array.shape) == 1:
            b_array= b_array.reshape(game.m,game.n)
        # import pdb
        # pdb.set_trace()
        if (b_array==board_arr).astype(np.float32).sum()==(board_arr.shape[0]*board_arr.shape[1]):
            new_b, new_p = game.getNextState(b_array,1,action)
            children.append((new_b,action,val))
        #     print(b_array)
            if verbose:
                print('new board')
                game.display(new_b)

    #             print(new_b)
                print(f'action {action}')
                print(f'value {val}\n')
            
    return children


import copy
def get_parent(board_arr,action):
    x,y = action //9,action%9
    parent = copy.copy(board_arr)
    parent[x,y] = 0
    return -parent

def get_board_size(tree):
    Ssa = {}
    game = g  = tree.game
    for (bstr,action),val in tree.Qsa.items():
        b_array = g.str_rep_to_array(bstr)
        size=np.sum(b_array!=0)
        Ssa[(bstr,action)] = size
    return Ssa

def get_largest_board(tree,offset=0):
    game = g  = tree.game
    Ssa = get_board_size(tree)
    maxsize=np.max(list(Ssa.values())) # list here important
    board_q_l = []
    for (bstr,action), size in Ssa.items():
        if size==maxsize - offset:
            val = tree.Qsa[(bstr,action)]
            b_array = g.str_rep_to_array(bstr)
            new_b, new_p = game.getNextState(b_array,1,action)
            board_q_l.append((new_b,val))
    return board_q_l

def traverse_tree_principal_variation(board_arr,tree):
    children = get_children(board_arr,tree,verbose=False)
    board_sequence = []
    while len(children) > 0:
        bestval = np.max([val for _,_,val in children])
        best_board_a_val = [(b,a,val) for b,a,val in children if val==bestval][0]
        board_sequence.append(best_board_a_val)
        children = get_children(-best_board_a_val[0],tree,verbose=False)
    return board_sequence