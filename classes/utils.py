import numpy as np
import os

class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class dotdict(dict):
    # def __getattr__(self, name):
    #     return self[name]
    def __getattr__(self, attr):
        if attr.startswith('__'):
            raise AttributeError
        return self.get(attr, None)

def save_moves(moves_result_multigame,tosave='both',subjectID='default_ai',model_class=None, model_instance=None, temp=None,
                fd = '../models/moves/',
    ):
    """
    [SZ] for saving agent's moves in csv:
    rows: round
    cols: code for all black pieces; code for all white; color of the player; code for current move; rt; subject id;
    all but rt and subjectID will already be present. Color need to be mapped from 0,1 to 'black', 'white'; rt will be 0; subjectID will be the name of the model class + instance + temp
    """
    if not os.path.exists(fd):
        os.makedirs(fd)

    if tosave =='both':
        moves_result_all = np.vstack(moves_result_multigame)


    elif isinstance(tosave,int):
        moves_result_all = moves_result_multigame[tosave]

    else:
        print('tosave undefined')
        return

    # map number to color string
    moves_result_all = moves_result_all.astype('object') # need to recast it into object for string assignment to work
    color_list = np.array(['black','white'])
    moves_result_all[:,2] = color_list[moves_result_all[:,2].astype(np.int16)]

    nrow = moves_result_all.shape[0]
    rt_col = np.zeros((nrow,1))
    subjectID_col = np.zeros((nrow,1),dtype=object)
    if subjectID is None:
        subjectID = model_class+ '_' + model_instance + '_temp' + str(temp).replace('.','dot')

    subjectID_col[:] = subjectID

    value_col = moves_result_all[:,[4]] # value column index is 4 in moves_result_all, because no subjectID there yet

    moves_result_all = np.hstack([moves_result_all[:,:4], rt_col, subjectID_col, value_col])

    if model_class is not None:
        # fd += model_class + '/'
        fd = os.path.join(fd, model_class)

    if not os.path.exists(fd):
        os.makedirs(fd)

    fn = os.path.join(fd, subjectID+'.csv')

    with open(fn,'a') as f:
        # f.write("\n") # new line is not needed if want continous rows 
        np.savetxt(f,moves_result_all,delimiter=",",fmt='%d %d %s %d %d %s %f') #black position, white position, color, move, rt, id, value
    
    print(f'moves for {subjectID} saved at {fn}')

def select_n_instances_each_from_iters(iters,n):
    '''
    [SZ] uneven sampling; first 1/3 of the iters sample 2/3 of the n, vice versa
    Because rapid change in model in early iterations. 
    skip 0th index, since it's probably easy to beat. 
    '''
    first_third_n = int(2/3 * n)
    rest_n = n - first_third_n
    iters_subsamp = {}
    for k,v in iters.items():
        N = len(v)
        first_third_N = int(N)*1/3
        first_third_inds = np.linspace(1,first_third_N-1,first_third_n).astype(int)
        rest_inds = np.linspace(first_third_N-1,N-1,rest_n + 1).astype(int)[1:] # drop the first, since duplicate from the above
        tot_inds = np.concatenate([first_third_inds,rest_inds])
        iters_subsamp[k] = np.array(v)[tot_inds]

    return iters_subsamp


