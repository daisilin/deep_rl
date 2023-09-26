import os,sys
import pandas as pd 
import numpy as np 
sys.path.insert(0,'../classes')
sys.path.insert(0,'../classes/cog_related')

sys.path.insert(0,'./classes')
sys.path.insert(0,'./classes/cog_related')

import cog_value_net as cvn
import anl
import statsmodels.formula.api as smf
from keras import backend as K

from pandarallel import pandarallel


from beck.beck_game import BeckGame as Game

game = Game(4,9,4)

def get_board(row):
    bp = anl.decode_board(row['bp'])
    wp = anl.decode_board(row['wp'])
    board = np.array(list(bp),dtype=int).reshape(4,9) -np.array(list(wp),dtype=int).reshape(4,9)
    return board

def get_self_oppo_feat(row,inv_d):
    '''
    take a series from moves, turn into a series features
    '''
    # inv_d: result of cog_value_net.get_inv_dist_to_center(game)
    board = get_board(row)
    self_board = board if row['player']=='black' else -board
    self_feat,self_header = cvn.get_all_feat(self_board,inv_d)
    oppo_feat,oppo_header = cvn.get_all_feat(-self_board,inv_d)

    self_header = ['self_'+x for x in self_header]
    oppo_header = ['oppo_'+x for x in oppo_header]
    all_feat=pd.Series(list(self_feat)+list(oppo_feat),index=list(self_header)+list(oppo_header))
#     all_feat = np.append(self_feat,oppo_feat)
    return all_feat

def load_moves_get_features(moves_dir_one_instance,iter_csv,epsilon=1e-9):
    '''
    Take moves from one instance, filter and get feature 
    moves_dir_one_instance: eg '/scratch/zz737/fiar/tournaments/tournament_4/moves/raw/checkpoints_mcts100_cpuct2_id-3754964'
    iter_csv: eg 'mcts100_cpuct2;50.csv'
    '''
    one_iter_csv = os.path.join(moves_dir_one_instance, iter_csv)
    one_iter_moves = pd.read_csv(one_iter_csv,sep=' ',header=None)
    one_iter_moves.columns=['bp','wp','player','move','rt','iter','value']

    g = Game(4, 9, 4)
    inv_d = cvn.get_inv_dist_to_center(g)
    pandarallel.initialize()
    # testing, with the iloc
    one_iter_feats = one_iter_moves.parallel_apply(get_self_oppo_feat,axis=1,args=[inv_d])

    # add transformed value to the feature df
    value = one_iter_moves['value']
    value = value * 0.5 + 0.5
    value[value==1] -= epsilon
    value[value==0] += epsilon 
    one_iter_feats['value'] = value
    one_iter_feats['value_logit'] = np.log(value/(1-value))

    # drop 4 feat, never happen
    try:
        one_iter_feats = one_iter_feats.drop(['self_4','oppo_4'],axis=1)
    except:
        Warning('no feature self_4 or oppo_4')


    return one_iter_feats

def regression(one_iter_feats, formula=None,stddz=True, path=None):
    columns = one_iter_feats.columns
    if stddz:
        one_iter_feats = (one_iter_feats - one_iter_feats.mean(axis=0)) / one_iter_feats.std(axis=0)
    if formula is None:
        columns_no_endog = [x for x in columns if 'value' not in x] 
        formula = '+'.join(columns_no_endog)
        formula = 'value_logit~'+formula
        if stddz:
            formula = formula + '-1' # no intercept if standardized
    res = smf.ols(formula,data=one_iter_feats).fit()

    res_df = results_summary_to_dataframe(res)
    if path is not None:
        res_df.to_csv(path)

    return res, res_df


def results_summary_to_dataframe(results):
    '''take the result of an statsmodel results table and transforms it into a dataframe'''
    pvals = results.pvalues
    coeff = results.params
    rsquared = results.rsquared
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]

    results_df = pd.DataFrame({"pvals":pvals,
                               "coeff":coeff,
                               "conf_lower":conf_lower,
                               "conf_higher":conf_higher
                                })
    #Reordering...
    results_df = results_df[["coeff","pvals","conf_lower","conf_higher"]]
    results_df['r2'] = rsquared
    return results_df



def get_iters_test_board_value(test_board,iters,game):
    agent_keys = iters.keys()
    v_l_dict = {}
    for k in agent_keys:
        p_iters = [f'{k};{x}' for x in iters[k]]
        v_l = get_participant_iters_test_board_value(test_board,p_iters,game)
        v_l_dict[k] = v_l
    return v_l_dict
        

def get_participant_iters_test_board_value(test_board,participant_iters,game):
    v_l = []
    for p in participant_iters:
        nmcts, nnet = tournament.get_player(game, participants_dir, p)
        p,v = nnet.predict(test_board)
        v_l.append(v)
        K.clear_session()
    return np.array(v_l)


def load_opt_value_test_boards(filter_3iar=False):
    hqfd = '/home/zz737/projects/fiar/cog_model/fourinarow/Analysis notebooks/new/Heuristic quality'
    move_stats_hvh = np.loadtxt(os.path.join(hqfd,'move_stats_hvh.txt'),dtype=int)
    feature_counts = np.loadtxt(os.path.join(hqfd,'optimal_feature_vals.txt'))[:,-35:]
    optimal_move_values = np.loadtxt(os.path.join(hqfd,'opt_hvh.txt'))[:,-36:]
    optimal_move_values_board = np.loadtxt(os.path.join(hqfd,'opt_hvh.txt'),dtype='object')[:,:2]
    g = Game(4, 9, 4)
    string_to_int_array = lambda x: (np.array(list(x[0]),dtype=int) -np.array(list(x[1]),dtype=int)).reshape(4,9)
    optimal_move_values_board_reshaped_int = np.apply_along_axis(string_to_int_array,1,optimal_move_values_board)

    player_color = move_stats_hvh[:,1]
    optimal_board_values = np.full_like(player_color,fill_value=np.nan,dtype=float)
    optimal_board_values[player_color==0] = np.nanmax(optimal_move_values[player_color==0,:],axis=1)
    optimal_board_values[player_color==1] = -np.nanmin(optimal_move_values[player_color==1,:],axis=1)

    optimal_move_values_board_reshaped_int[player_color==1] = optimal_move_values_board_reshaped_int[player_color==1] * (-1)

    if filter_3iar:
        # filter out boards with 3iar for self
        all_feats = cvn.get_all_feat_self_oppo(optimal_move_values_board_reshaped_int,cvn.get_inv_dist_to_center(g))
        optimal_move_values_board_reshaped_int = optimal_move_values_board_reshaped_int[all_feats[:,3]==0]

    return optimal_move_values_board_reshaped_int, optimal_board_values


#======tree analysis==========
def get_depth_one_move_one_board_one_model(canonicalBoard, tree):
    starting_npieces = np.sum(canonicalBoard!=0)
    tree.refresh()
    _ = tree.getActionProb(canonicalBoard)
    Npieces = {}
    for k in tree.Es.keys():
        try:
            k_board = np.frombuffer(k,dtype=int)
        except: # for bfts, the keys are readable string representation and we cannot use frombuffer to decode
            k_board = game.str_rep_to_array(k)
        Npieces[k] = np.sum(k_board!=0)
    max_npieces = np.max(list(Npieces.values()))
    depth = max_npieces - starting_npieces

    max_possible_depth = canonicalBoard.shape[0] * canonicalBoard.shape[1] - starting_npieces
    if max_possible_depth ==0:
        depth_percent = np.nan
    else:
        depth_percent = depth / max_possible_depth
    return depth, depth_percent

folder = '/scratch/zz737/fiar/tournaments/tournament_4/checkpoints_mcts100_cpuct2_id-3754964/'
def get_iter_list_in_folder(folder):
    '''
    folder contains the checkpoint iterations
    '''
    file_lists = os.listdir(folder)
    iters = list(set([int(x.split('_')[1].split('.')[0]) for x in file_lists if x.startswith('checkpoint_')]))
#     iters = [f'checkpoint_{x}.pth.tar' for x in iters]
    return iters

from tqdm import tqdm
def get_depth_one_move_one_board_all_models(canonicalBoard, folder):
    iters = get_iter_list_in_folder(folder)
    depth_dict = {}
    for it in tqdm(iters, desc="get depth per model"):
        try:
            _,nnet,tree=tournament.get_player(g, participants_dir_one, f'mcts100_cpuct2;{it}',is_return_mcts=True)
            depth_dict[it] = get_depth_one_move_one_board_one_model(canonicalBoard, tree)
        except:
            pass
        K.clear_session()
    return depth_dict


# this version use already decoded boards
def get_depth_one_move_all_boards_one_model(canonicalBoards_list, tree, selected_inds = None, tot = 50):
    
    Nmoves = canonicalBoards_list.shape[0]
    if selected_inds is None:
        selected_inds = np.random.choice(np.arange(Nmoves),size=tot,replace=False)
    depth_sample_list = []
    for ind in tqdm(selected_inds, desc = "get depth sample board"):
        cb = canonicalBoards_list[ind]
        encoded_board = game.encode_pieces(cb)
        npieces = np.sum(cb!=0)
        if (npieces > 5) and (npieces < 25):
            depth,depth_ratio = get_depth_one_move_one_board_one_model(cb, tree)
            depth_sample_list.append({'depth':depth, 'depth_ratio':depth_ratio, 'npieces':npieces,**encoded_board})
    return pd.DataFrame(depth_sample_list)



# moves: encoded board info and move info as in saved moves
# def get_depth_one_move_all_boards_one_model(moves, tree, selected_inds = None, tot = 50):
    
#     Nmoves = len(moves)
#     if selected_inds is None:
#         selected_inds = np.random.choice(np.arange(Nmoves),size=tot,replace=False)
#     depth_sample_list = []
#     for ind in tqdm(selected_inds, desc = "get depth sample board"):
#         b = va.get_board(moves.iloc[ind])
#         cb = b
#         npieces = np.sum(b!=0)
#         if (npieces > 5) and (npieces < 25):
#             if np.sum(b==1) !=np.sum(b==-1):
#                 cb = -b
#             depth,depth_ratio = get_depth_one_move_one_board_one_model(cb, tree)
#             depth_sample_list.append({'depth':depth, 'depth_ratio':depth_ratio, 'npieces':npieces})
#     return pd.DataFrame(depth_sample_list)

def get_depth_one_move_all_boards_all_models(moves, folder, tot = 100, iters=None):
    Nmoves = len(moves)
    selected_inds = np.random.choice(np.arange(Nmoves),size=tot,replace=False)
    depth_all_models = {}
    
    if iters is None:
        iters = get_iter_list_in_folder(folder)
    
    for it in iters:
        _,nnet,tree=tournament.get_player(g, folder, f'mcts100_cpuct2;{it}',is_return_mcts=True)
        depth_one_model = get_depth_one_move_all_boards_one_model(moves, tree, selected_inds = selected_inds, tot = tot)
        depth_all_models[it] = depth_one_model
        K.clear_session()
    
    return depth_all_models

# def get_depth_one_move_all_boards_all_models(moves, folder, tot = 100):
#     Nmoves = len(moves)
#     selected_inds = np.random.choice(np.arange(Nmoves),size=tot,replace=False)
#     depth_dict_list = []
#     for ind in tqdm(selected_inds, desc="get depth sample boards"):
#         b = va.get_board(moves.iloc[ind])
#         npieces = np.sum(b!=0)
#         if (npieces > 5) and (npieces < 30):
#             if np.sum(b==1) != np.sum(b==-1):
#                 b=-b
#             depth_dict = get_depth_one_move_one_board_all_models(b, folder)

#             depth_dict_list.append((npieces,depth_dict))
#     return depth_dict_list

# entropy analysis
EPS = 1e-9
entropy = lambda p:-(p * np.log(p+EPS)).sum(axis=-1) # across the feature axis, leave the batch axis
def get_entropy_one_model_all_boards(canonicalBoards_list,nnet):
    npieces_list = (canonicalBoards_list!=0).sum(axis=(1,2))
    mask = (npieces_list > 5) & (npieces_list < 25)
    if mask.sum()==0:
        print('not enough boards satisfying the constraints')
        return 
    p_list,_ = nnet.predict_batch(canonicalBoards_list[mask])
    ent_list = entropy(p_list)
    npieces_list = npieces_list[mask]

    cb_encoded_list = []
    for cb in canonicalBoards_list[mask]:
        cb_encoded_list.append(game.encode_pieces(cb))
    cb_encoded_list = pd.DataFrame(cb_encoded_list)

    ent_res = pd.DataFrame({'entropy':ent_list,'npieces':npieces_list})
    ent_res = pd.concat([ent_res,cb_encoded_list],axis=1)
    return ent_res

def get_Ns_weighted_mean_entropy_one_move_all_boards_one_model(canonicalBoards_list, tree, selected_inds = None, tot = 50):
    '''
    the framework here is copied from get_depth_one_move_all_boards_one_model
    perhaps merge them in the future?
    '''
    Nmoves = canonicalBoards_list.shape[0]
    if selected_inds is None:
        selected_inds = np.random.choice(np.arange(Nmoves),size=tot,replace=False)
    depth_sample_list = []
    for ind in tqdm(selected_inds, desc = "get depth sample board"):
        cb = canonicalBoards_list[ind]
        encoded_board = game.encode_pieces(cb)
        npieces = np.sum(cb!=0)
        if (npieces > 5) and (npieces < 25):
            mean_ent = get_Ns_weighted_mean_entropy_one_move_one_board_one_model(cb, tree)
            depth_sample_list.append({'Ns_weighted_mean_entropy':mean_ent, 'npieces':npieces,**encoded_board})
    return pd.DataFrame(depth_sample_list)

def get_Ns_weighted_mean_entropy_one_move_one_board_one_model(canonicalBoard, tree):
    '''
    mean entropy for all states during a tree getActionProb (100 searchs), weighted by visit counts of those states
    note that some Ns = 0, the corresponding Ps will not contribute to the entropy, which makes sense 
    '''
    tree.refresh()
    tree.getActionProb(canonicalBoard)

    ns_ps=pd.concat([pd.Series(tree.Ns),pd.Series(tree.Ps)],keys=['Ns','Ps'],axis=1)
    ns_ps['entropy']=entropy(np.stack(ns_ps['Ps'].values,axis=0))
    mean_ent = (ns_ps['Ns'] * ns_ps['entropy']).sum()/ns_ps['Ns'].sum()
    return mean_ent



# value policy correlation
def policy_val_correlation_one_board_one_model(canonicalBoard, game, nnet, flip_val=False):
    valids = game.getValidMoves(canonicalBoard, 1)
    a_batch = np.nonzero(valids)[0] # the indices of valid actions
    x_l = a_batch // game.n
    y_l = a_batch % game.n
    n_valids = len(x_l)
    ind_l = np.arange(n_valids)
    board_batch = np.tile(canonicalBoard,(n_valids,1,1)) # vectorized way of getting a batch for evaluation; canonicalBoard, from self's perspective
    if flip_val:
        board_batch = -board_batch
        board_batch[ind_l, x_l, y_l] = -1
        _,val_batch=nnet.predict_batch(board_batch)
        val_batch = -val_batch
    else:
        board_batch[ind_l, x_l, y_l] = 1
        _,val_batch=nnet.predict_batch(board_batch)
        
    policy,_ = nnet.predict(canonicalBoard)
    policy = policy[valids.astype(bool)]
    
    corr = np.corrcoef(val_batch.squeeze(),policy)[0,1]
    return corr

from tqdm import tqdm
def policy_val_correlation_all_board_one_model(canonicalBoard_batch,game, nnet, flip_val=False):
    corr_l=[]
    npieces_list = (canonicalBoard_batch!=0).sum(axis=(1,2))
    mask = (npieces_list > 5) & (npieces_list < 25)
    if mask.sum()==0:
        print('not enough boards satisfying the constraints')
        return 
    cb_encoded_list = []
    for cb in tqdm(canonicalBoard_batch[mask], desc="policy value corr"):
        corr = policy_val_correlation_one_board_one_model(cb,game,nnet,flip_val=flip_val)
        corr_l.append(corr)
        cb_encoded_list.append(game.encode_pieces(cb))

    cb_encoded_list = pd.DataFrame(cb_encoded_list)
    corr_l = np.array(corr_l)
    npieces_list = npieces_list[mask]
    pvcorr_res = pd.DataFrame({'pvcorr':corr_l,'npieces':npieces_list})
    pvcorr_res = pd.concat([pvcorr_res, cb_encoded_list],axis=1)
    return pvcorr_res
