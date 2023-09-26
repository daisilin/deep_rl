'''For depth, value quality, etc.'''

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

import create_database_sz as cd
import tournament_new_sz as tn

game = Game(4,9,4)
import value_analysis as va
from tqdm import tqdm

import fcntl

METRIC_RES_DIR = '/scratch/zz737/fiar/metric'
METRIC_DATABASE_LOC = '/scratch/zz737/fiar/metric/all_players_metrics.p'
METRIC_DATABASE_backup_LOC = '/scratch/zz737/fiar/metric/all_players_metrics_backup.p'

METRIC_DB_COLUMNS = ['value_func_quality','depth;median','depth_ratio;median','entropy;median','elo','depth;mean','depth;sem','entropy;mean','entropy;sem','depth_ratio;mean','depth_ratio;sem']

all_p = pd.read_pickle(cd.DATABASE_LOC)

# mask = all_p['tournament']==8
# mask &= all_p['n_res']==3
# mask = (all_p['tournament']==8)|(all_p['tournament']==12) | (all_p['tournament']==13) | (all_p['tournament']==14) | (all_p['tournament']==15) | (all_p['tournament']==16)
mask = (all_p['other_type']=='n_mcts_changed')
# mask &= all_p['n_res']==3

participants = all_p.loc[mask]
print(f'{len(participants)} participants for derived metrics')

def read_agg_res_into_dm_db():
    dm_db_df = pd.read_pickle(METRIC_DATABASE_LOC)
    for id in participants.id:
        tosave_folder = ';'.join(id.split(';')[:-1]) # get rid of iter, keep the rest to make a folder
        tosave_folder_full = os.path.join(METRIC_RES_DIR, tosave_folder)
        fn_agg = f'depth_entropy_value_agg_{id}.p'
        fn_agg = os.path.join(tosave_folder_full,fn_agg)
        try:
            agg_res_df = pd.read_pickle(fn_agg)
            dm_db_df.loc[id].update(agg_res_df.iloc[1:])
        except:
            print(f'no derived metric for {id}')
    dm_db_df.to_pickle(METRIC_DATABASE_LOC)
    print('metric db updated!')
    return dm_db_df

def load_relevant_p_metric():
    dm_db_df = pd.read_pickle(METRIC_DATABASE_LOC)
    dm_db_df = dm_db_df.astype(np.float32)
    all_p = pd.read_pickle(cd.DATABASE_LOC)
    all_p_dm_join = all_p.join(dm_db_df,on='id')
    all_p_dm_join_relevant = all_p_dm_join.loc[all_p_dm_join['depth;median'].notna()]
    all_p_dm_join_relevant = all_p_dm_join_relevant.loc[all_p_dm_join_relevant.value_func_iter!='best']
    
    all_p_dm_join_relevant['value_func_iter'] = all_p_dm_join_relevant['value_func_iter'].astype(np.int64)
    # all_p_dm_join_relevant['elo'] = all_p_dm_join_relevant['elo'].astype(np.int64)
    return all_p_dm_join_relevant


def get_all_p_metric(update_old=False,backup=False):
    all_p = pd.read_pickle(cd.DATABASE_LOC)
    index = all_p.id
    new_p_metrics = pd.DataFrame(index=index,columns=METRIC_DB_COLUMNS)
    if update_old:
        all_p_metrics_old = pd.read_pickle(METRIC_DATABASE_LOC)
        if backup:
            all_p_metrics_old.to_pickle(METRIC_DATABASE_backup_LOC)
        new_p_metrics.update(all_p_metrics_old)
    new_p_metrics.to_pickle(METRIC_DATABASE_LOC)
    return new_p_metrics


def add_sgd_iter(all_p_dm_join_relevant):
    '''
    by default, value_func_iter / mcts_iter is just accepted iter #, does not mean how many iterations of sgd have been performed on the model, for model continuous_training=False
    
    all_p_dm_join_relevant = load_relevant_p_metric()
    no iter='best', otherwise would give error
    
    '''
    all_p_dm_join_relevant['value_func_sgd_iter'] =all_p_dm_join_relevant['value_func_iter']

    gpb=all_p_dm_join_relevant.loc[mask_with_reject].groupby('model_line')

    all_p_dm_join_relevant.loc[mask_with_reject,'value_func_sgd_iter']=gpb['value_func_iter'].rank(ascending=True)
    
    return all_p_dm_join_relevant


def get_heuristic_quality(nnet,opt_boards,opt_values,color=False):
    if color: # if nnet accept color input
        colors = (opt_boards.sum(axis=(1,2))==0).astype(np.int32) # black = 1
        args = (opt_boards, colors)
    else:
        args = [opt_boards]
    nnet_vals = nnet.predict_batch(*args)[1]
    nnet_vals = np.squeeze(nnet_vals)
    q = np.corrcoef(nnet_vals, opt_values)[0,1]
    return q

def get_depth_one_move_one_board_one_model(tree, *args):
    '''
    *args: to capture with/without color input (b,c) or [b]
    '''
    canonicalBoard = args[0]
    starting_npieces = np.sum(canonicalBoard!=0)
    tree.refresh()
    _ = tree.getActionProb(*args)
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

def get_depth_one_move_all_boards_one_model(canonicalBoards_list, tree, selected_inds = None, tot = 50, color=False):
    if color:
        colors = (canonicalBoards_list.sum(axis=(1,2))==0).astype(np.int32) # black = 1
        args_l = list(zip(canonicalBoards_list, colors))
    else:
        args_l = canonicalBoards_list[:,None,:,:] #add dimension to make later code consistent
    Nmoves = canonicalBoards_list.shape[0]
    if selected_inds is None:
        selected_inds = np.random.choice(np.arange(Nmoves),size=tot,replace=False)
    depth_sample_list = []
    for ind in tqdm(selected_inds, desc = "get depth sample board"):
        args = args_l[ind]
        cb = args[0]
        encoded_board = game.encode_pieces(cb)
        npieces = np.sum(cb!=0)
        if (npieces > 5) and (npieces < 25):
            depth,depth_ratio = get_depth_one_move_one_board_one_model(tree,*args)
            depth_sample_list.append({'depth':depth, 'depth_ratio':depth_ratio, 'npieces':npieces,**encoded_board})
    return pd.DataFrame(depth_sample_list)


def get_nnet_based_metrics(nnet,opt_boards, opt_values=None,color=False,to_compute_l=['value_func_quality','entropy']):
    if color: # if nnet accept color input
        colors = (opt_boards.sum(axis=(1,2))==0).astype(np.int32) # black = 1
        args = (opt_boards, colors)
    else:
        args = [opt_boards]
    nnet_pol,nnet_vals = nnet.predict_batch(*args)
    res_dict = {}
    for k in to_compute_l:
        if k =='value_func_quality':
            nnet_vals = np.squeeze(nnet_vals)
            if opt_values is None:
                print('optimal values not given, cannot compute value function quality')
                break
            q = np.corrcoef(nnet_vals, opt_values)[0,1]
            res_dict[k] = q
        elif k=='entropy':
            EPS = 1e-9
            entropy = lambda p:-(p * np.log(p+EPS)).sum(axis=-1)
            npieces_list = (opt_boards!=0).sum(axis=(1,2))
            mask = (npieces_list > 5) & (npieces_list < 25)
            if mask.sum()==0:
                print('not enough boards satisfying the constraints')
                break 
    
            ent_list = entropy(nnet_pol[mask])
            npieces_list = npieces_list[mask]

            cb_encoded_list = []
            for cb in opt_boards[mask]:
                cb_encoded_list.append(game.encode_pieces(cb))
            cb_encoded_list = pd.DataFrame(cb_encoded_list)

            ent_res = pd.DataFrame({'entropy':ent_list,'npieces':npieces_list})
            ent_res = pd.concat([ent_res,cb_encoded_list],axis=1)
            res_dict[k] = ent_res

    return res_dict
    
def get_tree_based_metrics(tree, opt_boards, color=False, to_compute_l=['depth'], **kwargs):
    if color: # if nnet accept color input
        colors = (opt_boards.sum(axis=(1,2))==0).astype(np.int32) # black = 1
        args = (opt_boards, colors)
    else:
        args = [opt_boards]

    res_dict = {}
    for k in to_compute_l:
        if k=='depth':
            depth_res = get_depth_one_move_all_boards_one_model(opt_boards, tree, color=color,**kwargs)
            res_dict[k] = depth_res
    return res_dict

def main(i,test_mode=0):
    # participants = 
    one_info = participants.iloc[i]
    ai,nnet,tree = tn.get_player(game,one_info)
    opt_boards,opt_values = va.load_opt_value_test_boards()
    all_p_metrics = pd.read_pickle(METRIC_DATABASE_LOC)
    
    if pd.notna(all_p_metrics.loc[one_info.id]['depth;median']):
        print(f'probably already did this, skipping: {one_info.id}')
        return

    if test_mode:
        opt_boards = opt_boards[:7]
        opt_values = opt_values[:7]

    nnet_res = get_nnet_based_metrics(nnet,opt_boards, opt_values=opt_values,color=one_info.color,to_compute_l=['value_func_quality','entropy'])
    tree_res = get_tree_based_metrics(tree, opt_boards, color=one_info.color, to_compute_l=['depth'], tot = opt_boards.shape[0])

    
    res_multi_df_l = [nnet_res['entropy'] , tree_res['depth']]
    res_multi_df = res_multi_df_l[0]
    for res in res_multi_df_l[1:]:
        res_multi_df = res_multi_df.join(res.set_index(['bp','wp','npieces']),on=['bp','wp','npieces'],how='inner')

    tosave_folder = ';'.join(one_info.id.split(';')[:-1]) # get rid of iter, keep the rest to make a folder
    tosave_folder_full = os.path.join(METRIC_RES_DIR, tosave_folder)
    if not os.path.exists(tosave_folder_full):
        os.makedirs(tosave_folder_full)
        print(f'{tosave_folder_full} created!')
    
    fn_board = f'depth_entropy_board_{one_info.id}.p'
    fn_board = os.path.join(tosave_folder_full,fn_board)

    # need to update to include depth;mean, se; entropy;mean, sem; for previous agents, manually added
    df_agg = {'id':one_info.id,'value_func_quality':nnet_res['value_func_quality'],
    'depth;median':res_multi_df['depth'].median(),'depth_ratio;median':res_multi_df['depth_ratio'].median(),
    'entropy;median':res_multi_df['entropy'].median(),
    'depth;mean':res_multi_df['depth'].mean(),'depth;sem':res_multi_df['depth'].sem(),
    'depth_ratio;mean':res_multi_df['depth_ratio'].mean(),'depth_ratio;sem':res_multi_df['depth_ratio'].sem(),
    'entropy;mean':res_multi_df['entropy'].mean(),'entropy;sem':res_multi_df['entropy'].sem(),
    }
    df_agg = pd.Series(df_agg)
    fn_agg = f'depth_entropy_value_agg_{one_info.id}.p'
    fn_agg = os.path.join(tosave_folder_full,fn_agg)

    save=True
    if save:
        res_multi_df.to_pickle(fn_board)
        print(f'saved at {fn_board}')
        df_agg.to_pickle(fn_agg)
        print(f'saved at {fn_agg}')

    return res_multi_df,df_agg

if __name__ == '__main__':
    args = sys.argv[1:]
    print(args)
    main(int(args[0]))

