import sys,os
import pandas as pd
import numpy as np
import copy


COLUMNS=['tree_type','value_func_type','other_type','n_mcts','cpuct','value_func_iter','mcts_iter','n_bfts','pruning_thresh','center','2_con','2_uncon','3','4','oppo_scale','id','tournament','mcts_location','value_func_location','n_res','color','continuous_training','tempThreshold','dir_alpha','model_line']



root_dir = '/scratch/xl1005/deep-master/tournaments'
DATABASE_LOC = '/scratch/xl1005/deep-master/tournaments/all_players.pkl'
DATABASE_LOC_backup = '/scratch/xl1005/deep-master/tournaments/all_players_backup.pkl'

TOURNAMENT_RES_LOC = '/scratch/xl1005/deep-master/tournaments/ai_all_player_round_robin_base.pkl'
# TOURNAMENT_RES_LOC = tn.TOURNAMENT_BASE_LOC
TOURNAMENT_RES_LOC_backup = '//scratch/xl1005/deep-master/tournaments/ai_all_player_round_robin_base_backup.pkl'
# TOURNAMENT_RES_LOC_backup = tn.TOURNAMENT_BASE_backup_LOC

REPEATED_TOURNAMENT_RES_LOC = '/scratch/xl1005/deep-master/tournaments/ai_all_player_repeated_round_robin_base.pkl'
REPEATED_TOURNAMENT_RES_LOC_backup = '/scratch/xl1005/deep-master/tournaments/ai_all_player_repeated_round_robin_base_backup.pkl'

import tournament_new_sz as tn

select_row_by_id = lambda id,df:df.loc[df['id']==id].iloc[0]

def expand_old_database_with_new_cols(to_save=False,backup=False,**kwargs):
    old_db = pd.read_pickle(DATABASE_LOC)
    if backup:
        old_db.to_pickle(DATABASE_LOC_backup)
        print(f'Database backed up at {DATABASE_LOC_backup}')
    for k,val in kwargs.items():
        if k not in old_db.columns:
            old_db[k] = val 
        else:
            print(f'{k} already in')
    if to_save:
        old_db.to_pickle(DATABASE_LOC)
    return old_db

def merge_nn_with_bfts(all_players,bfts_id_partial,mcts_id):
    '''
    all_players: database containing info for all players
    bfts_id_partial: e.g. bfts100;prune2, not the full id of a cog+bfts player, but give info about the bfts parameters
    mcts_id: e.g. full id of a nn+mcts player
    get the info for a cog + mcts player
    '''
    bfts_params_keys = ['n_bfts','pruning_thresh']
    one_player = all_players.loc[all_players['id'] == mcts_id].iloc[0]
    one_player = copy.copy(one_player)
    one_player.loc['tree_type'] = 'bfts'
    one_player.loc['value_func_type'] = 'nn'
    one_player.loc['n_mcts'] = np.nan
    one_player.loc['cpuct'] = np.nan
    one_player.loc['mcts_iter'] = np.nan
    one_player.loc['mcts_location']=np.nan
    one_player.loc['mcts_location']=np.nan
    mask = [bfts_id_partial in i for i in all_players['id']]
    one_player.loc[bfts_params_keys] = all_players.loc[mask,bfts_params_keys].iloc[0]
    one_player.loc['id'] = mcts_id + ';' + bfts_id_partial
    return one_player

def merge_cog_with_mcts(all_players,cog_id_partial,mcts_id):
    '''
    all_players: database containing info for all players
    cog_id_partial: e.g. cog_id_1, not the full id of a cog+bfts player, but give info about the value function parameters
    mcts_id: e.g. full id of a nn+mcts player
    get the info for a cog + mcts player
    '''
    cog_val_params_keys = ['center','2_con','2_uncon','3','4','oppo_scale']
    one_player = all_players.loc[all_players['id'] == mcts_id].iloc[0]
    one_player = copy.copy(one_player)
    one_player.loc['tree_type'] = 'mcts'
    one_player.loc['value_func_type'] = 'cog'
    one_player.loc['value_func_location']=np.nan
    one_player.loc['value_func_iter']=np.nan
    mask = [cog_id_partial in i for i in all_players['id']]
    one_player.loc[cog_val_params_keys] = all_players.loc[mask,cog_val_params_keys].iloc[0]
    one_player.loc['id'] = cog_id_partial + ';' + mcts_id

    return one_player

def extend_change_nmcts(query='model_line=="tournament_16;mcts100;cpuct2;id-res3-0" and other_type.isna()',
    n_mcts_l=[2,20,50,400],backup=True):
    '''
    change the n_mcts of nn+mcts agents trained with certain n_mcts, thereby creating new agents
    query: for df_curr.query(query), select part of the old df on the basis of which to create new ones
    return None, new_df to match the format in extend_database_and_base_tournament_result  
    '''

    df_curr = pd.read_pickle(DATABASE_LOC)
    old_N = df_curr.shape[0]
    if backup:
        df_curr.to_pickle(DATABASE_LOC_backup)
        print(f'Database backed up at {DATABASE_LOC_backup}') 

    base_df = df_curr.query(query)
    extended_df_l = []
    id_change=lambda x,n:f'mcts{n};'+x
    for n_mcts in n_mcts_l:
        extended_df = copy.copy(base_df)
        extended_df['other_type']='n_mcts_changed'
        extended_df['n_mcts'] = n_mcts
        extended_df['id'] = extended_df['id'].apply(id_change,args=[n_mcts])
        extended_df_l.append(extended_df)
    new_df = pd.concat(extended_df_l,axis=0,ignore_index=True)

    df_curr = pd.concat([df_curr,new_df],axis=0,ignore_index=True)
    df_curr = df_curr.loc[~df_curr['id'].duplicated()]
    new_N = df_curr.shape[0]
    print(f'{new_N - old_N} new players added')
    df_curr.to_pickle(DATABASE_LOC)
    print(f'new database saved at {DATABASE_LOC}')
    return df_curr,new_df

    


def create_database_trained_nn_mcts(root_dir=root_dir,tournament_dir='tournament_5',better_than_last=True, **kwargs):
    '''
    function for creating a dataframe with players of nn+mcts type, from 
    root_dir: eg. /scratch/zz737/fiar/tournaments #indicating location common to all different model lines
    tournament_dir:  eg. tournament_1, where groups of models live
    better_than_last: flag whether the models were trained such that iters that do not win 60% compared to the current best are not saved. Should be the default
    kwargs: additional conditions/hyperparameters to be assigned to the whole tournament_dir (manually deal with outliers)
    '''
    # tournaments_dir='tournament_5'

    

    all_players = pd.DataFrame([],columns=COLUMNS)

    d = tournament_dir
    # for d in tournaments_dirs:
    tour_dir = os.path.join(root_dir,d)
    tournament = int(d.split('_')[1])
    subdirs = os.listdir(tour_dir)
    for subdir in subdirs:
        if subdir.startswith('checkpoints_'):
            subdir_split = subdir.split('_')
            mctsN = subdir_split[1]
            cpuctX = subdir_split[2]

            mctsN = int(mctsN[4:])
            cpuctX = float(cpuctX[5:]) #careful about consequence (int -> float) for loading?

            # if len(subdir_split) > 3: # then it should have id-xxx or id_x, should make it consistent
            if len(subdir_split)==5: # which means the folder name contains id_x; but we want id-x 
                id = ';'.join([d,*subdir_split[1:-1]])
                id += f'-{subdir_split[-1]}'
            else:
                id = ';'.join([d,*subdir_split[1:]]) # combine tournament_x and mctsN_cpuctX_idOOO, to create an identifier for the trained nn+mcts
                
            
            additional_info = kwargs
            if 'res' in subdir_split[-1]:
                n_res = int(subdir_split[-1][3:].split('-')[0])
                additional_info['n_res'] = n_res
                
            
            location = os.path.join(tour_dir,subdir) # e.g. /scratch/zz737/fiar/tournaments/tournament_1/checkpoints_mcts100_cpuct1
            files = os.listdir(location)
            for f in files:
                if f.endswith('index') and (not f.startswith('temp')) and (not f.startswith('init')): # only the succeeded models are saved, with .index, in tournament1 and 5, not in 4
                    if f.startswith('best'):
                        value_func_iter = mcts_iter = 'best' # = best
                        curr_id = ';'.join([id,value_func_iter]) # add iter info to id
                    elif f.startswith('checkpoint_'):
                        iter =f.split('.')[0].split('_')[1]
                        value_func_iter = mcts_iter = int(iter) # = number
                        curr_id = ';'.join([id,iter]) # add iter info to id

                    

                    fn_without_index = '.'.join(f.split('.')[:-1])
                    mcts_location = value_func_location = os.path.join(location,fn_without_index)

                    player_info={'tree_type':'mcts', 'value_func_type':'nn', 'n_mcts':mctsN,'cpuct':cpuctX, \
                    'value_func_iter':value_func_iter,'mcts_iter':mcts_iter,'id':curr_id,'tournament':tournament,'mcts_location':mcts_location,'value_func_location':value_func_location,
                    'model_line':id,
                    }
                    
                    player_info.update(additional_info)
                    
                    player_info = pd.Series(player_info)
                    all_players = all_players.append(player_info,ignore_index=True)


    # for tournament_1, and all the best iter, we are sure they are better than last; tournament_4 not sure:
    all_players['better_than_last'] = better_than_last
    # all_players.loc[all_players['tournament']==1,'better_than_last'] = True
    all_players.loc[all_players['value_func_iter']=='best','better_than_last'] = True

    # sort df by tournament, n_mcts, cpuct, then iter
    gpb = all_players.groupby(['tournament','n_mcts','cpuct'])
    sort_fun = lambda x:x['value_func_iter'].replace('best',pd.NA).sort_values()
    gpb_res = gpb.apply(sort_fun)
    if gpb_res.shape[0]==1: # if only one sub category then it would be a df with one row, tournament, mcts, cpuct as the index, multiple columns for the index of the sorted value
        inds = gpb_res.columns
    else:
        inds=gpb_res.index.get_level_values(-1) # return multiindex, only need the last one to rearrange the df

    all_players= all_players.loc[inds]
    all_players = all_players.loc[~all_players['id'].duplicated()] # there might be duplicated rows, not sure why, need to be dropped
    all_players = all_players.reset_index(drop=True)
    return all_players


def extend_database(backup=True):
    # df_curr = pd.read_pickle(DATABASE_LOC)
    # if backup:
        # df_curr.to_pickle(DATABASE_LOC_backup)
        # print(f'Database backed up at {DATABASE_LOC_backup}')
    to_update_default = {'n_res':None,'color':False,'continuous_training':False,'tempThreshold':15,'dir_alpha':0}
    df_curr = expand_old_database_with_new_cols(to_save=False,backup=backup,**to_update_default)

    # new_tournaments = ['tournament_6','tournament_7']
    # tournament_to_update_pair = [('tournament_n',{dict of hyperparameters shared by all agents in tournament_n folder})]
    tournament_to_update_pair = [('tournament_8',{'color':False,'continuous_training':False,'tempThreshold':15,'dir_alpha':0}),
                             ('tournament_12',{'color':True,'continuous_training':False,'tempThreshold':15,'dir_alpha':0.03}),
                             ('tournament_13',{'color':True,'continuous_training':True,'tempThreshold':15,'dir_alpha':0.03}),
                             ('tournament_14',{'color':True,'continuous_training':False,'tempThreshold':15,'dir_alpha':0}),
                             ('tournament_15',{'color':False,'continuous_training':True,'tempThreshold':15,'dir_alpha':0}),
                             ('tournament_16',{'color':True,'continuous_training':True,'tempThreshold':15,'dir_alpha':0.3})
                            ]
    # better_than_last_l = [True,True]
    # N_new = 0
    new_df = []
    # for (d,better_than_last) in zip(new_tournaments,better_than_last_l):
    for d,to_update in tournament_to_update_pair:
        # players = create_database_trained_nn_mcts(root_dir=root_dir,tournament_dir=d,better_than_last=better_than_last)
        players = create_database_trained_nn_mcts(root_dir=root_dir,tournament_dir=d,better_than_last=True,**to_update)
        if d =='tournament_8': #manual update, when to_update does not apply to every subfolder in a tournament_x
            players.loc[players['id'].str.contains('-1'),'tempThreshold'] = 40
        new_df.append(players) # df append, not in place
        
    new_df = pd.concat(new_df,axis=0,ignore_index=True)
    df_curr = pd.concat([df_curr,new_df],axis=0,ignore_index=True)
    df_curr = df_curr.loc[~df_curr['id'].duplicated()]
    print(f'{new_df.shape[0]} new players added')
    df_curr.to_pickle(DATABASE_LOC)
    print(f'new database saved at {DATABASE_LOC}')
    return df_curr,new_df

def create_hybrid(all_p,new_tournaments = ['tournament_5','tournament_6','tournament_7'],iter_l=['best']):
    '''
    given the database df
    for each of the new tournament folders, for each iteration in iter_l
    create hybrid: nn+bfts, cog+mcts
    '''
    new_df = []
    for tournament in new_tournaments:
        tour_dir=os.path.join(root_dir,tournament)
        trained_id_list = os.listdir(tour_dir)
        for trained_id in trained_id_list:
            _,nmcts,cpuct,_,id=trained_id.split('_')
            for iter in iter_l:
                
                cog_id_partial = 'cog_id_1'

                mcts_id = nn_id = ';'.join([tournament,nmcts,cpuct,f'id-{id}',str(iter)])
                print(mcts_id)
                # cog + mcts
                try:
                    df=merge_cog_with_mcts(all_p,cog_id_partial,mcts_id)
                    new_df.append(pd.DataFrame(df).T) # need the return to be df for easy concat
                    # nn + bfs
                    bfts_id_partial = 'bfts100;prune2'
                    df = merge_nn_with_bfts(all_p, bfts_id_partial, nn_id)
                    new_df.append(pd.DataFrame(df).T)
                except:
                    print(f'probably {mcts_id} does not exist')
                
    
    new_df = pd.concat(new_df,axis=0,ignore_index=True)

    return new_df

def create_hybrid_and_extend(backup=True,new_tournaments = ['tournament_5','tournament_6','tournament_7'],iter_l=['best']):
    '''
    use create_hybrid; save the combined; 
    used by extend_database_and_base_tournament_result; be CAREFUL about the additional arguments though!!!
    '''
    df_curr = pd.read_pickle(DATABASE_LOC)
    old_N = df_curr.shape[0]
    if backup:
        df_curr.to_pickle(DATABASE_LOC_backup)
        print(f'Database backed up at {DATABASE_LOC_backup}') 
    new_df = create_hybrid(df_curr,new_tournaments = new_tournaments,iter_l=iter_l)
    df_curr = pd.concat([df_curr,new_df],axis=0,ignore_index=True)
    df_curr = df_curr.loc[~df_curr['id'].duplicated()]
    new_N = df_curr.shape[0]
    print(f'{new_N - old_N} new players added')
    df_curr.to_pickle(DATABASE_LOC)
    print(f'new database saved at {DATABASE_LOC}')
    return df_curr,new_df

def extend_database_and_base_tournament_result(extend_func = extend_database, tour_backup=True, **kwargs):
    ''' 
    for extending database and the base tournament result at the same time
    extend_func: a function for extending the database, which should return the final db and the added db;
    need to accept **kwargs
    tour_backup: whether to backup the loaded tournament result
    **kwargs: for extend_func
    '''
    _,new_df = extend_func(**kwargs)
    # curr_tournament_res = pd.read_pickle(TOURNAMENT_RES_LOC)
    curr_tournament_res = pd.read_pickle(REPEATED_TOURNAMENT_RES_LOC)
    if tour_backup:
        # curr_tournament_res.to_pickle(TOURNAMENT_RES_LOC_backup)
        curr_tournament_res.to_pickle(REPEATED_TOURNAMENT_RES_LOC_backup)
    ids_to_be_added=new_df['id']

    # ids_final=list(curr_tournament_res.index)
    # ids_final.extend(ids_to_be_added)
    # ids_final = list(set(ids_final)) # get rid of duplicates!

    ids_final_row=list(curr_tournament_res.index)
    ids_final_row.extend(ids_to_be_added)
    ids_final_row = list(set(ids_final_row)) # get rid of duplicates!
    

    ids_final_col = pd.MultiIndex.from_product([ids_final_row,('row_win','col_win','draw')])

    # curr_tournament_res_new = pd.DataFrame([],index=ids_final,columns=ids_final)
    curr_tournament_res_new = pd.DataFrame(0,index=ids_final_row,columns=ids_final_col)
    curr_tournament_res_new.loc[curr_tournament_res.index,curr_tournament_res.columns]=curr_tournament_res.values

    curr_tournament_res_new.to_pickle(REPEATED_TOURNAMENT_RES_LOC)
    print(f'new tournament base result saved at {REPEATED_TOURNAMENT_RES_LOC}')

    return curr_tournament_res_new, new_df

def main():
    # the original function for creating the original database
    # ===nn+mcts players ====
    tournaments_dirs = ['tournament_1','tournament_4']
    better_than_last_l = [True, np.nan]
    # tournaments_dirs = ['tournament_1']

    # all_players = pd.DataFrame([],columns=COLUMNS)

    all_players = []
    

    for (d,better_than_last) in zip(tournaments_dirs,better_than_last_l):

        players = create_database_trained_nn_mcts(root_dir=root_dir,tournament_dir=d,better_than_last=better_than_last)
        all_players.append(players)
    all_players = pd.concat(all_players,axis=0,ignore_index=True)

    # ===add cog_value + bfts====
    feat_w = [0.01,0.2,0.05,2,100]
    oppo_scale = 0.1
    cog_value_params = {'center':0.01,'2_con':0.2,'2_uncon':0.05,'3':2,'4':100,'oppo_scale':oppo_scale}
    
    # no pruning
    player_info = {'tree_type':'bfts','value_func_type':'cog','n_bfts':100,'pruning_thresh':2}
    player_info.update(cog_value_params)
    player_info['id'] = f";bfts{player_info['n_bfts']};prune{player_info['pruning_thresh']};cog_id_1" #cog +bfts naming: ;bftsX;pruneX;cog_id_X
    player_info = pd.Series(player_info)
    all_players = all_players.append(player_info,ignore_index=True)

    # some pruning
    player_info = {'tree_type':'bfts','value_func_type':'cog','n_bfts':100,'pruning_thresh':1}
    player_info.update(cog_value_params)
    player_info['id'] = f";bfts{player_info['n_bfts']};prune{player_info['pruning_thresh']};cog_id_1" #cog +bfts naming: ;bftsX;pruneX;cog_id_X
    player_info = pd.Series(player_info)
    all_players = all_players.append(player_info,ignore_index=True)

    # ===add cog_value + mcts====
    # use the best of 3754964, and the middle iter20 of the tournament1;mcts100;cpuct2
    mcts_id_l = ['tournament_4;mcts100;cpuct2;id-3754964;best','tournament_1;mcts100;cpuct2;21']
    cog_id_partial = 'cog_id_1'
    for m_id in mcts_id_l:
        player = merge_cog_with_mcts(all_players,cog_id_partial,m_id)
        all_players = all_players.append(player,ignore_index=True)

    # ====add nn + bfts=====
    # use the best of 3754964, and the middle iter20 of the tournament1;mcts100;cpuct2
    mcts_id_l = ['tournament_4;mcts100;cpuct2;id-3754964;best','tournament_1;mcts100;cpuct2;21']
    bfts_id_partial_l = ['bfts100;prune2','bfts100;prune1']
    for m_id in mcts_id_l:
        for bfts_id_partial in bfts_id_partial_l:
            player = merge_nn_with_bfts(all_players,bfts_id_partial,m_id)
            all_players = all_players.append(player,ignore_index=True)

    all_players.to_pickle(DATABASE_LOC)
    return all_players

def create_blank_tournament_result():
    all_players = pd.read_pickle(DATABASE_LOC)
    results_df = pd.DataFrame(index=all_players['id'], columns=all_players['id'])
    results_df.to_pickle(TOURNAMENT_RES_LOC)
    return results_df

def create_blank_repeated_tournament_result():
    all_players = pd.read_pickle(DATABASE_LOC)
    columns = pd.MultiIndex.from_product([all_players['id'],['row_win','col_win','draw']]) # for repeated tournaments, init all entries to 0, because game results need to accumulate
    rows = all_players['id']
    results_df = pd.DataFrame(0,index=rows, columns=columns)
    results_df.to_pickle(REPEATED_TOURNAMENT_RES_LOC)
    return results_df


OLD_TOURNAMENT = 3 # where tournament result happened
OLD_TOURNAMENT_train = 1 # where the models were trained
OLD_RES_LOC = f'/scratch/zz737/fiar/tournaments/results/tournament_{OLD_TOURNAMENT}/round_robin_combined.csv' #important! don't mistaken the name
def migrate_old_tournament_result():

    old_res_df = pd.read_csv(OLD_RES_LOC).set_index('Unnamed: 0')
    inds = old_res_df.index
    sub_inds = [i for i in inds if '_' in i] # excluding human, greedy, random
    inds_replace_func = lambda s:f'tournament_{OLD_TOURNAMENT_train};'+s.replace('_',';') # turn old naming, e.g. mcts100_cpuct2;1 to new naming: tournament1;mcts100;cpuct2;1
    old_res_df_new_inds=old_res_df.loc[sub_inds,sub_inds].rename(inds_replace_func).rename(inds_replace_func,axis=1) # rename old results
    ai_all_player_round_robin_base=pd.read_pickle(TOURNAMENT_RES_LOC)
    ai_all_player_round_robin_base.to_pickle(TOURNAMENT_RES_LOC_backup) # backup earlier version
    to_be_updated_inds= list(old_res_df_new_inds.index)
    ai_all_player_round_robin_base.update(old_res_df_new_inds)
    ai_all_player_round_robin_base.to_pickle(TOURNAMENT_RES_LOC)

    return ai_all_player_round_robin_base