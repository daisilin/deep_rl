import os
import pandas as pd
import repeated_tournament as rt

NEW_TOURNAMENT_DIR = rt.NEW_TOURNAMENT_DIR
TOURNAMENT_BASE_LOC = rt.TOURNAMENT_BASE_LOC
participants_info = rt.participants_info

### This script takes a CSV of tournament results and converts
### them to minimal chess notation so that BayesELO can calculate
### ELO ratings. 

###[SZ] or takes a pickle file

# participants_dir = '/scratch/zz737/fiar/tournaments/tournament_1'
# results_dir = '/scratch/zz737/fiar/tournaments/results/tournament_3'

# # [SZ] turned the script into a function to parallel process multiple tournaments
# def main(results_dir,fn='round_robin_combined.csv',to_fn='pgn.pgn'):
#     # human = pd.read_csv(os.path.join(results_dir, 'vs_human.csv'), index_col=0)
#     round_robin = pd.read_csv(os.path.join(results_dir, fn), index_col=0)

#     # results = human.fillna(0) + round_robin.fillna(0)
#     results = round_robin.fillna(0)

#     with open(os.path.join(results_dir, to_fn), 'w+') as f:
#         print(os.path.join(results_dir, to_fn))
#         for p in results.index:
#             for q in results.columns:
#                 outcome = results.loc[p, q]
#                 print(p)
#                 print(q)
#                 if (outcome == 0):# or ('mcts80' in p) or ('mcts80' in q):
#                     continue
#                 elif outcome == 1:
#                     outcome = '1-0'
#                 elif outcome == -1:
#                     outcome = '0-1'
#                 else:
#                     outcome = '1/2-1/2'
#                 f.write(f'[White "{p}"][Black "{q}"][Result "{outcome}"] 1. c4 Nf6\n')

# [SZ] for repeated tournament, 
def main(results_dir,fn='round_robin_combined.p',to_fn='pgn.pgn',participants=None):
    '''
    participants: a list of participant ids; by defult, the loaded tournament result df's index
    '''
    # human = pd.read_csv(os.path.join(results_dir, 'vs_human.csv'), index_col=0)
    round_robin = pd.read_pickle(os.path.join(results_dir, fn))

    # results = human.fillna(0) + round_robin.fillna(0)
    # results = round_robin.fillna(0)
    if participants is None:
        participants = round_robin.index
    else:
        round_robin = round_robin.loc[participants,participants]
    with open(os.path.join(results_dir, to_fn), 'w+') as f:
        print(os.path.join(results_dir, to_fn))
        for p in participants:
            for q in participants:
                res_one = round_robin.loc[p,q]

                if res_one.shape[0] !=3:
                    res_one = res_one.iloc[0]
                # for i in range(res_one.row_win):
                #     f.write(f'[White "{p}"][Black "{q}"][Result "1-0"] 1. c4 Nf6\n')
                # for i in range(res_one.col_win):
                #     f.write(f'[White "{p}"][Black "{q}"][Result "0-1"] 1. c4 Nf6\n')
                # for i in range(res_one.draw):
                #     f.write(f'[White "{p}"][Black "{q}"][Result "1/2-1/2"] 1. c4 Nf6\n')
                for i in range(res_one.row_win):
                    f.write(f'[White "{p}"]\n[Black "{q}"]\n[Result "1-0"]\n1-0\n\n')
                for i in range(res_one.col_win):
                    f.write(f'[White "{p}"]\n[Black "{q}"]\n[Result "0-1"]\n0-1\n\n')
                for i in range(res_one.draw):
                    f.write(f'[White "{p}"]\n[Black "{q}"]\n[Result "1/2-1/2"]\n1/2-1/2\n\n')                
    print('Done.')