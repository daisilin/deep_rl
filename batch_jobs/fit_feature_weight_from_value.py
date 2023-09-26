import os,sys
import numpy as np 
import pandas as pd 
import argparse 

sys.path.insert(0,'../classes')
sys.path.insert(0,'../classes/cog_related')
sys.path.insert(0,'../analysis')
import value_analysis as va 
import importlib
importlib.reload(va)

def main(iter, iter_csv_template="mcts100_cpuct2;{}.csv", moves_dir_no_raw='/scratch/zz737/fiar/tournaments/tournament_4/moves/',model_instance_str='checkpoints_mcts100_cpuct2_id-3754964'):


    moves_dir_raw = os.path.join(moves_dir_no_raw,'raw')
    iter_csv = iter_csv_template.format(iter)

    moves_dir_one_instance = os.path.join(moves_dir_raw,model_instance_str)
    try:
        one_iter_feat = va.load_moves_get_features(moves_dir_one_instance,iter_csv,epsilon=1e-9)
    except:
        print(f'no {os.path.join(moves_dir_one_instance,iter_csv)}')
        return None, None

    path = os.path.join(moves_dir_no_raw,'feat_w_from_value_fit',model_instance_str,iter_csv)

    res,res_df = va.regression(one_iter_feat, formula=None,stddz=True, path=path)

    return res, res_df


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('iter',metavar='iter',type=int,help='which model instance iteration to fit')
    my_parser.add_argument('-t','--iter_csv_template',type=str,help='like mcts100_cpuct2\;{}.csv')
    my_parser.add_argument('-d','--moves_dir_no_raw',type=str,help='like /scratch/zz737/fiar/tournaments/tournament_4/moves/')
    my_parser.add_argument('-m','--model_instance_str',type=str,help='like checkpoints_mcts100_cpuct2_id-3754964')

    args = my_parser.parse_args()
    print(args)

    main(args.iter, iter_csv_template=args.iter_csv_template, moves_dir_no_raw=args.moves_dir_no_raw, model_instance_str=args.model_instance_str)