import numpy as np
import pandas as pd
import json
from collections import defaultdict
import os

direc = '/scratch/zz737/fiar/tournaments/tournament_2/moves/'

def main():

    filenames = [f for f in os.listdir(direc + 'raw/') if f.endswith('.csv')]
    filenames = sorted(filenames,key=lambda f : f.casefold().replace('100.csv','99.csv'))

    if not os.path.exists(direc + 'splits/'):
        os.mkdir(direc + 'splits/')
    for i,f in enumerate(filenames):
    #     df = pd.read_csv(direc + 'raw/' + f,header=None,delim_whitespace=True,names=['bp','wp','color','move','rt','participant_id'])
        df = pd.read_csv(direc + 'raw/' + f,header=None,delim_whitespace=True,names=['bp','wp','color','move','rt','participant_id','value'])
        df.insert(loc=5,column='group',value=(5*(np.random.permutation(len(df))/len(df))).astype(int)+1)
        if not os.path.exists(direc + 'splits/' + str(i+1)):
            os.mkdir(direc + 'splits/' + str(i+1))
        with open(direc + 'splits/' + str(i+1) + '/data.csv','w') as f:
            f.write(df.to_csv(None, index = False, header=False,sep='\t',line_terminator ='\n')[:-1])
        for g in range(1,6):
            with open(direc + 'splits/' + str(i+1) + '/' + str(g) + '.csv','w') as f:
                f.write(df[df['group']==g].to_csv(None, index = False, header=False,sep='\t',line_terminator ='\n')[:-1])

        print(f'{f} done!')

if __name__ == '__main__':
    main()
