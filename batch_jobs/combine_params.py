import numpy as np

def expand_params(theta):
    thresh = theta[0];
    delta = theta[2];
    w_center = theta[5];
    w = theta[6:10]
    lapse = theta[3];
    c_act = theta[4];
    gamma = theta[1];
    return np.hstack([[10000,thresh,gamma,lapse,1,1,w_center],
                     np.tile(w,4),
                     [0],
                     c_act*np.tile(w,4),
                     [0],
                     [delta]*17])

direc = '/scratch/zz737/fiar/tournaments/tournament_3/moves'
behavior_names = list(range(189))

def main():
    
    params_short = np.vstack([np.loadtxt(direc + '/splits/' + str(i+1) + '/params' + str(g) + '.csv',delimiter=',') 
               for i in range(len(behavior_names)) for g in range(1,6)])
    params = np.vstack([expand_params(np.loadtxt(direc + '/splits/' + str(i+1) + '/params' + str(g) + '.csv',delimiter=',')) 
               for i in range(len(behavior_names)) for g in range(1,6)])
    np.savetxt(direc + 'params_transfer_pilot_long.txt',params,fmt = '%6f')

if __name__ == '__main__':
    main()