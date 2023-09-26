'''
[SZ] To deal with agents with the same type, eg mcts100_cpuct2, but from different folders/tournaments
also expand to include: cog_model+bfts, cog_model + mcts, nn + bfts, nn + mcts
so need new labeling system
human, greedy, random

cog_model: fix parameter
bfts: 
	pruning: [2,1.5,1,0.5]
	num: [50,100,150,200,300]

mcts:old tournament; longer trained mcts100cpuct2
nn: same

nn+bfts: use which nn and which bfts? (best of each line / best of the best)
cog + mcts: which mcts? (best of each line/ best of the best/ co-trained: which means use a different cog param)

'''

