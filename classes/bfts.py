"""
Best-First Tree Search (BFTS) implementation for game playing.

This module implements a variant of Best-First Search specifically designed for
game trees, with pruning and neural network evaluation integration.
"""

import logging
import math
from typing import List, Tuple, Dict, Optional, Any
import copy
import numpy as np
from utils import dotdict

# Configure logging
log = logging.getLogger(__name__)

# Constants
EPS = 1e-8
DEFAULT_ARGS = dotdict({
    'numBFSsims': 5,      # Number of BFTS simulations
    'PruningThresh': 0.5  # Threshold for pruning branches
})

class BFTS:
    """
    Best-First Tree Search implementation for game playing.
    
    Uses neural network evaluation and pruning to efficiently explore
    game trees and select promising moves.
    """
    
    def __init__(self, game, nnet, args: dotdict):
        """
        Initialize BFTS.
        
        Args:
            game: Game instance providing game mechanics
            nnet: Neural network for position evaluation
            args: Configuration arguments
        """
        self.game = game
        self.nnet = nnet
        self.args = args
        self.reset_storage()
        
    def reset_storage(self):
        """Reset all storage dictionaries."""
        self.tree = {}  # Complete game tree
        self.Qsa = {}   # State-action values
        self.Qs = {}    # State values
        self.Es = {}    # Game ended flags
        self.Vs = {}    # Valid moves cache

    @staticmethod
    def prune(v_batch: np.ndarray, pruning_thresh: float) -> np.ndarray:
        """
        Prune moves based on value threshold.
        
        Args:
            v_batch: Batch of values to consider
            pruning_thresh: Threshold for pruning
            
        Returns:
            Indices of moves to keep
        """
        vmax = np.max(v_batch)
        return np.nonzero(np.abs(v_batch - vmax) < pruning_thresh)[0]

    def getActionProb(self, canonicalBoard: np.ndarray, temp: float = 0, 
                     refresh_first: bool = False) -> List[float]:
        """
        Get action probabilities for a given board position.
        
        Args:
            canonicalBoard: Board state in canonical form
            temp: Temperature parameter (not used currently)
            refresh_first: Whether to reset storage before search
            
        Returns:
            Action probability distribution
        """
        if refresh_first:
            self.reset_storage()
            
        # Perform simulations
        for _ in range(self.args.numBFSsims):
            self.search(canonicalBoard)
            
        # Get Q-values for all actions
        s = np.array_str(canonicalBoard)
        values = [
            self.Qsa.get((s, a), -10000) 
            for a in range(self.game.getActionSize())
        ]
        
        # Select best action deterministically
        bestAs = np.array(np.argwhere(values == np.max(values))).flatten()
        bestA = np.random.choice(bestAs)
        
        # Create probability distribution
        probs = [0] * len(values)
        probs[bestA] = 1
        return probs

    def search(self, canonicalBoard: np.ndarray) -> float:
        """
        Perform one iteration of best-first search.
        
        Args:
            canonicalBoard: Board state in canonical form
            
        Returns:
            Value of the position from current player's perspective
        """
        s = np.array_str(canonicalBoard)
        
        # Check if game is ended
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1) * 100
        if self.Es[s] != 0:
            return -self.Es[s]
            
        # Expand leaf node
        if s not in self.Vs:
            return self._expand_leaf(canonicalBoard, s)
            
        # Select and explore best action
        return self._select_and_explore(canonicalBoard, s)

    def _expand_leaf(self, canonicalBoard: np.ndarray, s: str) -> float:
        """
        Expand a leaf node and evaluate all valid moves.
        
        Args:
            canonicalBoard: Board state in canonical form
            s: String representation of board state
            
        Returns:
            Best value found among children
        """
        # Get valid moves
        valids = self.game.getValidMoves(canonicalBoard, 1)
        self.Vs[s] = valids
        
        # Prepare batch of next positions
        valid_actions = np.nonzero(valids)[0]
        x_coords = valid_actions // self.game.n
        y_coords = valid_actions % self.game.n
        n_valids = len(valid_actions)
        
        # Create batch of next positions
        board_batch = np.tile(canonicalBoard, (n_valids, 1, 1))
        inds = np.arange(n_valids)
        board_batch[inds, x_coords, y_coords] = 1
        
        # Evaluate positions
        _, v_batch = self.nnet.predict_batch(board_batch)
        
        # Handle terminal positions
        for i, board in enumerate(board_batch):
            s_next = np.array_str(board)
            self.Es[s_next] = self.game.getGameEnded(board, 1) * 100
            if self.Es[s_next] != 0:
                v_batch[i] = self.Es[s_next]
                
        # Prune and store values
        to_keep = self.prune(v_batch, self.args.PruningThresh)
        best_val = -float('inf')
        
        for idx in to_keep:
            value = v_batch[idx]
            self.Qsa[(s, valid_actions[idx])] = value
            best_val = max(best_val, value)
            
        return -best_val

    def _select_and_explore(self, canonicalBoard: np.ndarray, s: str) -> float:
        """
        Select best action and explore resulting position.
        
        Args:
            canonicalBoard: Board state in canonical form
            s: String representation of board state
            
        Returns:
            Value of selected action after exploration
        """
        # Find best action among previously evaluated ones
        cur_best = -float('inf')
        best_act = -1
        val_dict = {}
        valids = self.Vs[s]
        
        for a in range(self.game.getActionSize()):
            if valids[a] and (s, a) in self.Qsa:
                val = self.Qsa[(s, a)]
                val_dict[a] = val
                if val > cur_best:
                    cur_best = val
                    best_act = a
                    
        # Explore best action
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, best_act)
        next_s = self.game.getCanonicalForm(next_s, next_player)
        v_backpropped = self.search(next_s)
        
        # Update value and return
        self.Qsa[(s, best_act)] = v_backpropped
        val_dict[best_act] = v_backpropped
        return -max(val_dict.values())

def get_children(board_arr: np.ndarray, tree: BFTS, verbose: bool = True) -> List[Tuple]:
    """
    Get all child positions from current board state.
    
    Args:
        board_arr: Current board state
        tree: BFTS instance containing search tree
        verbose: Whether to print debug information
        
    Returns:
        List of (new_board, action, value) tuples
    """
    children = []
    for (bstr, action), val in tree.Qsa.items():
        b_array = tree.game.str_rep_to_array(bstr)
        if np.array_equal(b_array, board_arr):
            new_b, new_p = tree.game.getNextState(b_array, 1, action)
            children.append((new_b, action, val))
            
            if verbose:
                log.debug("New board:")
                tree.game.display(new_b)
                log.debug(f"Action: {action}, Value: {val}\n")
                
    return children

def get_parent(board_arr: np.ndarray, action: int) -> np.ndarray:
    """
    Get parent position by undoing an action.
    
    Args:
        board_arr: Current board state
        action: Action to undo
        
    Returns:
        Parent board state
    """
    x, y = action // 9, action % 9
    parent = board_arr.copy()
    parent[x, y] = 0
    return -parent

def get_board_size(tree: BFTS) -> Dict:
    """
    Get number of pieces for each position in the tree.
    
    Args:
        tree: BFTS instance
        
    Returns:
        Dictionary mapping (board_str, action) to board size
    """
    return {
        (bstr, action): np.sum(tree.game.str_rep_to_array(bstr) != 0)
        for (bstr, action) in tree.Qsa
    }

def get_largest_board(tree: BFTS, offset: int = 0) -> List[Tuple[np.ndarray, float]]:
    """
    Get positions with the most pieces.
    
    Args:
        tree: BFTS instance
        offset: How many pieces fewer than maximum to consider
        
    Returns:
        List of (board, value) tuples
    """
    Ssa = get_board_size(tree)
    maxsize = max(Ssa.values())
    
    board_q_l = []
    for (bstr, action), size in Ssa.items():
        if size == maxsize - offset:
            val = tree.Qsa[(bstr, action)]
            b_array = tree.game.str_rep_to_array(bstr)
            new_b, new_p = tree.game.getNextState(b_array, 1, action)
            board_q_l.append((new_b, val))
            
    return board_q_l

def traverse_tree_principal_variation(board_arr: np.ndarray, tree: BFTS) -> List[Tuple]:
    """
    Find principal variation starting from given position.
    
    Args:
        board_arr: Starting board state
        tree: BFTS instance
        
    Returns:
        List of (board, action, value) tuples along principal variation
    """
    children = get_children(board_arr, tree, verbose=False)
    board_sequence = []
    
    while children:
        bestval = max(val for _, _, val in children)
        best_board_a_val = next(
            bav for bav in children if bav[2] == bestval
        )
        board_sequence.append(best_board_a_val)
        children = get_children(-best_board_a_val[0], tree, verbose=False)
        
    return board_sequence