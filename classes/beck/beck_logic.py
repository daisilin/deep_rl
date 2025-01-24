"""
Board class for m,n,k-game variant (adapted from Othello/Reversi implementation).

Original Author: Eric P. Nichols
Original Date: Feb 8, 2008
Modified for m,n,k-game variant

Board representation:
    1 = white pieces
    -1 = black pieces
    0 = empty squares

The board uses (x,y) coordinate system where:
    - x represents the column (first dimension)
    - y represents the row (second dimension)
    - (0,0) is the top-left corner
    - pieces[x][y] accesses the piece at column x, row y
"""

class Board:
    """
    A board class implementing game mechanics for a variant of m,n,k-game with piece flipping rules 
    similar to Othello/Reversi.
    
    Attributes:
        m (int): Number of rows on the board
        n (int): Number of columns on the board
        k (int): Number of pieces in a row needed to win
        pieces (list): 2D list representing the game board
        __directions (list): List of 8 possible move directions as (dx, dy) tuples
    """

    # Eight possible directions on the board as (x,y) offsets
    __directions = [
        ( 1,  1), # diagonal down-right
        ( 1,  0), # right
        ( 1, -1), # diagonal up-right
        ( 0, -1), # up
        (-1, -1), # diagonal up-left
        (-1,  0), # left
        (-1,  1), # diagonal down-left
        ( 0,  1)  # down
    ]

    def __init__(self, m, n, k):
        """
        Initialize the game board.

        Args:
            m (int): Number of rows
            n (int): Number of columns
            k (int): Number of pieces in a row needed to win
        """
        self.m = m
        self.n = n
        self.k = k
        
        # Initialize empty board
        self.pieces = [[0] * self.n for _ in range(self.m)]

    def __getitem__(self, index):
        """
        Enable bracket indexing syntax for the board.
        
        Args:
            index (int): Column index
            
        Returns:
            list: The specified column of the board
        """
        return self.pieces[index]

    def countDiff(self, color):
        """
        Count the difference between pieces of the given color and the opposite color.

        Args:
            color (int): 1 for white, -1 for black

        Returns:
            int: Positive number means more pieces of given color,
                 negative number means more pieces of opposite color
        """
        count = 0
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == color:
                    count += 1
                elif self[x][y] == -color:
                    count -= 1
        return count

    def get_legal_moves(self, color):
        """
        Get all legal moves for the given color.

        Args:
            color (int): 1 for white, -1 for black

        Returns:
            list: List of legal move coordinates as (x,y) tuples
        """
        moves = set()
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == color:
                    newmoves = self.get_moves_for_square((x, y))
                    if newmoves:
                        moves.update(newmoves)
        return list(moves)

    def has_legal_moves(self, color):
        """
        Check if the given color has any legal moves available.

        Args:
            color (int): 1 for white, -1 for black

        Returns:
            bool: True if legal moves exist, False otherwise
        """
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == color:
                    newmoves = self.get_moves_for_square((x, y))
                    if newmoves and len(newmoves) > 0:
                        return True
        return False

    def get_moves_for_square(self, square):
        """
        Get all legal moves that use the given square as a base.

        For example, if square (3,4) contains a black piece and (3,5) and (3,6) contain
        white pieces, and (3,7) is empty, (3,7) would be a legal move because all pieces
        from there back to (3,4) would be flipped.

        Args:
            square (tuple): (x,y) coordinates of the square

        Returns:
            list: List of legal move coordinates as (x,y) tuples, or None if square is empty
        """
        x, y = square
        color = self[x][y]

        if color == 0:  # Empty square
            return None

        moves = []
        for direction in self.__directions:
            move = self._discover_move(square, direction)
            if move:
                moves.append(move)

        return moves

    def execute_move(self, move, color):
        """
        Execute the given move on the board and flip pieces as necessary.

        Args:
            move (tuple): (x,y) coordinates of the move
            color (int): Color of the piece to play (1=white, -1=black)

        Raises:
            AssertionError: If the move results in no flips (invalid move)
        """
        flips = [
            flip for direction in self.__directions
            for flip in self._get_flips(move, direction, color)
        ]
        
        assert len(list(flips)) > 0, "Invalid move - no pieces would be flipped"
        
        for x, y in flips:
            self[x][y] = color

    def _discover_move(self, origin, direction):
        """
        Find the endpoint for a legal move starting at origin and moving in direction.

        Args:
            origin (tuple): Starting (x,y) coordinates
            direction (tuple): (dx,dy) movement direction

        Returns:
            tuple: (x,y) endpoint coordinates if move is legal, None otherwise
        """
        x, y = origin
        color = self[x][y]
        flips = []

        for x, y in Board._increment_move(origin, direction, self.n):
            if self[x][y] == 0:
                if flips:
                    return (x, y)
                return None
            elif self[x][y] == color:
                return None
            elif self[x][y] == -color:
                flips.append((x, y))

    def _get_flips(self, origin, direction, color):
        """
        Get list of pieces that would be flipped for a move.

        Args:
            origin (tuple): (x,y) coordinates of the move
            direction (tuple): (dx,dy) direction to check
            color (int): Color making the move

        Returns:
            list: Coordinates of pieces that would be flipped
        """
        flips = [origin]

        for x, y in Board._increment_move(origin, direction, self.n):
            if self[x][y] == 0:
                return []
            if self[x][y] == -color:
                flips.append((x, y))
            elif self[x][y] == color and len(flips) > 0:
                return flips

        return []

    @staticmethod
    def _increment_move(move, direction, n):
        """
        Generator for incrementing moves in a given direction.

        Args:
            move (tuple): Starting (x,y) coordinates
            direction (tuple): (dx,dy) direction to move
            n (int): Board size

        Yields:
            list: New [x,y] coordinates after each increment
        """
        move = list(map(sum, zip(move, direction)))
        
        while all(map(lambda x: 0 <= x < n, move)):
            yield move
            move = list(map(sum, zip(move, direction)))