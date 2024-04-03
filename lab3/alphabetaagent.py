from typing import Tuple, Optional
from exceptions import AgentException
from connect4 import Connect4
import copy


class AlphaBetaAgent:
    def __init__(self, my_token: str):
        self.my_token: str = my_token

    def decide(self, connect4: Connect4) -> int:
        if connect4.who_moves != self.my_token:
            raise AgentException('Not my round')
        
        if self.my_token == 'x':
            _, best_move = self._alphabeta(connect4, False, 6, 0.0, 0.0)
        else:
            _, best_move = self._alphabeta(connect4, True, 6, 0.0, 0.0)
            
        return best_move

    def _alphabeta(self, s: Connect4, x: bool, d: int, alpha: float, beta: float) -> Tuple[float, Optional[int]]:
        if s._check_game_over():
            if s.wins == 'o':
                return 1, None
            elif s.wins == 'x':
                return -1, None
            else:
                return 0, None

        if d == 0:
            return 0.2 * s.count_corners_taken(self.my_token), None

        best_value: float = float("-inf") if x else float("inf")
        best_move: Optional[int] = None

        for drop in s.possible_drops():
            new_connect4 = copy.deepcopy(s)
            new_connect4.drop_token(drop)
            value, _ = self._alphabeta(new_connect4, not x, d - 1, alpha, beta)

            if x and value > best_value:
                best_value = value
                best_move = drop
                alpha = best_value
            elif not x and value < best_value:
                best_value = value
                best_move = drop
                beta = best_value

            if best_value >= beta if x else best_value <= alpha:
                break

        return best_value, best_move

                
        
