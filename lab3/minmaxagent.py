from typing import Tuple, Optional
from exceptions import AgentException
from connect4 import Connect4
import copy


class MinMaxAgent:
    def __init__(self, my_token: str):
        self.my_token: str = my_token

    def decide(self, connect4: Connect4) -> int:
        if connect4.who_moves != self.my_token:
            raise AgentException('Not my round')
        
        if self.my_token == 'x':
            _, best_move = self._minmax(connect4, False, 3)
        else:
            _, best_move = self._minmax(connect4, True, 3)
            
        return best_move

    def _minmax(self, s: Connect4, is_maximizer: bool, d: int) -> Tuple[float, Optional[int]]:
        if s._check_game_over():
            if s.wins == 'o':
                return 1, None
            elif s.wins == 'x':
                return -1, None
            else:
                return 0, None

        if d == 0:
            return 0.2 * s.count_corners_taken(self.my_token), None

        best_value: float = float("-inf") if is_maximizer else float("inf")
        best_move: Optional[int] = None

        for drop in s.possible_drops():
            new_connect4 = copy.deepcopy(s)
            new_connect4.drop_token(drop)
            value, _ = self._minmax(new_connect4, not is_maximizer, d - 1)

            if (is_maximizer and value > best_value) or (not is_maximizer and value < best_value):
                best_value = value
                best_move = drop

        return best_value, best_move
