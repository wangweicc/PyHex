from typing import Callable

from gameui.Config import Config
from hexcore.Algorithms import MCTS
from hexcore.Board import Team, Board, Piece


class Player:
    """游戏玩家"""

    def __init__(self, team: Team):
        self.team = team  # 玩家所属队伍
        self.board = None  # 棋盘
        self.operation: Callable = None  # 下棋动作

    def set_board(self, board: Board):
        self.board = board

    def set_piece(self, piece: Piece):
        """玩家下一步棋"""
        piece.set_team(self.team)  # 设置棋子的所属方
        return self.board.set_piece(piece)

    def bind_play_operation(self, func: Callable):
        """绑定下棋时执行的操作"""
        self.operation = func

    def let_me_play(self):
        """轮到玩家操作时自动调用"""
        if self.operation:
            self.operation()


class Human(Player):
    """人类玩家"""

    def __init__(self, team: Team):
        super(Human, self).__init__(team)


class AI(Player):
    """AI玩家"""

    def __init__(self, team: Team):
        super(AI, self).__init__(team)
        self.team = team
        self.level = Config.ai_level

    def let_me_play(self):
        mcts = MCTS(self.board.state(), self.team.value)
        print(f"[AI] {self.team} searching in {self.level}s...", end='')
        mcts.search(Config.ai_level)

        best_move = mcts.best_move()
        if best_move is None:
            # 如果没有找到最佳移动，选择第一个可用的移动
            available_moves = [(piece.row, piece.col) for piece in self.board.items() if piece.team == Team.NONE]
            if available_moves:
                row, col = available_moves[0]
            else:
                return
        else:
            row, col = best_move

        self.set_piece(Piece(row, col))
        simulate_times, node_count, run_time = mcts.statistics
        print(
            f"\r[AI] {self.team} set piece at ({row}, {col})\t| sims={simulate_times}, nodes={node_count}, time={run_time:.2f}s")