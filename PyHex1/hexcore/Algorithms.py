from copy import deepcopy
from math import log, sqrt
from queue import Queue
from random import choice, random
from time import process_time
from typing import List, Iterator, Callable, Tuple, Dict

Pos = Tuple[int, int]
State = List[List[int]]
RedTeam, BlueTeam, NoneTeam = -1, 1, 0


class BFS:
    """广度优先搜索算法, 用于裁判判断棋子是否联通两个边界, 并获取联通的路径"""

    def __init__(self, state: State):
        self.state = state
        self.size = len(state)

    def get_neighbors(self, pos: Pos, team: int) -> Iterator[Pos]:
        """
        获取某个棋子的邻接棋子, 邻接棋子指: 当前位置周围 6 个方向中, 与当前棋子同属一方的棋子
        :param pos: 棋子位置
        :param team: 棋子所属的队伍
        :return: 该棋子邻接棋子的迭代器
        """
        directions = [(1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0)]  # 棋子的六个移动方向
        for dx, dy in directions:
            row, col = pos[0] + dx, pos[1] + dy
            if not (0 <= row < self.size and 0 <= col < self.size):  # 如果坐标不在棋盘范围内, 无效
                continue
            if team == self.state[row][col]:  # 如果属于同一方, 是邻接棋子
                yield row, col

    def find_path(self, start: Pos, team: int, stop_condition: Callable[[Pos], bool]) -> List[Pos]:
        """
        寻找指定起点到 "满足条件的点" 的路径, 如果没有, 返回 []
        :param team: 获取红队还是蓝队的路径
        :param start: 给定的起点位置
        :param stop_condition: 搜索结束条件, 用于指定终点位置
        :return: 起点到终点的路径
        """
        queue = Queue()
        visited = {}  # 记录已访问的结点, key 为结点坐标x,y, value 为其上一级结点坐标x,y
        queue.put(start)  # 起点位置入队 x,y
        visited[start] = (-1, -1)  # 起点已经访问, 无上级结点
        while not queue.empty():  # 如果队列未空
            node = queue.get()  # 队头结点出队
            if stop_condition(node):  # 如果 node 已经是终点
                path = []  # 记录起点到终点的路径
                pre = node  # 从终点反推回起点
                while pre != (-1, -1):
                    path.append(pre)
                    pre = visited[pre]  # pre 回到上一级
                path.reverse()  # 反转一次, 前面得到的是终点逆推到起点的路径
                return path
            # 把 node 相邻的, 且没有访问过的结点入队
            for nb in self.get_neighbors(node, team):
                if nb not in visited:
                    queue.put(nb)
                    visited[nb] = node  # 记录相邻结点的上一级结点
        return []

    def find_red_path(self) -> List[Pos]:
        """检查红方是否已经联通上下两边, 如果是, 返回路径, 如果没有, 返回 []"""

        # 遍历棋盘第一行(红方上界), 去找是否存在连接了最后一行(红方下界)的路径
        for col in range(self.size):
            if self.state[0][col] != RedTeam:
                continue  # 该位置没有红方棋子
            start_pos = 0, col
            last_row = self.size - 1  # 下界的行号
            path = self.find_path(start_pos, RedTeam, lambda pos: pos[0] == last_row)
            if path:
                return path
        return []

    def find_blue_path(self) -> List[Pos]:
        """检查蓝方是否已经联通左右两边, 如果是, 返回路径, 如果没有, 返回 []"""
        # 遍历棋盘第一列(蓝方左边界), 去找是否存在连接了最后一列(蓝方右边界)的路径
        for row in range(self.size):
            if self.state[row][0] != BlueTeam:
                continue  # 该位置没有蓝方棋子
            start_pos = row, 0
            last_col = self.size - 1  # 右边界的列号
            path = self.find_path(start_pos, BlueTeam, lambda pos: pos[1] == last_col)
            if path:
                return path
        return []

    def get_winner(self):
        """获取已经联通自己边界的的队伍"""
        if self.find_red_path():
            return RedTeam
        elif self.find_blue_path():
            return BlueTeam
        else:
            return NoneTeam


class UnionFind:
    """带权路径压缩的并查集"""

    def __init__(self):
        self.parent = {}  # 存储结点之间的关系
        self.rank = {}  # 存储结点对应的树高

        # edge_one 和 edge_two 用做标记, 当棋子落在己方边界位置, 就与对应的 edge 连接
        # 最后判断 edge_one 和 edge_two 是否连接即可知道棋盘两个边界是否联通
        self.edge_one = (-1, -1)
        self.edge_two = (-2, -2)
        self.parent[self.edge_one] = self.edge_one
        self.parent[self.edge_two] = self.edge_two
        self.rank[self.edge_one] = 0
        self.rank[self.edge_two] = 0

    def find(self, x: Pos) -> Pos:
        """查找结点的根节点(代表元), 如果没找到就加入并查集, 查找过程中会进行路径压缩"""
        if x not in self.parent:  # 元素不存在
            self.parent[x] = x
            self.rank[x] = 0

        while x != self.parent[x]:  # 还没有达到根节点
            gx = self.parent[self.parent[x]]  # 祖父结点
            self.parent[x] = gx  # 隔代压缩, 减小树高
            x = gx
        return x

    def connected(self, x: Pos, y: Pos) -> bool:
        """判断两个结点是否联通"""
        return self.find(x) == self.find(y)  # 根节点则联通

    def union(self, x: Pos, y: Pos) -> bool:
        """连接两个元素"""
        rx = self.find(x)  # x 的根节点
        ry = self.find(y)

        if rx == ry:
            return False

        # 将对应树高小的结点挂到树高大的结点上, 降低合并后的树高
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:  # 一样高, 随便挂, 整体树高 +1
            self.parent[rx] = ry
            self.rank[ry] += 1
        return True

    def union_edge_one(self, p: Pos):
        """将结点 p 与标记位置 1 联通"""
        return self.union(self.edge_one, p)

    def union_edge_two(self, p: Pos):
        """将结点 p 与标记位置 2 联通"""
        return self.union(self.edge_two, p)

    def edge_connected(self) -> bool:
        """通过判断两个标记位置是否联通判断整个棋盘的两个边界是否联通"""
        return self.connected(self.edge_one, self.edge_two)


class HexEvaluator:
    """海克斯棋专用评估器 - 平衡进攻和防守"""

    def __init__(self, size: int):
        self.size = size

    def evaluate_state(self, state: State, team: int) -> float:
        """评估棋盘状态对指定队伍的有利程度 - 同时考虑进攻和防守"""
        if team == RedTeam:
            own_potential = self._calculate_potential(state, RedTeam, True)  # 红方进攻潜力
            opponent_potential = self._calculate_potential(state, BlueTeam, False)  # 蓝方进攻潜力
        else:
            own_potential = self._calculate_potential(state, BlueTeam, True)  # 蓝方进攻潜力
            opponent_potential = self._calculate_potential(state, RedTeam, False)  # 红方进攻潜力

        # 平衡公式：自己的潜力 - 对手的潜力 * 权重
        # 当对手潜力很高时（快要赢了），防守变得更重要
        defense_weight = 1.5 if opponent_potential > 0.7 else 1.0
        return own_potential - defense_weight * opponent_potential

    def _calculate_potential(self, state: State, team: int, is_own: bool) -> float:
        """计算指定队伍的连接潜力"""
        if team == RedTeam:
            return self._calculate_red_potential(state, is_own)
        else:
            return self._calculate_blue_potential(state, is_own)

    def _calculate_red_potential(self, state: State, is_own: bool) -> float:
        """计算红方连接上下边界的潜力"""
        # 创建距离图
        top_distances = [[float('inf')] * self.size for _ in range(self.size)]
        bottom_distances = [[float('inf')] * self.size for _ in range(self.size)]

        # 初始化顶部距离
        for col in range(self.size):
            if state[0][col] == RedTeam:
                top_distances[0][col] = 0
            elif state[0][col] == NoneTeam:
                top_distances[0][col] = 1

        # 从顶部传播距离
        for row in range(1, self.size):
            for col in range(self.size):
                if state[row][col] == BlueTeam:  # 对手棋子是障碍
                    continue

                min_neighbor = float('inf')
                for dr, dc in [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]:
                    r, c = row + dr, col + dc
                    if 0 <= r < self.size and 0 <= c < self.size:
                        min_neighbor = min(min_neighbor, top_distances[r][c])

                if min_neighbor < float('inf'):
                    if state[row][col] == RedTeam:
                        top_distances[row][col] = min_neighbor
                    else:  # 空位
                        top_distances[row][col] = min_neighbor + 1

        # 初始化底部距离
        for col in range(self.size):
            if state[self.size - 1][col] == RedTeam:
                bottom_distances[self.size - 1][col] = 0
            elif state[self.size - 1][col] == NoneTeam:
                bottom_distances[self.size - 1][col] = 1

        # 从底部传播距离
        for row in range(self.size - 2, -1, -1):
            for col in range(self.size):
                if state[row][col] == BlueTeam:  # 对手棋子是障碍
                    continue

                min_neighbor = float('inf')
                for dr, dc in [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]:
                    r, c = row + dr, col + dc
                    if 0 <= r < self.size and 0 <= c < self.size:
                        min_neighbor = min(min_neighbor, bottom_distances[r][c])

                if min_neighbor < float('inf'):
                    if state[row][col] == RedTeam:
                        bottom_distances[row][col] = min_neighbor
                    else:  # 空位
                        bottom_distances[row][col] = min_neighbor + 1

        # 计算最短路径
        min_path = float('inf')
        for col in range(self.size):
            path_length = top_distances[0][col] + bottom_distances[self.size - 1][col]
            if path_length < min_path:
                min_path = path_length

        # 转换为潜力值（距离越短，潜力越高）
        return 1.0 / (min_path + 1) if min_path < float('inf') else 0.0

    def _calculate_blue_potential(self, state: State, is_own: bool) -> float:
        """计算蓝方连接左右边界的潜力"""
        # 创建距离图
        left_distances = [[float('inf')] * self.size for _ in range(self.size)]
        right_distances = [[float('inf')] * self.size for _ in range(self.size)]

        # 初始化左侧距离
        for row in range(self.size):
            if state[row][0] == BlueTeam:
                left_distances[row][0] = 0
            elif state[row][0] == NoneTeam:
                left_distances[row][0] = 1

        # 从左侧传播距离
        for col in range(1, self.size):
            for row in range(self.size):
                if state[row][col] == RedTeam:  # 对手棋子是障碍
                    continue

                min_neighbor = float('inf')
                for dr, dc in [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]:
                    r, c = row + dr, col + dc
                    if 0 <= r < self.size and 0 <= c < self.size:
                        min_neighbor = min(min_neighbor, left_distances[r][c])

                if min_neighbor < float('inf'):
                    if state[row][col] == BlueTeam:
                        left_distances[row][col] = min_neighbor
                    else:  # 空位
                        left_distances[row][col] = min_neighbor + 1

        # 初始化右侧距离
        for row in range(self.size):
            if state[row][self.size - 1] == BlueTeam:
                right_distances[row][self.size - 1] = 0
            elif state[row][self.size - 1] == NoneTeam:
                right_distances[row][self.size - 1] = 1

        # 从右侧传播距离
        for col in range(self.size - 2, -1, -1):
            for row in range(self.size):
                if state[row][col] == RedTeam:  # 对手棋子是障碍
                    continue

                min_neighbor = float('inf')
                for dr, dc in [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]:
                    r, c = row + dr, col + dc
                    if 0 <= r < self.size and 0 <= c < self.size:
                        min_neighbor = min(min_neighbor, right_distances[r][c])

                if min_neighbor < float('inf'):
                    if state[row][col] == BlueTeam:
                        right_distances[row][col] = min_neighbor
                    else:  # 空位
                        right_distances[row][col] = min_neighbor + 1

        # 计算最短路径
        min_path = float('inf')
        for row in range(self.size):
            path_length = left_distances[row][0] + right_distances[row][self.size - 1]
            if path_length < min_path:
                min_path = path_length

        # 转换为潜力值（距离越短，潜力越高）
        return 1.0 / (min_path + 1) if min_path < float('inf') else 0.0

    def evaluate_move(self, state: State, move: Pos, team: int) -> float:
        """评估一个移动的价值 - 同时考虑进攻和防守价值"""
        row, col = move

        # 测试这个移动的效果
        test_state = [row[:] for row in state]  # 深拷贝
        test_state[row][col] = team

        # 计算进攻价值（这个移动对自己连接路径的帮助）
        offensive_value = self.evaluate_state(test_state, team)

        # 计算防守价值（这个移动对对手连接路径的阻碍）
        opponent = BlueTeam if team == RedTeam else RedTeam
        original_opponent_potential = self.evaluate_state(state, opponent)
        new_opponent_potential = self.evaluate_state(test_state, opponent)
        defensive_value = original_opponent_potential - new_opponent_potential

        # 综合价值：进攻价值 + 防守价值 * 权重
        # 早期游戏更注重进攻，后期更注重防守
        empty_cells = sum(1 for row in state for cell in row if cell == NoneTeam)
        game_phase = 1.0 - (empty_cells / (self.size * self.size))  # 0=开始, 1=结束
        defense_weight = 0.8 + game_phase * 0.4  # 后期防守权重更高

        return offensive_value + defense_weight * defensive_value


class Node:
    """
    蒙特卡洛搜索树的结点
    """

    def __init__(self, move: Pos = None, team: int = NoneTeam, parent=None):
        self.move = move  # 落子位置
        self.team = team  # 棋子所属队伍
        self.parent = parent  # 父节点
        self.visits = 0  # 该结点被访问的次数
        self.reward = 0  # 该结点处的获胜次数
        self.children = []  # 子节点

    @property
    def value(self):
        """根据 uct 算法得出该结点的 value"""
        if self.visits == 0:
            return float('inf')
        # UCT公式: exploitation + exploration
        exploitation = self.reward / self.visits
        exploration = sqrt(2 * log(self.parent.visits) / self.visits)
        return exploitation + 1.4 * exploration  # 增加探索权重


class BoardState:
    """棋盘状态类"""

    def __init__(self, state: State, turn: int):
        self.state = deepcopy(state)  # 当前棋盘状态, 二维数组
        self.size = len(state)  # 棋盘大小
        self.turn = turn  # 当前下棋方
        self.red_uf = UnionFind()  # 红方的并查集
        self.blue_uf = UnionFind()  # 蓝方的并查集
        self.evaluator = HexEvaluator(self.size)
        self.init_union_find()

    def init_union_find(self):
        """根据棋盘状态, 初始化对应的并查集"""
        for row in range(self.size):
            for col in range(self.size):
                turn = self.state[row][col]
                self.update_union_find((row, col), turn)

    def get_winner(self) -> int:
        """使用并查集判断获胜者"""
        if self.red_uf.edge_connected():
            return RedTeam
        elif self.blue_uf.edge_connected():
            return BlueTeam
        return NoneTeam

    def get_neighbors(self, pos: Pos, team: int) -> Iterator[Pos]:
        """
        获取棋子的邻居列表
        :param pos: 棋子坐标
        :param team: 棋子所属队伍
        :return: 邻居坐标列表
        """
        directions = [(1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0)]  # 棋子的六个移动方向
        for dx, dy in directions:
            row, col = pos[0] + dx, pos[1] + dy
            if not (0 <= row < self.size and 0 <= col < self.size):  # 如果坐标不在棋盘范围内, 无效
                continue
            if team == self.state[row][col]:  # 如果属于同一方, 是邻接棋子
                yield row, col

    def change_turn(self):
        """交换下棋方"""
        if self.turn == RedTeam:
            self.turn = BlueTeam
        elif self.turn == BlueTeam:
            self.turn = RedTeam

    def update_union_find(self, move: Pos, turn: int):
        """更新并查集状态, 尝试将给定坐标与并查集中的结点联通"""
        if turn == NoneTeam:
            return

        row, col = move
        if turn == RedTeam:
            if row == 0:  # 红队棋子下在第一行(红方上边界)
                self.red_uf.union_edge_one(move)
            if row == self.size - 1:  # 红队棋子下在最后一行(红方下边界)
                self.red_uf.union_edge_two(move)
            # 棋子下在非边界位置, 将它与邻居连接起来
            for nb in self.get_neighbors(move, turn):
                self.red_uf.union(move, nb)

        elif turn == BlueTeam:
            if col == 0:
                self.blue_uf.union_edge_one(move)
            if col == self.size - 1:
                self.blue_uf.union_edge_two(move)
            for nb in self.get_neighbors(move, turn):
                self.blue_uf.union(move, nb)

    def set_piece(self, move: Pos):
        """下一步棋, 修改棋盘的状态, 更新并查集"""
        row, col = move
        self.state[row][col] = self.turn
        self.update_union_find(move, self.turn)
        self.change_turn()

    def get_moves(self) -> List[Pos]:
        """获取可以下棋的位置"""
        moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.state[row][col] == NoneTeam:
                    moves.append((row, col))
        return moves

    def evaluate_move(self, move: Pos) -> float:
        """评估一个移动的价值"""
        return self.evaluator.evaluate_move(self.state, move, self.turn)

    def print(self):
        for row in range(self.size):
            for col in range(self.size):
                print(self.state[row][col], end='\t')
            print()


class MCTS:
    """
    蒙特卡洛搜索树 - 平衡进攻和防守
    """

    def __init__(self, init_state, turn: int):
        self.root_state = BoardState(init_state, turn)
        self.root = Node((-1, -1), turn, None)

        # 一些统计信息
        self.run_time = 0
        self.simulate_times = 0

    def search(self, time_limit: int = 1) -> None:
        """在限定的时间内对树进行展开和模拟"""
        start_time = process_time()
        simulate_times = 0

        while process_time() - start_time < time_limit:
            node, state = self.select()
            reward = self.simulate(state)
            self.back_propagate(node, reward)
            simulate_times += 1

        # 记录统计信息
        self.run_time = process_time() - start_time
        self.simulate_times = simulate_times

    def select(self) -> Tuple[Node, BoardState]:
        """选择一个结点, 用于下一步模拟操作"""
        node = self.root
        state_copy = BoardState(self.root_state.state, self.root_state.turn)

        while node.children:  # 如果没达到叶子节点, 一直深入下去
            # 选择UCB值最大的子节点
            best_child = None
            best_value = -float('inf')

            for child in node.children:
                ucb_value = child.value
                if ucb_value > best_value:
                    best_value = ucb_value
                    best_child = child

            node = best_child
            state_copy.set_piece(node.move)

            # 如果子节点还没有被探索, 直接选择它
            if node.visits == 0:
                return node, state_copy

        # 如果达到叶子结点, 就进行扩展
        if self.expand(node, state_copy):
            # 选择第一个子节点继续
            if node.children:
                node = node.children[0]
                state_copy.set_piece(node.move)

        return node, state_copy

    def expand(self, parent: Node, state: BoardState):
        # 如果游戏在该节点处已经结束, 无需扩展
        if state.get_winner() != NoneTeam:
            return False

        moves = state.get_moves()
        if not moves:
            return False

        # 按移动价值排序
        moves_with_score = []
        for move in moves:
            score = state.evaluate_move(move)
            moves_with_score.append((move, score))

        # 按分数降序排序
        moves_with_score.sort(key=lambda x: x[1], reverse=True)

        # 只扩展最有希望的前N个移动
        max_children = min(15, len(moves_with_score))
        for i in range(max_children):
            move, score = moves_with_score[i]
            new_node = Node(move, state.turn, parent)
            parent.children.append(new_node)

        return True

    def simulate(self, state: BoardState) -> float:
        """使用快速评估模拟"""
        # 使用完整的游戏模拟而不是静态评估
        moves = state.get_moves()
        temp_state = BoardState(state.state, state.turn)

        while moves and temp_state.get_winner() == NoneTeam:
            # 使用启发式选择移动
            if len(moves) > 5:
                # 评估前几个移动
                moves_with_score = []
                for move in moves[:10]:  # 只评估前10个以节省时间
                    score = temp_state.evaluate_move(move)
                    moves_with_score.append((move, score))
                moves_with_score.sort(key=lambda x: x[1], reverse=True)
                move = moves_with_score[0][0]
            else:
                move = choice(moves)

            temp_state.set_piece(move)
            moves.remove(move)

        winner = temp_state.get_winner()
        if winner == state.turn:
            return 1.0
        elif winner != NoneTeam:
            return 0.0
        else:
            # 平局或未结束，使用评估函数
            own_potential = state.evaluator.evaluate_state(temp_state.state, state.turn)
            opponent = BlueTeam if state.turn == RedTeam else RedTeam
            opponent_potential = state.evaluator.evaluate_state(temp_state.state, opponent)
            return (own_potential - opponent_potential + 1) / 2  # 映射到[0,1]

    def back_propagate(self, node: Node, reward: float):
        """从给定结点反向传播, 更新其父节点信息"""
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent

    def best_move(self) -> Pos:
        """获取最佳下棋位置"""
        if not self.root.children:
            return None

        # 选择访问次数最多的节点（更可靠）
        max_visits = max(ch.visits for ch in self.root.children)
        max_visit_chs = [ch for ch in self.root.children if ch.visits == max_visits]

        # 如果有多个，选择胜率最高的
        if len(max_visit_chs) > 1:
            best_win_rate = -1
            best_child = None
            for child in max_visit_chs:
                win_rate = child.reward / child.visits
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_child = child
            return best_child.move
        else:
            return max_visit_chs[0].move

    @property
    def tree_node_num(self) -> int:
        """统计树的结点数量"""
        queue = Queue()
        count = 0
        queue.put(self.root)
        while not queue.empty():
            node = queue.get()
            count += 1
            for child in node.children:
                queue.put(child)
        return count

    @property
    def statistics(self) -> tuple:
        """本次搜索的开销信息"""
        return self.simulate_times, self.tree_node_num, self.run_time