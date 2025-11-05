import math
import random
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

# /Users/benreber/grad_code/PSUAI800_ConnectFour/src/ai/mcts.py
"""
Monte Carlo Tree Search implementation suitable for Connect Four AI.

Expected Game interface (the MCTS code is game-agnostic but requires these methods):
- clone() -> returns a deep copy of the game state
- legal_moves() -> list of legal move objects (e.g., column indices)
- apply_move(move) -> applies move in-place and switches player
- is_terminal() -> bool, whether the game is finished
- winner() -> returns the winning player id, or None for no winner yet, or a special value for draw
- current_player -> property or attribute indicating which player's turn it is

Usage:
    mcts = MCTS(game, time_limit=1.0)  # 1 second search
    best_move = mcts.search()
"""



class Node:
    __slots__ = ("parent", "move", "player", "children", "wins", "visits", "untried_moves")

    def __init__(self, parent: Optional["Node"], move: Optional[Any], player: Optional[Any], untried_moves: List[Any]):
        # parent: parent Node or None for root
        # move: the move that led from parent -> this node
        # player: the player who made that move (None for root)
        self.parent = parent
        self.move = move
        self.player = player
        self.children: Dict[Any, "Node"] = {}
        self.wins: float = 0.0  # total reward from perspective of node.player
        self.visits: int = 0
        self.untried_moves = list(untried_moves)

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def best_child(self, c_param: float = 1.4142135623730951) -> "Node":
        best_score = -float("inf")
        best = None
        for child in self.children.values():
            if child.visits == 0:
                exploitation = 0.0
                exploration = c_param * math.sqrt(math.log(self.visits + 1e-9) / 1.0)
                uct = exploitation + exploration
            else:
                exploitation = child.wins / child.visits
                exploration = c_param * math.sqrt(math.log(self.visits + 1e-9) / child.visits)
                uct = exploitation + exploration
            if uct > best_score:
                best_score = uct
                best = child
        if best is None:
            raise RuntimeError("No children to select from")
        return best

    def add_child(self, move: Any, player: Any, untried_moves: List[Any]) -> "Node":
        node = Node(parent=self, move=move, player=player, untried_moves=untried_moves)
        self.children[move] = node
        return node


class MCTS:
    def __init__(self, game, time_limit: Optional[float] = None, iteration_limit: Optional[int] = 1000, exploration_constant: float = 1.4142135623730951):
        """
        game: initial game state implementing the Game interface documented above.
        time_limit: max seconds to run search (exclusive with iteration_limit). If None, uses iteration_limit.
        iteration_limit: max iterations if time_limit is None.
        exploration_constant: constant used in UCT.
        """
        self.root_game = game
        self.time_limit = time_limit
        self.iteration_limit = iteration_limit if time_limit is None else float("inf")
        self.c = exploration_constant

    def search(self):
        root_state = self.root_game.clone()
        root_legal = root_state.legal_moves()
        if not root_legal:
            return None
        root_node = Node(parent=None, move=None, player=None, untried_moves=tuple(root_legal))

        start_time = time.time()
        iterations = 0

        while iterations < self.iteration_limit and (self.time_limit is None or time.time() - start_time < self.time_limit):
            node = root_node
            state = root_state.clone()

            # SELECTION
            while not state.is_terminal() and node.is_fully_expanded():
                if not node.children:
                    break
                node = node.best_child(self.c)
                state.apply_move(node.move)

            # EXPANSION
            if not state.is_terminal() and node.untried_moves:
                idx = random.randrange(len(node.untried_moves))
                move = node.untried_moves.pop(idx)
                player_who_moved = state.current_player
                state.apply_move(move)
                child_untried = state.legal_moves()
                node = node.add_child(move, player_who_moved, child_untried)

            # SIMULATION
            winner = self._simulate(state)

            # BACKPROPAGATION
            self._backpropagate(node, winner)

            iterations += 1

        # choose the move with highest visits
        best_move = None
        best_visits = -1
        for move, child in root_node.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_move = move

        return best_move

    def _simulate(self, state):
        # Heuristic rollout: try immediate win, block opponent threats, prefer center
        def winner_after(s, mv):
            ss = s.clone()
            ss.apply_move(mv)
            return ss.winner()
        def opp_player(pl):
            return 2 if pl == 1 else 1
        while not state.is_terminal():
            moves = state.legal_moves()
            if not moves:
                break
            # 1) Play immediate winning move if any
            win_move = None
            for mv in moves:
                if winner_after(state, mv) == state.current_player:
                    win_move = mv
                    break
            if win_move is not None:
                state.apply_move(win_move)
                continue
            # 2) Otherwise, avoid giving opponent an immediate win next
            safe_moves = []
            for mv in moves:
                ss = state.clone()
                ss.apply_move(mv)
                opp_moves = ss.legal_moves()
                opp_wins = any(winner_after(ss, omv) == ss.current_player for omv in opp_moves)
                if not opp_wins:
                    safe_moves.append(mv)
            cand = safe_moves if safe_moves else moves
            # 3) Bias toward center columns to improve play quality
            if isinstance(cand[0], int):  # gravity columns
                center = (state.cols - 1) / 2.0
                cand.sort(key=lambda x: abs(x - center))
                move = cand[0] if random.random() < 0.7 else random.choice(cand[: min(3, len(cand))])
            else:
                move = random.choice(cand)
            state.apply_move(move)
        return state.winner()

    def _backpropagate(self, node: Node, winner: Optional[Any]):
        # Walk up to root, updating visits and wins.
        # We add reward = 1.0 if winner == node.player, 0.5 for draw (if winner indicates draw), else 0.
        # If node.player is None (root), we still increment visits but cannot assign wins from its perspective.
        while node is not None:
            node.visits += 1
            if winner is None:
                # no winner info: skip adding wins
                pass
            else:
                # treat a special draw value as 0.5 reward
                if winner == "draw" or winner == "DRAW" or winner == 0:
                    reward = 0.5
                else:
                    reward = 1.0 if node.player is not None and winner == node.player else 0.0
                node.wins += reward
            node = node.parent


# Adapter for Connect Four-like board used in app
class ConnectGame:
    def __init__(self, board, player: int, win_len: int = 4, gravity: bool = True):
        self.board = deepcopy(board)
        self.current_player = player
        self.rows = len(board)
        self.cols = len(board[0]) if self.rows else 0
        self.win_len = win_len
        self.gravity = gravity

    def clone(self):
        return ConnectGame(deepcopy(self.board), self.current_player, self.win_len, self.gravity)

    def legal_moves(self):
        if self.gravity:
            out = []
            for c in range(self.cols):
                for r in range(self.rows - 1, -1, -1):
                    if self.board[r][c] == 0:
                        out.append(c)
                        break
            return out
        else:
            # not used in app currently
            return [(c, r) for r in range(self.rows) for c in range(self.cols) if self.board[r][c] == 0]

    def apply_move(self, move):
        if self.gravity:
            c = move
            for r in range(self.rows - 1, -1, -1):
                if self.board[r][c] == 0:
                    self.board[r][c] = self.current_player
                    break
        else:
            c, r = move
            if self.board[r][c] == 0:
                self.board[r][c] = self.current_player
        # switch player
        self.current_player = 2 if self.current_player == 1 else 1

    def is_terminal(self):
        w = self.winner()
        if w is not None:
            return True
        # full board
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] == 0:
                    return False
        return True

    def winner(self):
        # returns 1 or 2 if winner, 0 for draw, None for ongoing
        dirs = [(0, 1), (1, 0), (1, 1), (-1, 1)]
        for r in range(self.rows):
            for c in range(self.cols):
                p = self.board[r][c]
                if p == 0:
                    continue
                for dr, dc in dirs:
                    cnt = 1
                    rr, cc = r + dr, c + dc
                    while 0 <= rr < self.rows and 0 <= cc < self.cols and self.board[rr][cc] == p:
                        cnt += 1
                        rr += dr
                        cc += dc
                    rr, cc = r - dr, c - dc
                    while 0 <= rr < self.rows and 0 <= cc < self.cols and self.board[rr][cc] == p:
                        cnt += 1
                        rr -= dr
                        cc -= dc
                    if cnt >= self.win_len:
                        return p
        # draw?
        full = all(self.board[r][c] != 0 for r in range(self.rows) for c in range(self.cols))
        return 0 if full else None


def get_move(board, player, params: Optional[Dict[str, Any]] = None):
    params = params or {}
    win_len = int(params.get("win_len", 4))
    gravity = bool(params.get("gravity", True))
    sims = int(params.get("simulations", params.get("n_simulations", params.get("num_simulations", params.get("iterations", params.get("rollouts", 800))))))
    c = float(params.get("uct_c", params.get("c", params.get("exploration", 1.4))))
    tlimit = params.get("time_limit", None)
    try:
        tlimit = float(tlimit) if tlimit is not None else None
    except Exception:
        tlimit = None

    game = ConnectGame(board, player, win_len=win_len, gravity=gravity)
    if tlimit is not None and tlimit > 0.0:
        m = MCTS(game, time_limit=tlimit, iteration_limit=None, exploration_constant=c)
    else:
        m = MCTS(game, iteration_limit=sims, exploration_constant=c)
    best = m.search()
    if best is None:
        moves = game.legal_moves()
        return random.choice(moves) if moves else None
    return best