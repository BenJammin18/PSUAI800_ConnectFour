from __future__ import annotations
import math
import random
from copy import deepcopy
from typing import List, Tuple, Optional, Dict, Any

# Board cells: 0 empty, 1 = player1 (Red), 2 = player2 (Yellow)
# Moves: if gravity=True, move is column index; else move is (col, row)


def legal_moves(board: List[List[int]], gravity: bool) -> List[Any]:
    rows = len(board)
    cols = len(board[0]) if rows else 0
    if gravity:
        out = []
        for c in range(cols):
            if any(board[r][c] == 0 for r in range(rows)):
                out.append(c)
        return out
    else:
        return [(c, r) for r in range(rows) for c in range(cols) if board[r][c] == 0]


def apply_move(board: List[List[int]], move: Any, player: int, gravity: bool) -> Optional[Tuple[int, int]]:
    if gravity:
        c = int(move)
        for r in range(len(board) - 1, -1, -1):
            if board[r][c] == 0:
                board[r][c] = player
                return r, c
        return None
    else:
        c, r = move
        if board[r][c] == 0:
            board[r][c] = player
            return r, c
        return None


def is_full(board: List[List[int]]) -> bool:
    return all(cell != 0 for row in board for cell in row)


def check_winner(board: List[List[int]], win_len: int) -> int:
    rows = len(board)
    cols = len(board[0]) if rows else 0
    dirs = [(0, 1), (1, 0), (1, 1), (-1, 1)]
    for r in range(rows):
        for c in range(cols):
            p = board[r][c]
            if p == 0:
                continue
            for dr, dc in dirs:
                cnt = 1
                rr, cc = r + dr, c + dc
                while 0 <= rr < rows and 0 <= cc < cols and board[rr][cc] == p:
                    cnt += 1
                    rr += dr
                    cc += dc
                rr, cc = r - dr, c - dc
                while 0 <= rr < rows and 0 <= cc < cols and board[rr][cc] == p:
                    cnt += 1
                    rr -= dr
                    cc -= dc
                if cnt >= win_len:
                    return p
    return 0


def evaluate_position(board: List[List[int]], player: int, win_len: int) -> int:
    # Heuristic inspired by standard Connect Four evaluation
    # Score windows of size win_len across all directions and center bias
    opponent = 2 if player == 1 else 1
    rows = len(board)
    cols = len(board[0]) if rows else 0

    def score_window(window: List[int]) -> int:
        score = 0
        p_count = window.count(player)
        o_count = window.count(opponent)
        e_count = window.count(0)
        if p_count == win_len:
            score += 100000
        elif p_count == win_len - 1 and e_count == 1:
            score += 1200
        elif p_count == win_len - 2 and e_count == 2:
            score += 50
        if o_count == win_len - 1 and e_count == 1:
            score -= 1500
        elif o_count == win_len - 2 and e_count == 2:
            score -= 60
        return score

    score = 0
    # Center column preference
    center_col = cols // 2 if cols else 0
    if cols:
        center_vals = [board[r][center_col] for r in range(rows)]
        score += center_vals.count(player) * 6
        # slight adversarial bias
        score -= center_vals.count(opponent) * 4

    # Horizontal
    for r in range(rows):
        row_list = board[r]
        for c in range(cols - win_len + 1):
            window = row_list[c : c + win_len]
            score += score_window(window)

    # Vertical
    for c in range(cols):
        col_list = [board[r][c] for r in range(rows)]
        for r in range(rows - win_len + 1):
            window = col_list[r : r + win_len]
            score += score_window(window)

    # Diagonal down-right
    for r in range(rows - win_len + 1):
        for c in range(cols - win_len + 1):
            window = [board[r + i][c + i] for i in range(win_len)]
            score += score_window(window)

    # Diagonal up-right
    for r in range(win_len - 1, rows):
        for c in range(cols - win_len + 1):
            window = [board[r - i][c + i] for i in range(win_len)]
            score += score_window(window)

    return score


def minimax(board: List[List[int]], depth: int, alpha: float, beta: float, maximizing_player: bool, player: int, gravity: bool, win_len: int) -> Tuple[float, Optional[Any]]:
    winner = check_winner(board, win_len)
    opponent = 2 if player == 1 else 1

    if winner == player:
        return 1e9, None
    if winner == opponent:
        return -1e9, None
    if depth == 0 or is_full(board):
        return float(evaluate_position(board, player, win_len)), None

    moves = legal_moves(board, gravity)
    if not moves:
        return 0.0, None

    # Center bias ordering to improve pruning
    if moves and isinstance(moves[0], int):
        center = (len(board[0]) - 1) / 2.0
        moves.sort(key=lambda x: abs(x - center))

    if maximizing_player:
        value = -math.inf
        best_move = random.choice(moves)
        for mv in moves:
            nb = deepcopy(board)
            apply_move(nb, mv, player, gravity)
            score, _ = minimax(nb, depth - 1, alpha, beta, False, player, gravity, win_len)
            if score > value:
                value = score
                best_move = mv
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value, best_move
    else:
        value = math.inf
        best_move = random.choice(moves)
        for mv in moves:
            nb = deepcopy(board)
            apply_move(nb, mv, opponent, gravity)
            score, _ = minimax(nb, depth - 1, alpha, beta, True, player, gravity, win_len)
            if score < value:
                value = score
                best_move = mv
            beta = max(-math.inf, min(beta, value))
            if alpha >= beta:
                break
        return value, best_move


def immediate_tactics(board: List[List[int]], player: int, gravity: bool, win_len: int) -> Optional[Any]:
    # 1) Play winning move now
    for mv in legal_moves(board, gravity):
        nb = deepcopy(board)
        apply_move(nb, mv, player, gravity)
        if check_winner(nb, win_len) == player:
            return mv
    # 2) Block opponent's immediate win
    opponent = 2 if player == 1 else 1
    for mv in legal_moves(board, gravity):
        nb = deepcopy(board)
        apply_move(nb, mv, opponent, gravity)
        if check_winner(nb, win_len) == opponent:
            return mv
    return None


def get_move(board: List[List[int]], player: int, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    params = params or {}
    gravity = bool(params.get("gravity", True))
    win_len = int(params.get("win_len", 4))
    depth = int(params.get("depth", params.get("max_depth", 4)))
    use_ab = bool(params.get("use_alpha_beta", True))

    moves = legal_moves(board, gravity)
    if not moves:
        return None

    # Quick tactics first
    mv = immediate_tactics(board, player, gravity, win_len)
    if mv is not None:
        return mv

    # Minimax
    if use_ab:
        _, best = minimax(board, depth, -math.inf, math.inf, True, player, gravity, win_len)
    else:
        # Fallback: alpha=beta-less (still using same routine works fine)
        _, best = minimax(board, depth, -math.inf, math.inf, True, player, gravity, win_len)

    return best if best in moves else random.choice(moves)


def best_move(board: List[List[int]], player: int, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    return get_move(board, player, params)


class Minimax:
    def __init__(self, board: List[List[int]], player: int, params: Optional[Dict[str, Any]] = None):
        self.board = deepcopy(board)
        self.player = player
        self.params = params or {}
        self.gravity = bool(self.params.get("gravity", True))
        self.win_len = int(self.params.get("win_len", 4))
        self.depth = int(self.params.get("depth", self.params.get("max_depth", 4)))
        self.use_ab = bool(self.params.get("use_alpha_beta", True))

    def search(self) -> Optional[Any]:
        mv = immediate_tactics(self.board, self.player, self.gravity, self.win_len)
        if mv is not None:
            return mv
        if self.use_ab:
            _, best = minimax(self.board, self.depth, -math.inf, math.inf, True, self.player, self.gravity, self.win_len)
        else:
            _, best = minimax(self.board, self.depth, -math.inf, math.inf, True, self.player, self.gravity, self.win_len)
        moves = legal_moves(self.board, self.gravity)
        return best if best in moves else (random.choice(moves) if moves else None)
