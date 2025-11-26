from __future__ import annotations
import os
import pickle
import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple


# Minimal tabular Q-learning agent with epsilon-greedy policy.
# Designed to integrate with the existing Streamlit app which calls one of
# get_move | policy_move | act(board, player, params) and expects a legal move.


# In-memory Q-table: maps (state_features_tuple, action) -> float
Q_TABLE: Dict[Tuple[Tuple[int, ...], Any], float] = {}
DECISION_STATS: Dict[str, int] = {
    "tactic": 0,
    "explore": 0,
    "exploit": 0,
    "fallback": 0,
}
# When True, skip auto-loading a cached/pretrained Q-table for this process
_DISABLE_AUTO_LOAD: bool = False


def _cache_path() -> str:
    base = os.path.join(os.path.expanduser("~"), ".cache", "psuai800_connect4")
    try:
        os.makedirs(base, exist_ok=True)
    except Exception:
        pass
    return os.path.join(base, "q_table.pkl")


def save_q_table(path: Optional[str] = None) -> None:
    path = path or _cache_path()
    try:
        with open(path, "wb") as f:
            pickle.dump(Q_TABLE, f)
    except Exception:
        # Non-fatal
        pass


def load_q_table(path: Optional[str] = None) -> None:
    """Load Q-table from disk. Checks repo location first, then cache."""
    if _DISABLE_AUTO_LOAD:
        return

    # Priority: explicit path > repo pretrained > user cache
    search_paths = []
    if path:
        search_paths.append(path)
    else:
        # Check repo location first (for version-controlled pretrained table)
        repo_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..",
            "data",
            "pretrained_q_table.pkl",
        )
        search_paths.append(os.path.normpath(repo_path))
        # Fallback to cache
        search_paths.append(_cache_path())

    for p in search_paths:
        try:
            if os.path.exists(p):
                with open(p, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    Q_TABLE.clear()
                    Q_TABLE.update(data)
                return  # Successfully loaded
        except Exception:
            continue  # Try next path
    # Non-fatal if all paths fail


### Board helpers (local copies to avoid cross-module imports)


def legal_moves(board: List[List[int]], gravity: bool) -> List[Any]:
    rows = len(board)
    cols = len(board[0]) if rows else 0
    if gravity:
        out = []
        for c in range(cols):
            # if any empty cell in column, it's legal
            if any(board[r][c] == 0 for r in range(rows)):
                out.append(c)
        return out
    else:
        return [(c, r) for r in range(rows) for c in range(cols) if board[r][c] == 0]


def apply_move(
    board: List[List[int]], move: Any, player: int, gravity: bool
) -> Optional[Tuple[int, int]]:
    if gravity or isinstance(move, int):
        c = int(move) if not isinstance(move, tuple) else move
        if isinstance(c, tuple):
            c, r = c
            if board[r][c] == 0:
                board[r][c] = player
                return r, c
            return None
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


def is_full(board: List[List[int]]) -> bool:
    return all(cell != 0 for row in board for cell in row)


### Feature extraction (bucketed to keep Q-table size manageable)


def extract_features(
    board: List[List[int]], player: int, win_len: int, gravity: bool
) -> Tuple[int, ...]:
    rows = len(board)
    cols = len(board[0]) if rows else 0
    opponent = 2 if player == 1 else 1

    # center bias: counts in center column
    center_col = cols // 2 if cols else 0
    center_vals = [board[r][center_col] for r in range(rows)] if cols else []
    center_p = center_vals.count(player)
    center_o = center_vals.count(opponent)
    center_diff = max(-3, min(3, center_p - center_o))

    # threats/opportunities: windows of size win_len with one gap
    def count_windows(p_id: int) -> int:
        cnt = 0
        # Horizontal
        for r in range(rows):
            for c in range(cols - win_len + 1):
                w = [board[r][c + i] for i in range(win_len)]
                if w.count(p_id) == win_len - 1 and w.count(0) == 1:
                    cnt += 1
        # Vertical
        for c in range(cols):
            for r in range(rows - win_len + 1):
                w = [board[r + i][c] for i in range(win_len)]
                if w.count(p_id) == win_len - 1 and w.count(0) == 1:
                    cnt += 1
        # Diagonal down-right
        for r in range(rows - win_len + 1):
            for c in range(cols - win_len + 1):
                w = [board[r + i][c + i] for i in range(win_len)]
                if w.count(p_id) == win_len - 1 and w.count(0) == 1:
                    cnt += 1
        # Diagonal up-right
        for r in range(win_len - 1, rows):
            for c in range(cols - win_len + 1):
                w = [board[r - i][c + i] for i in range(win_len)]
                if w.count(p_id) == win_len - 1 and w.count(0) == 1:
                    cnt += 1
        return cnt

    my_threats = count_windows(player)
    opp_threats = count_windows(opponent)

    moves_cnt = len(legal_moves(board, gravity))
    moves_bucket = 0 if moves_cnt <= 3 else (1 if moves_cnt <= 6 else 2)

    # Basic occupancy ratio bucket
    my_cells = sum(1 for r in range(rows) for c in range(cols) if board[r][c] == player)
    opp_cells = sum(
        1 for r in range(rows) for c in range(cols) if board[r][c] == opponent
    )
    diff_cells = my_cells - opp_cells
    diff_bucket = (
        -2
        if diff_cells < -4
        else (
            2
            if diff_cells > 4
            else (0 if -1 <= diff_cells <= 1 else (1 if diff_cells > 1 else -1))
        )
    )

    return (
        int(player),
        int(gravity),
        int(win_len),
        center_diff,
        min(3, my_threats),
        min(3, opp_threats),
        moves_bucket,
        diff_bucket,
    )


### Immediate tactics (win-now / block-next)


def immediate_tactics(
    board: List[List[int]], player: int, gravity: bool, win_len: int
) -> Optional[Any]:
    # Play winning move now
    for mv in legal_moves(board, gravity):
        nb = deepcopy(board)
        apply_move(nb, mv, player, gravity)
        if check_winner(nb, win_len) == player:
            return mv
    # Block opponent immediate win
    opp = 2 if player == 1 else 1
    for mv in legal_moves(board, gravity):
        nb = deepcopy(board)
        apply_move(nb, mv, opp, gravity)
        if check_winner(nb, win_len) == opp:
            return mv
    return None


### Policy selection


def select_action(
    board: List[List[int]], player: int, params: Dict[str, Any]
) -> Optional[Any]:
    gravity = bool(params.get("gravity", True))
    win_len = int(params.get("win_len", 4))
    epsilon = float(params.get("epsilon", 0.1))

    moves = legal_moves(board, gravity)
    if not moves:
        return None

    # Immediate tactics
    mv = immediate_tactics(board, player, gravity, win_len)
    if mv is not None:
        DECISION_STATS["tactic"] = DECISION_STATS.get("tactic", 0) + 1
        return mv

    features = extract_features(board, player, win_len, gravity)

    # Epsilon-greedy
    if random.random() < epsilon:
        choice = random.choice(moves)
        DECISION_STATS["explore"] = DECISION_STATS.get("explore", 0) + 1
        return choice

    # Exploit: choose action with max Q
    best_a = None
    best_q = -1e9
    for a in moves:
        q = Q_TABLE.get((features, a), 0.0)
        if q > best_q:
            best_q = q
            best_a = a

    if best_a is not None:
        DECISION_STATS["exploit"] = DECISION_STATS.get("exploit", 0) + 1
        return best_a

    # Fallback: no learned preferences yet, use center bias heuristic
    if isinstance(moves[0], int):
        center = (len(board[0]) - 1) / 2.0
        moves.sort(key=lambda x: abs(x - center))
        choice = moves[0]
    else:
        choice = random.choice(moves)
    DECISION_STATS["fallback"] = DECISION_STATS.get("fallback", 0) + 1
    return choice


### Public API expected by the app


def get_move(
    board: List[List[int]], player: int, params: Optional[Dict[str, Any]] = None
) -> Optional[Any]:
    params = params or {}
    enabled = bool(params.get("enabled", False))
    if not enabled:
        # If RL disabled, signal fallback by returning None
        return None
    # gravity & win_len are read inside select_action
    try:
        # Lazy-load Q-table once per process unless disabled
        if not Q_TABLE and not _DISABLE_AUTO_LOAD:
            load_q_table()
        mv = select_action(board, player, params)
        if mv is None:
            # Defensive fallback
            moves = legal_moves(board, bool(params.get("gravity", True)))
            if moves:
                DECISION_STATS["fallback"] = DECISION_STATS.get("fallback", 0) + 1
                return random.choice(moves)
            return None
        return mv
    except Exception:
        # Any unexpected error: fallback to random legal move
        moves = legal_moves(board, bool(params.get("gravity", True)))
        if moves:
            DECISION_STATS["fallback"] = DECISION_STATS.get("fallback", 0) + 1
            return random.choice(moves)
        return None


def get_stats() -> Dict[str, Any]:
    """
    Return lightweight runtime stats useful to display in the app.
    - q_table_size: number of (state, action) entries
    - decisions: counts of tactic/explore/exploit/fallback
    """
    return {
        "q_table_size": len(Q_TABLE),
        "decisions": dict(DECISION_STATS),
    }


def reset_stats(clear_q_table: bool = False) -> None:
    """Reset in-memory decision counters; optionally clear Q-table entries.

    Parameters
    ----------
    clear_q_table : bool
        If True, also empties Q_TABLE (forget learned values in current session).
    """
    DECISION_STATS.update({"tactic": 0, "explore": 0, "exploit": 0, "fallback": 0})
    if clear_q_table:
        Q_TABLE.clear()
        # Delete persisted cache and disable future auto-loads this session
        try:
            path = _cache_path()
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
        global _DISABLE_AUTO_LOAD
        _DISABLE_AUTO_LOAD = True


# Aliases for app compatibility
def policy_move(
    board: List[List[int]], player: int, params: Optional[Dict[str, Any]] = None
) -> Optional[Any]:
    return get_move(board, player, params)


def act(
    board: List[List[int]], player: int, params: Optional[Dict[str, Any]] = None
) -> Optional[Any]:
    return get_move(board, player, params)


def train_self_play(
    episodes: int = 100, params: Optional[Dict[str, Any]] = None
) -> None:
    """
    Minimal self-play trainer to populate Q-table.
    Uses gravity-only episodes on a default 6x7 board.
    Not wired to UI; can be invoked manually.

    Alpha decay: Learning rate decreases over episodes for stable convergence.
    Formula: alpha_t = alpha_initial / (1 + alpha_decay * episode)
    """
    params = params or {
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 0.1,
        "win_len": 4,
        "gravity": True,
    }
    alpha_initial = float(params.get("alpha", 0.1))
    alpha_decay = float(params.get("alpha_decay", 0.0001))  # Decay rate
    gamma = float(params.get("gamma", 0.99))
    epsilon = float(params.get("epsilon", 0.1))
    win_len = int(params.get("win_len", 4))
    gravity = bool(params.get("gravity", True))

    def new_board(rows=6, cols=7):
        return [[0 for _ in range(cols)] for _ in range(rows)]

    for episode in range(episodes):
        # Decay learning rate over time
        alpha = alpha_initial / (1.0 + alpha_decay * episode)
        board = new_board()
        player = 1
        history: List[
            Tuple[Tuple[int, ...], Any, int]
        ] = []  # (features, action, player)
        # play until terminal
        while True:
            moves = legal_moves(board, gravity)
            if not moves:
                break
            # epsilon-greedy over current Q
            features = extract_features(board, player, win_len, gravity)
            if random.random() < epsilon:
                a = random.choice(moves)
            else:
                # choose best known or center-biased
                best_a = None
                best_q = -1e9
                for m in moves:
                    qv = Q_TABLE.get((features, m), 0.0)
                    if qv > best_q:
                        best_q = qv
                        best_a = m
                if best_a is None:
                    if isinstance(moves[0], int):
                        center = (len(board[0]) - 1) / 2.0
                        moves.sort(key=lambda x: abs(x - center))
                        a = moves[0]
                    else:
                        a = random.choice(moves)
                else:
                    a = best_a

            apply_move(board, a, player, gravity)
            history.append((features, a, player))
            w = check_winner(board, win_len)
            if w != 0 or is_full(board):
                # Terminal state reached - backpropagate discounted rewards
                # Rewards per README: +1 win, -1 loss, 0 draw (from player's perspective)

                # Monte Carlo return with proper discounting
                # Start with terminal reward and work backwards
                G = 0.0  # Accumulated discounted return

                # Backpropagate from terminal state to initial state
                for feat, act, pl in reversed(history):
                    # Determine immediate reward for this player's move
                    if w == pl:
                        # This player won
                        reward = 1.0
                    elif w == 0:
                        # Draw
                        reward = 0.0
                    else:
                        # This player lost
                        reward = -1.0

                    # Update return: first occurrence gets immediate reward
                    # Subsequent (earlier) states get discounted future return
                    if G == 0.0:
                        # This is the terminal or most recent state in reverse
                        G = reward
                    else:
                        # Earlier states: discount the future return
                        G = gamma * G

                    # Q-learning update: Q(s,a) <- Q(s,a) + α[G - Q(s,a)]
                    key = (feat, act)
                    q_old = Q_TABLE.get(key, 0.0)
                    Q_TABLE[key] = q_old + alpha * (G - q_old)

                break
            # switch player
            player = 2 if player == 1 else 1

    # persist learned table
    save_q_table()
