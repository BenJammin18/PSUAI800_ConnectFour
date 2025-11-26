"""
Test script for the trained Q-learning agent.
Validates that the agent can play games and makes reasonable decisions.
"""
import sys
sys.path.insert(0, 'src')

from ai.rl_agent import (
    Q_TABLE, load_q_table, get_move, get_stats, reset_stats,
    legal_moves, apply_move, check_winner, is_full
)
import random


def new_board(rows=6, cols=7):
    return [[0 for _ in range(cols)] for _ in range(rows)]


def print_board(board):
    for row in board:
        print('|' + '|'.join(['X' if c == 1 else 'O' if c == 2 else ' ' for c in row]) + '|')
    print('-' * (len(board[0]) * 2 + 1))


def play_game(ai_player=1, opponent_strategy='random', verbose=False):
    """Play one game. Returns winner (1, 2, or 0 for draw)."""
    board = new_board()
    params = {
        'enabled': True,
        'gravity': True,
        'win_len': 4,
        'epsilon': 0.0  # Pure exploitation for testing
    }
    
    player = 1
    moves_made = 0
    
    while True:
        moves = legal_moves(board, True)
        if not moves:
            return 0  # Draw
        
        if player == ai_player:
            # AI move
            move = get_move(board, player, params)
            if move is None:
                move = random.choice(moves)
        else:
            # Opponent move
            if opponent_strategy == 'random':
                move = random.choice(moves)
            elif opponent_strategy == 'center':
                # Prefer center column
                center = 3
                moves.sort(key=lambda x: abs(x - center))
                move = moves[0]
        
        apply_move(board, move, player, True)
        moves_made += 1
        
        if verbose and moves_made <= 10:
            print(f"Move {moves_made}: Player {player} -> Column {move}")
        
        winner = check_winner(board, 4)
        if winner != 0:
            return winner
        
        if is_full(board):
            return 0
        
        player = 2 if player == 1 else 1


def test_q_table_loading():
    """Test 1: Verify Q-table loads correctly."""
    print("TEST 1: Q-table Loading")
    print("-" * 60)
    
    # Clear and reload
    Q_TABLE.clear()
    load_q_table()
    
    size = len(Q_TABLE)
    print(f"✓ Q-table loaded: {size:,} entries")
    
    if size < 100:
        print("⚠ WARNING: Q-table seems small. Training may not have worked.")
        return False
    
    # Check for reasonable Q-values
    values = list(Q_TABLE.values())
    avg_q = sum(values) / len(values)
    min_q = min(values)
    max_q = max(values)
    
    print(f"✓ Q-value range: [{min_q:.4f}, {max_q:.4f}]")
    print(f"✓ Average Q-value: {avg_q:.4f}")
    
    if not (-1.5 <= min_q <= 1.5) or not (-1.5 <= max_q <= 1.5):
        print("⚠ WARNING: Q-values outside expected range [-1.5, 1.5]")
        return False
    
    print("✓ Q-table looks healthy\n")
    return True


def test_agent_decisions():
    """Test 2: Verify agent makes decisions."""
    print("TEST 2: Agent Decision Making")
    print("-" * 60)
    
    reset_stats()
    board = new_board()
    params = {
        'enabled': True,
        'gravity': True,
        'win_len': 4,
        'epsilon': 0.0
    }
    
    # Make a few moves
    for i in range(5):
        player = (i % 2) + 1
        move = get_move(board, player, params)
        if move is None:
            print(f"✗ Failed to get move on turn {i+1}")
            return False
        apply_move(board, move, player, True)
        print(f"  Move {i+1}: Player {player} chose column {move}")
    
    stats = get_stats()
    print(f"\n✓ Decision stats: {stats['decisions']}")
    
    # Should have used exploit or tactic decisions
    exploit_count = stats['decisions'].get('exploit', 0)
    tactic_count = stats['decisions'].get('tactic', 0)
    fallback_count = stats['decisions'].get('fallback', 0)
    
    if exploit_count + tactic_count == 0:
        print("⚠ WARNING: No exploit/tactic decisions - agent not using Q-table")
        return False
    
    if fallback_count > exploit_count:
        print("⚠ WARNING: Too many fallback decisions")
    
    print("✓ Agent is making learned decisions\n")
    return True


def test_winrate_vs_random():
    """Test 3: Play games against random opponent."""
    print("TEST 3: Performance vs Random Opponent")
    print("-" * 60)
    
    reset_stats()
    games = 100
    wins = 0
    losses = 0
    draws = 0
    
    for i in range(games):
        # Alternate who goes first
        ai_player = 1 if i % 2 == 0 else 2
        winner = play_game(ai_player=ai_player, opponent_strategy='random')
        
        if winner == ai_player:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1
    
    winrate = (wins / games) * 100
    print(f"  Games played: {games}")
    print(f"  Wins: {wins} ({wins/games*100:.1f}%)")
    print(f"  Losses: {losses} ({losses/games*100:.1f}%)")
    print(f"  Draws: {draws} ({draws/games*100:.1f}%)")
    
    stats = get_stats()
    total_decisions = sum(stats['decisions'].values())
    print(f"\n  Decision breakdown over {total_decisions} decisions:")
    for decision_type, count in stats['decisions'].items():
        pct = (count / total_decisions * 100) if total_decisions > 0 else 0
        print(f"    {decision_type}: {count} ({pct:.1f}%)")
    
    # Agent should beat random opponent at least 60% of the time
    if winrate < 60:
        print(f"\n⚠ WARNING: Win rate {winrate:.1f}% is below expected 60%")
        print("  Agent may need more training or has issues")
        return False
    
    print(f"\n✓ Win rate {winrate:.1f}% - Agent performs well!\n")
    return True


def test_tactical_awareness():
    """Test 4: Verify agent blocks/wins when obvious."""
    print("TEST 4: Tactical Awareness")
    print("-" * 60)
    
    params = {
        'enabled': True,
        'gravity': True,
        'win_len': 4,
        'epsilon': 0.0
    }
    
    # Test 1: Win in one move
    board = new_board()
    # Set up: X X X _ (horizontal win available in column 3)
    board[5][0] = 1
    board[5][1] = 1
    board[5][2] = 1
    
    move = get_move(board, 1, params)
    if move != 3:
        print(f"✗ Failed to take winning move (chose {move} instead of 3)")
        return False
    print("✓ Takes winning move when available")
    
    # Test 2: Block opponent win
    board = new_board()
    # Set up: O O O _ (opponent about to win in column 3)
    board[5][0] = 2
    board[5][1] = 2
    board[5][2] = 2
    
    move = get_move(board, 1, params)
    if move != 3:
        print(f"✗ Failed to block opponent win (chose {move} instead of 3)")
        return False
    print("✓ Blocks opponent winning move")
    
    print("✓ Tactical awareness working correctly\n")
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("TRAINED Q-LEARNING AGENT TEST SUITE")
    print("=" * 60)
    print()
    
    tests = [
        test_q_table_loading,
        test_agent_decisions,
        test_tactical_awareness,
        test_winrate_vs_random,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}\n")
            results.append(False)
    
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if all(results):
        print("\n✓ ALL TESTS PASSED - Agent is ready to use!")
        sys.exit(0)
    else:
        print("\n⚠ Some tests failed - review output above")
        sys.exit(1)
