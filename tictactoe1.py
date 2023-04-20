import numpy as np
import random


class TicTacToe:
    def __init__(self, board=None):
        self.board = board if board is not None else np.zeros(
            (3, 3), dtype=int)

    def is_valid_move(self, row, col):
        return self.board[row, col] == 0

    def make_move(self, row, col, player):
        if self.is_valid_move(row, col):
            self.board[row, col] = player
            return True
        return False

    def get_valid_moves(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def check_winner(self):
        for i in range(3):
            if self.board[i, :].sum() in [3, -3] or self.board[:, i].sum() in [3, -3]:
                return True
        if np.trace(self.board) in [3, -3] or np.trace(np.fliplr(self.board)) in [3, -3]:
            return True
        return False

    def is_draw(self):
        return len(self.get_valid_moves()) == 0

    def is_terminal(self):
        return self.check_winner() or self.is_draw()

    def play_random_opponent(self):
        valid_moves = self.get_valid_moves()
        if valid_moves:
            move = random.choice(valid_moves)
            self.make_move(move[0], move[1], -1)

    def copy(self):
        return TicTacToe(self.board.copy())

    def __str__(self):
        return str(self.board)


def state_to_str(state):
    return ''.join(map(str, state.flatten()))


def str_to_state(state_str):
    return np.array(list(map(int, state_str)), dtype=int).reshape((3, 3))


def q_learning(alpha=0.5, gamma=1, num_episodes=100000, epsilon=0.1):
    q_values = {}
    for _ in range(num_episodes):
        game = TicTacToe()
        while not game.is_terminal():
            state = state_to_str(game.board)
            if state not in q_values:
                q_values[state] = np.zeros(9)

            if random.random() < epsilon:
                row, col = random.choice(game.get_valid_moves())
            else:
                action = np.argmax(q_values[state])
                row, col = action // 3, action % 3

            next_game = game.copy()
            next_game.make_move(row, col, 1)
            next_game.play_random_opponent()

            if not next_game.is_terminal():
                next_state = state_to_str(next_game.board)
                if next_state not in q_values:
                    q_values[next_state] = np.zeros(9)

                reward = 0
                max_q_next = np.max(q_values[next_state])
            else:
                if next_game.check_winner():
                    reward = 1
                elif next_game.is_draw():
                    reward = 0
                else:
                    reward = -1
                max_q_next = 0

            action = row * 3 + col
            q_values[state][action] += alpha * \
                (reward + gamma * max_q_next - q_values[state][action])
            game = next_game

    return q_values


def optimal_policy(q_values, state):
    if state_to_str(state) not in q_values:
        return None
    action = np.argmax(q_values[state_to_str(state)])
    return action


if __name__ == "__main__":
    q_values = q_learning(alpha=0.5, gamma=1, num_episodes=100000, epsilon=0.1)

    # Test the optimal policy for a specific state
    test_state = np.array([
        [1, -1, 0],
        [0,  1, 0],
        [0,  0, -1]
    ])

    print("State:")
    print(test_state)

    action = optimal_policy(q_values, test_state)
    if action is not None:
        row, col = action // 3, action % 3
        print(f"Optimal move for cross-player (X): ({row}, {col})")
    else:
        print("No optimal move found.")
