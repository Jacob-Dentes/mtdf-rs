extern crate mtdf;

use mtdf::game::{Evaluation, GameState, Player};
use mtdf::mtdf::MTDBot;

// X is the maximizing player, O is the minimizing player
const X: Player = Player::Maximizing;
const O: Player = Player::Minimizing;

// An enum representing what is in a square of the board
#[derive(Hash, Clone, Eq, PartialEq, Copy, Debug)]
enum Square {
    X,
    O,
    Empty,
}

type Board = [Square; 9];

#[derive(Hash, Clone, Eq, PartialEq, Debug)]
struct TicTacToe {
    board: Board,
    x_turn: bool,
}

impl TicTacToe {
    pub fn new() -> Self {
        TicTacToe {
            board: [Square::Empty; 9],
            x_turn: true,
        }
    }

    pub fn do_move(&self, loc: usize) -> Self {
        let mut new_board = self.board.clone();
        new_board[loc] = if self.x_turn { Square::X } else { Square::O };
        TicTacToe {
            board: new_board,
            x_turn: !self.x_turn,
        }
    }

    // returns Some(X), Some(O), and Some(Empty) on X win, O win, and draw, respectively
    // returns None if the game is not over
    pub fn check_over(&self) -> Option<Square> {
        let check_line = |board: &Board, start: usize, step: usize| {
            if board[start] == board[start + step] && board[start] == board[start + 2 * step] {
                board[start].clone()
            } else {
                Square::Empty
            }
        };

        let three_in_a_row = [
            (0, 1), // horizontal wins
            (3, 1),
            (6, 1),
            (0, 3), // vertical wins
            (1, 3),
            (2, 3),
            (0, 4), // diagonal wins
            (2, 2),
        ];

        for (start, step) in three_in_a_row {
            let res = check_line(&self.board, start, step);
            match res {
                Square::X => return Some(Square::X),
                Square::O => return Some(Square::O),
                Square::Empty => (),
            }
        }

        if self.board.iter().all(|s| s != &Square::Empty) {
            return Some(Square::Empty);
        }

        None
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn board_string(&self) -> String {
        let mut board_str = "".to_string();
        let board: Vec<char> = self
            .board
            .iter()
            .map(|c| match c {
                Square::X => 'X',
                Square::O => 'O',
                Square::Empty => ' ',
            })
            .collect();
        board_str += format!("\n{} | {} | {}\n", board[0], board[1], board[2]).as_str();
        let division = "---------\n";
        board_str += division;
        board_str += format!("{} | {} | {}\n", board[3], board[4], board[5]).as_str();
        board_str += division;
        board_str += format!("{} | {} | {}\n", board[6], board[7], board[8]).as_str();

        board_str
    }
}

impl GameState for TicTacToe {
    fn evaluate(&self) -> Evaluation {
        match self.check_over() {
            Some(Square::X) => return Evaluation::Exact(2.0 * X.sign()),
            Some(Square::O) => return Evaluation::Exact(2.0 * O.sign()),
            Some(Square::Empty) => return Evaluation::Exact(0.0),
            None => (),
        }

        // TODO: We could make a more sophisticated heuristic to handle
        // incomplete games, but TicTacToe will be able to search all the
        // way to depth anyway.
        Evaluation::Heuristic(0.0)
    }

    fn moves(&self) -> impl IntoIterator<Item = Self> {
        let mut res_vec: Vec<TicTacToe> = Vec::with_capacity(9);

        for (i, square) in self.board.iter().enumerate() {
            if square == &Square::Empty {
                res_vec.push(self.do_move(i));
            }
        }

        res_vec
    }

    fn turn(&self) -> Player {
        if self.x_turn {
            X
        } else {
            O
        }
    }
}

fn main() {
    // empty board
    let root = TicTacToe::new();

    println!("Simulating a bot at the root.");
    let bot = MTDBot::<TicTacToe>::new();
    let now = std::time::Instant::now();
    let res = bot.mtdf(&root, &10);
    println!("Solved root in {} microseconds", now.elapsed().as_micros());

    // TicTacToe can be solved in a draw, so our
    // evaluation should be about 0.0
    assert!(res.0 <= root.epsilon());

    println!("Simulating a bot with an X in the top left.");
    // board with an X in the top left
    let x_top_left = root.do_move(0);
    let res = bot.mtdf(&x_top_left, &10);

    // An optimal O player must play middle
    // in response to corner
    assert!(res.1.board()[4] == Square::O);

    println!("Simulating a bot with an X in the top left, O in the middle.");
    // board with an X in top left, O in middle right
    let x_o_board = x_top_left.do_move(5);
    let res = bot.mtdf(&x_o_board, &10);

    // O played suboptimally, so X should win
    assert!((res.0 - 2.0).abs() <= root.epsilon());

    // simulate two bots playing
    let mut game = root;
    println!("Simulating a bot TicTacToe game.\n{}", game.board_string());
    while game.check_over().is_none() {
        game = bot.mtdf(&game, &10).1;
        println!("{}", game.board_string());
    }
}
