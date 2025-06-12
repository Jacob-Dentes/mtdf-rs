extern crate mtdf;

use mtdf::game::{Evaluation, GameState, Player};
use mtdf::mtdf::MTDBot;

// see https://en.wikipedia.org/wiki/Wythoff%27s_game
// This game should NOT be solved with MTDf, but it can be!

const P1: Player = Player::Maximizing;
const P2: Player = Player::Minimizing;

enum MovePile {
    Left,
    Right,
    Both,
}

type Move = (MovePile, usize);

type Piles = (usize, usize);
#[derive(Hash, Clone, PartialEq, Eq, Debug)]
struct WythoffGame {
    piles: Piles,
    turn: usize,
}

impl WythoffGame {
    pub fn new(n: usize, m: usize) -> Self {
        WythoffGame {
            piles: (n, m),
            turn: 0,
        }
    }

    #[inline(always)]
    pub fn do_move(&self, m: Move) -> Self {
        assert!(m.1 >= 1);
        match m.0 {
            MovePile::Both => {
                assert!(m.1 <= self.piles.0 && m.1 <= self.piles.1);
                WythoffGame {
                    piles: (self.piles.0 - m.1, self.piles.1 - m.1),
                    turn: self.turn + 1,
                }
            }
            MovePile::Left => {
                assert!(m.1 <= self.piles.0);
                WythoffGame {
                    piles: (self.piles.0 - m.1, self.piles.1),
                    turn: self.turn + 1,
                }
            }
            MovePile::Right => {
                assert!(m.1 <= self.piles.1);
                WythoffGame {
                    piles: (self.piles.0, self.piles.1 - m.1),
                    turn: self.turn + 1,
                }
            }
        }
    }
}

// searches for a value of k such that floor(phi * k) == n,
// returns Some(None) if none exists
// returns None if we can't tell because of imprecision
fn find_k(n: usize) -> Option<Option<usize>> {
    // f64 can exactly represent integers up to 2^53
    const MAX_EXACT: usize = 1 << 53;
    if n >= MAX_EXACT {
        // n is too large for reliable conversion to f64
        return None;
    }

    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;

    let lower = (n as f64) / phi;
    let k = lower.ceil() as usize;

    let prod = phi * (k as f64);

    if prod < (n + 1) as f64 {
        if prod.floor() as usize == n {
            Some(Some(k))
        } else {
            Some(None)
        }
    } else {
        Some(None)
    }
}

impl GameState for WythoffGame {
    fn evaluate(&self) -> Evaluation {
        // if the game is over then the last player won
        if self.piles.0 == 0 && self.piles.1 == 0 {
            return Evaluation::Exact(self.turn().flip().sign());
        }

        let (pile1, pile2) = (
            self.piles.0.min(self.piles.1),
            self.piles.0.max(self.piles.1),
        );

        let res = find_k(pile1);

        if let Some(Some(k)) = res {
            if pile1 == pile2 + k {
                return Evaluation::Heuristic(self.turn().flip().sign());
            } else {
                return Evaluation::Heuristic(self.turn().sign());
            }
        }

        if let Some(None) = res {
            return Evaluation::Heuristic(self.turn().sign());
        }

        Evaluation::Heuristic(0.0)
    }

    fn turn(&self) -> Player {
        if self.turn % 2 == 0 {
            P1
        } else {
            P2
        }
    }

    fn moves(&self) -> impl IntoIterator<Item = Self> {
        let mut move_vec: Vec<Self> =
            Vec::with_capacity(self.piles.0 + self.piles.1 + self.piles.0.min(self.piles.1));
        for i in 1..=self.piles.0 {
            move_vec.push(self.do_move((MovePile::Left, i)));
        }
        for i in 1..=self.piles.1 {
            move_vec.push(self.do_move((MovePile::Right, i)));
        }
        for i in 1..=(self.piles.0.min(self.piles.1)) {
            move_vec.push(self.do_move((MovePile::Both, i)));
        }
        move_vec
    }
}

fn main() {
    println!(
        "Solving game with 7 stones in each pile (Expecting player 1 to win by taking all stones)."
    );
    let root = WythoffGame::new(7, 7);

    let bot = MTDBot::<WythoffGame>::new();

    let now = std::time::Instant::now();
    let res = bot.mtdf(&root, &10, None);

    println!("Solved root in {} microseconds.", now.elapsed().as_micros());
    println!("MTD(f) result: {:?}.\n", res);

    println!("Solving game with 1 stones in first pile, 2 in second. Expecting player 2 to win.");
    let root = WythoffGame::new(1, 2);

    let now = std::time::Instant::now();
    let res = bot.mtdf(&root, &10, None);

    println!("Solved game in {} microseconds.", now.elapsed().as_micros());
    println!("MTD(f) result: {:?}.\n", res);

    println!("Solving game with 50 stones in first pile, 45 in second. Expecting player 1 to win.");
    let root = WythoffGame::new(50, 45);

    let now = std::time::Instant::now();
    let res = bot.mtdf(&root, &10, None);

    println!("Solved game in {} microseconds.", now.elapsed().as_micros());
    println!("MTD(f) result: {:?}.\n", res);
}
