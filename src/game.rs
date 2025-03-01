//! Holds the base class for a game that MTD(f) can solve.
//! To use the crate, implement [`GameState`] for your game.

use std::cmp::PartialOrd;
use std::hash::Hash;

/// The value returned by an evaluation function.
///
/// Every evaluation is either [`Exact`](Evaluation::Exact) or [`Heuristic`](Evaluation::Heuristic).
///
/// - [`Exact`](Evaluation::Exact) values represent the value of a finished game.
///
/// - [`Heuristic`](Evaluation::Heuristic) values represent the estimated value of a game still in progress.
pub enum Evaluation {
    Exact(f32),
    Heuristic(f32),
}

impl Evaluation {
    /// Gets the value of the contents, ignoring the variant.
    pub fn val(&self) -> &f32 {
        match self {
            Self::Exact(t) | Self::Heuristic(t) => t,
        }
    }

    /// Like `Evaluation::val`, but consumes and gives ownership.
    pub fn take(self) -> f32 {
        match self {
            Self::Exact(t) | Self::Heuristic(t) => t,
        }
    }
}

/// The two players of a zero-sum game.
pub enum Player {
    Maximizing,
    Minimizing,
}

impl Player {
    /// The opposite player.
    pub const fn flip(&self) -> Self {
        match self {
            Self::Maximizing => Self::Minimizing,
            Self::Minimizing => Self::Maximizing,
        }
    }

    /// This is '1' for the maximizer and `-1` for the minimizer.
    pub const fn sign(&self) -> f32 {
        match self {
            Self::Maximizing => 1.0,
            Self::Minimizing => -1.0,
        }
    }
}

/// A two-player, zero-sum game that can be solved using MTDF
///
/// The trait is generic over T, the numeric value used to score the game.
/// Most commonly, games should implement `GameState<f32>` or `GameState<i32>`,
/// but any signed numeric type will work.
///
/// To implement this trait, you must implement [`GameState::evaluate`], [`GameState::moves`],
/// and [`GameState::turn`]. You can optionally implement [`GameState::order_moves`],
/// [`GameState::initial_f`], [`GameState::mtdf_window`], or [`GameState::abdad_defer_depth`]
///  to speed up the algorithm.
pub trait GameState: Sized + Clone + Hash + Eq {
    /// Returns the signed value of the game.
    /// If the game is over, it should be [`Evaluation::Exact`]
    /// containing the value of the game.
    ///
    /// If the game is still in progress, it should
    /// be [`Evaluation::Heuristic`] with a guess at the game's
    /// value.
    fn evaluate(&self) -> Evaluation;

    /// Returns the children of the game state.
    /// These are the game states that would result from every legal move.
    fn moves(&self) -> impl IntoIterator<Item = Self>;

    /// Returns the [`Player`] whose turn it is.
    fn turn(&self) -> Player;

    /// Re-orders games from most to least promising. If good moves are
    /// explored earlier, the algorithm can prune more nodes.
    ///
    /// The parameter `moves` is a vector of `(GameState, Option<T>)`. The first
    /// entry is a move, the second entry is `Some(T)` if there is a transposition
    /// table entry for this node, and `None` otherwise.
    ///
    /// By default, the move order will be determined by the transposition table
    /// entry first, then falls back on the evaluation if no entry exists.
    fn order_moves(&self, moves: Vec<(Self, Option<f32>)>) -> Vec<Self> {
        let mut moves: Vec<(f32, Self)> = moves
            .into_iter()
            .map(|(m, v)| {
                let s = self.turn().flip().sign();
                (s * v.unwrap_or(m.evaluate().take()), m)
            })
            .collect();
        moves.sort_unstable_by(|(v1, _), (v2, _)| {
            v1.partial_cmp(v2).unwrap_or(std::cmp::Ordering::Equal)
        });
        moves.into_iter().map(|(_, g)| g).collect()
    }

    /// Initial guess for the MTD(f) algorithm. The closer
    /// this is to the actual evaluation, the faster it will run.
    ///
    /// Avoid doing expensive computations here. Typically you
    /// return a constant representing how the game would end
    /// between perfect players.
    fn initial_f(&self) -> f32 {
        0.0
    }

    /// Window size for the MTD(f) algorithm. This should be a
    /// float strictly greater than zero. The magnitude should
    /// be related to the difference between the largest possible
    /// evaluation and the smallest possible evaluation.
    ///
    /// Engines tend to make this small on the first turn, then
    /// increase it on future turns.
    fn mtdf_window(&self) -> f32 {
        1.0
    }

    /// A tolerance parameter for accuracy. It should be a float
    /// strictly greater than zero. Return a smaller float if
    /// the algorithm has numeric issues.
    fn epsilon(&self) -> f32 {
        1e-10
    }

    /// The depth where the engine will start searching using
    /// multiple cores.
    fn abdad_defer_depth(&self) -> usize {
        3
    }
}
