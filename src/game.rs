use rustc_hash::FxHashMap;
use std::cmp::PartialOrd;
use std::hash::Hash;
use std::sync::RwLock;

type T = f32;

/// The value returned by an evaluation function.
///
/// Every evaluation is either [`Exact`](Evaluation::Exact) or [`Heuristic`](Evaluation::Heuristic).
///
/// - [`Exact`](Evaluation::Exact) values represent the value of a finished game.
///
/// - [`Heuristic`](Evaluation::Heuristic) values represent the estimated value of a game still in progress.
pub enum Evaluation {
    Exact(T),
    Heuristic(T),
}

impl Evaluation {
    /// Gets the value of the contents, ignoring the variant.
    pub fn val(&self) -> &T {
        match self {
            Self::Exact(t) | Self::Heuristic(t) => t,
        }
    }

    /// Like `Evaluation::val`, but consumes and gives ownership.
    pub fn take(self) -> T {
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
    pub fn flip(&self) -> Self {
        match self {
            Self::Maximizing => Self::Minimizing,
            Self::Minimizing => Self::Maximizing,
        }
    }

    /// This is '1' for the maximizer and `-1` for the minimizer.
    pub fn sign(&self) -> T {
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
/// and [`GameState::turn`]. You can optionally implement [`GameState::order_moves`] or
/// [`GameState::initial_f`] to speed up the algorithm.
pub trait GameState: Sized + Clone + Hash + Eq {
    /// Returns the signed value of the game.
    /// If the game is over, it should be ['Exact']
    /// containing the value of the game.
    ///
    /// If the game is still in progress, it should
    /// be ['Heuristic'] with a guess at the game's
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
    fn order_moves(&self, moves: Vec<(Self, Option<T>)>) -> Vec<Self> {
        let mut moves: Vec<(T, Self)> = moves
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
    fn initial_f(&self) -> T {
        0.0
    }
}

enum TranspositionEntry {
    Exact,
    Lowerbound,
    Upperbound,
}

pub struct MTDBot<G>
where
    G: GameState,
{
    table: Option<RwLock<FxHashMap<G, (TranspositionEntry, G, T, usize)>>>,
}

impl<G: GameState + Hash + Clone + Eq> MTDBot<G> {
    pub fn new() -> Self {
        let table = FxHashMap::default();
        Self {
            table: Some(RwLock::new(table)),
        }
    }

    pub fn new_memoryless() -> Self {
        Self { table: None }
    }

    pub fn solve(&self, game: &G, depth: usize) -> G {
        // TODO: add MTDF
        self.negamax(game, depth, T::NEG_INFINITY, T::INFINITY).1
    }

    fn negamax(
        &self,
        game: &G,
        depth: usize,
        mut alpha: T,
        beta: T,
        // locked_transposition: Option<&RwLock<FxHashMap<G, (TranspositionEntry, G, T, usize)>>>,
    ) -> (T, G) {
        let alpha_orig = alpha.clone();
        if let Some(transposition) = self.table {
            if let Ok(table) = transposition.read() {
                if let Some((entry, g, t, d)) = table.get(game) {
                    if d >= &depth {
                        match entry {
                            TranspositionEntry::Exact => return (t.clone(), g.clone()),
                            TranspositionEntry::Lowerbound => alpha = alpha.max(*t),
                            TranspositionEntry::Upperbound => alpha = alpha.min(*t),
                        }

                        if alpha >= beta {
                            match entry {
                                TranspositionEntry::Exact
                                | TranspositionEntry::Lowerbound
                                | TranspositionEntry::Upperbound => return (t.clone(), g.clone()),
                            }
                        }
                    }
                }
            }
        }

        let evaluation: Evaluation = game.evaluate();

        if let Evaluation::Exact(v) = evaluation {
            let res: T = game.turn().sign() * v;
            return (res, game.clone());
        }

        if depth == 0 {
            let res: T = game.turn().sign() * evaluation.take();
            return (res, game.clone());
        }

        let children = game.moves();
        let children: Vec<(G, Option<T>)> = {
            if let Some(transposition) = self.table {
                if let Ok(table) = transposition.read() {
                    children
                        .into_iter()
                        .map(|g| (table.get(&g), g))
                        .map(|(v, g)| (g, v.map(|s| s.2.clone())))
                        .collect()
                } else {
                    children.into_iter().map(|g| (g, None)).collect()
                }
            } else {
                children.into_iter().map(|g| (g, None)).collect()
            }
        };
        let mut children = game.order_moves(children);

        assert!(
            !children.is_empty(),
            "Evaluation returned Heuristic on a game with no legal moves."
        );
        let res1 = self.negamax(
            &children.remove(0),
            depth.saturating_sub(1),
            -beta.clone(),
            -alpha.clone(),
        );

        let (mut value, mut best_child) = (-res1.0, res1.1);
        alpha = alpha.max(value);

        for child in children {
            if alpha >= beta {
                break;
            }
            let nega_res = self.negamax(
                &child,
                depth.saturating_sub(1),
                -beta.clone(),
                -alpha.clone(),
            );
            let nega_res = (-nega_res.0, nega_res.1);
            if nega_res.0 > value {
                value = nega_res.0;
                best_child = child;
            }
            alpha = alpha.max(value);
        }

        if let Some(transposition) = locked_transposition {
            {
                if let Ok(mut table) = transposition.write() {
                    let entry_type = {
                        if value <= alpha_orig {
                            TranspositionEntry::Upperbound
                        } else if value >= beta {
                            TranspositionEntry::Lowerbound
                        } else {
                            TranspositionEntry::Exact
                        }
                    };
                    table.insert(
                        game.clone(),
                        (entry_type, best_child.clone(), value.clone(), depth),
                    );
                }
            }
        }

        (value, best_child)
    }
}
