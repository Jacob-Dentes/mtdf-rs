//! This module's primary struct is [`MTDBot`],
//! which runs the MTD(f) algorithm on a game.
use std::hash::Hash;
use std::sync::RwLock;

// use std::marker::{Send, Sync};
// use rayon::prelude::*;

use rustc_hash::FxHashMap;

use crate::game::*;

#[derive(Debug)]
enum TranspositionEntry {
    Exact,
    Lowerbound,
    Upperbound,
}

/// A bot that runs the MTD algorithm.
/// Bots constructed with [`MTDBot::new`] will have a cache
/// that significantly improves performance, at the cost of
/// memory overhead. Bots constructed with [`MTDBot::new_memoryless`]
/// will have no cache, and MTD(f) will fall back to Minimax.
///
/// The struct is parameterized on G, the type of a game that
/// implements [`crate::game::GameState`]. You must implement
/// this trait to run the MTDF algorithm.
pub struct MTDBot<G>
where
    G: GameState,
{
    table: Option<RwLock<FxHashMap<G, (TranspositionEntry, G, f32, usize)>>>,
}

impl<G: GameState + Hash + Clone + Eq> MTDBot<G> {
    /// Create a MTDBot with an underlying cache
    pub fn new() -> Self {
        let table = FxHashMap::default();
        Self {
            table: Some(RwLock::new(table)),
        }
    }

    /// Create a MTDBot with no underlying cache. Calls to
    /// MTD(f) with a memoryless bot will fall back to Minimax.
    /// It is strongly recommended to use [`MTDBot::new`]
    /// instead.
    pub fn new_memoryless() -> Self {
        Self { table: None }
    }

    fn negamax(&self, game: &G, depth: usize, mut alpha: f32, beta: f32) -> (f32, G) {
        let alpha_orig = alpha.clone();
        if let Some(transposition) = &self.table {
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
                                TranspositionEntry::Exact => return (t.clone(), g.clone()),
                                TranspositionEntry::Lowerbound | TranspositionEntry::Upperbound => {
                                    return (t.clone(), g.clone())
                                }
                            }
                        }
                    }
                }
            }
        }

        let evaluation: Evaluation = game.evaluate();

        if let Evaluation::Exact(v) = evaluation {
            let res: f32 = game.turn().sign() * v;
            return (res, game.clone());
        }

        if depth == 0 {
            let res: f32 = game.turn().sign() * evaluation.take();
            return (res, game.clone());
        }

        let children = game.moves();
        let children: Vec<(G, Option<f32>)> = {
            if let Some(transposition) = &self.table {
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

        let child1 = children.remove(0);
        let res1 = self.negamax(
            &child1,
            depth.saturating_sub(1),
            -beta.clone(),
            -alpha.clone(),
        );

        let (mut value, mut best_child) = (-res1.0, child1);
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
            let nega_res_val = -nega_res.0;
            if nega_res_val > value {
                value = nega_res_val;
                best_child = child;
            }
            alpha = alpha.max(value);
        }

        let entry_type = {
            if value <= alpha_orig {
                TranspositionEntry::Upperbound
            } else if value >= beta {
                TranspositionEntry::Lowerbound
            } else {
                TranspositionEntry::Exact
            }
        };

        if let Some(transposition) = &self.table {
            if let Ok(mut table) = transposition.write() {
                table.insert(
                    game.clone(),
                    (entry_type, best_child.clone(), value.clone(), depth),
                );
            }
        }

        (value, best_child)
    }

    fn mtdf_aux(&self, root: &G, f: &f32, d: &usize) -> (f32, G) {
        // let ncpus = num_cpus::get();
        let inner = |mut g: f32, mut ub: f32, mut lb: f32| {
            let beta = {
                if (g - lb).abs() < root.epsilon() {
                    g + root.mtdf_window()
                } else {
                    g
                }
            };

            // let window = root.mtdf_window();
            // let r = RwLock::new(root);
            // let results: Vec<(f32, G)> = (0..ncpus)
            //     .into_par_iter()
            //     .map(|_| self.negamax(&r.read().unwrap(), d.clone(), beta - window, beta.clone()))
            //     .collect();
            // let nega_res = results.into_iter().nth(0).unwrap();
            let nega_res = self.negamax(root, d.clone(), beta - root.mtdf_window(), beta);
            let best_move = nega_res.1;

            g = nega_res.0;
            if g < beta {
                ub = g
            } else {
                lb = g
            }

            (g, ub, lb, best_move)
        };

        let (mut g, mut ub, mut lb, mut best_move) =
            inner(f.clone(), f32::INFINITY, f32::NEG_INFINITY);

        while lb < ub - root.epsilon() {
            (g, ub, lb, best_move) = inner(g, ub, lb)
        }

        (g, best_move)
    }

    /// Runs the MTD(f) algorithm at GameState `root` to `depth`.
    ///
    /// Returns a tuple where the first element is the signed value
    /// of the game and the second element is the best move.
    ///
    /// The parameter `root` is the current gamestate, the parameter
    /// `depth` is the number of moves to search ahead.
    pub fn mtdf(&self, root: &G, depth: &usize) -> (f32, G) {
        // TODO: parallelize the algorithm
        if self.table.is_none() {
            let (n1, n2) = self.negamax(root, *depth, f32::NEG_INFINITY, f32::INFINITY);
            return (n1, n2);
        }
        let inner = |f: f32, d: &usize| self.mtdf_aux(root, &f, d);

        let (mut f, mut best) = inner(root.initial_f(), &1);

        for d in 2..=(*depth) {
            (f, best) = inner(f, &d);
        }

        (root.turn().sign() * f, best)
    }
}
