//! An implementation of the [MTD(f)](https://en.wikipedia.org/wiki/MTD(f)) algorithm in Rust.
//!
//! The MTD(f) solve two-player zero-sum games.
//! To use this package, you need to implement
//! the trait [`game::GameState`] for your game.

pub mod game;
