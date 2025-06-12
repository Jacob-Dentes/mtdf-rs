# MTDF
A (bad) Rust implementation of the [MTD(f)](https://en.wikipedia.org/wiki/MTD(f)) algorithm for two-player, zero-sum games.

## How to use
You need to implement `GameState` for your game, see [the docs](https://docs.rs/mtdf) for the required methods.

Check out the beginner examples at [examples/tictactoe.rs](https://github.com/Jacob-Dentes/mtdf-rs/blob/main/examples/tictactoe.rs) or [examples/wythoff.rs](https://github.com/Jacob-Dentes/mtdf-rs/blob/main/examples/wythoff.rs) for simple usage examples.

##  FAQ
- Does it work?
  - It probably works on one core, maybe not on multiple. Bug reports are welcome!
- Is it any good?
  - I doubt it.
