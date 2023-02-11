
# nnt pytorch tools

nnt (**n**eural **n**et **t**ools) provides a set of common utilities for pytorch model training. nnt helps keep training codebases clean and concise by abstracting away non problem specific code while remaining simple to use. The goal is to provide just enough utility so you don't feel like you have to surrender your train loop to a heavy, opinionated framework (e.g. Lightning).

A sampling of some of the tools provided:

 * Checkpoint saving, loading, and resuming
 * Metric tracking and logging, with tensorboard integration
 * Platform-agnostic file I/O (local disk, S3 and variants, etc)
 * Optimization helpers like param group reduction, adaptive gradient clipping, etc
 * Catch-all RNG seeder
 * A simple reusable component registry


Improvements, suggestions, and bug reports are welcome.


# License

MIT
