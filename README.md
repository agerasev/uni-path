# uni-path

Platform-independent Unix-style path manipulation.

## Rationale

Rust's [`std::path`](https://doc.rust-lang.org/std/path/struct.Path.html) module provides convenient way of path manipulation. It would be nice to use such paths not only with OS file system, but with virtual one (e.g. in-memory fs). Unfortunately, `std::path` is platform-dependent what means that its behavior is different on different platform.

## About

This crate is very similar to `std::path` because its source code was simply copied from `std::path` implementation and only the following points were modified:

+ Remove all platform-dependent conditions and leave only Unix code.
+ Use `str` and `String` instead of `OsStr` and `OsString`.
+ Remove all interactions with OS file system.
