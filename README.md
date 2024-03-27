# overlappogram

[![CI](https://github.com/jmbhughes/overlappogram/actions/workflows/CI.yml/badge.svg)](https://github.com/jmbhughes/overlappogram/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/jmbhughes/overlappogram/graph/badge.svg?token=u1qQvzybz4)](https://codecov.io/gh/jmbhughes/overlappogram)
[![DOI](https://zenodo.org/badge/759222503.svg)](https://zenodo.org/doi/10.5281/zenodo.10869534)

![overlappogram example](overlappogram.png)

🚧🚧🚧 **UNDER DEVELOPMENT** 🚧🚧🚧

**The package is still being developed. Expect breaking changes.**

Overlappogram is a Python package for inverting overlappogram observations of the Sun,
for examples MaGIXS, CubIXSS, or ECCCO observations.


## Install

`pip install overlappogram` or clone the repository and install manually

## How to Use

`overlappogram` comes with an executable main that you can run:

`unfold ./path/to/config.toml`

The `config.toml` file should be structured similar to the [example_config.toml](example_config.toml).
We provide more description of the config file [in the documentation](https://eccco-mission.github.io/overlappogram/configuration.html). 

## Getting Help

Please [open an issue](https://github.com/jmbhughes/overlappogram/issues/new)
or [create a discussion](https://github.com/jmbhughes/overlappogram/discussions/new/choose).
We prefer this over email so that other users can benefit from your questions.

## Cite

[Please cite using the Zenodo DOI.](https://zenodo.org/records/10869577)

## Contributors

The initial version of `overlappogram` was written by Dyana Beabout.
This version is written by [J. Marcus Hughes](https://github.com/jmbhughes).
