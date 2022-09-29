# MultiObjectiveExploration

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Maintainers](#maintainers)
- [License](#license)

## Background
1) evaluate various prediction models for DSE of CPU.
2) a hypervolume-improvement-based multi-objective optimization method and a uniformity-aware selection algorithm to select design points.
Further, AdaBoost is first introduced to GBRT model to improve the prediction accuracy.

## Install

This project uses torch, skopt, math, and several basic python packages. Go check them out if you don't have them locally installed.

## Usage

This is only a documentation package.

```sh
# run main() in main.python
$ python main.python
```

Major running configurations are defined in config.py
Performance metric dataset is listed in data_all_simpoint/ . For now, it only includes a demo sample record in 500.1-refrate-1.txt

## Maintainers
[@multiobjectiveDSE](https://github.com/multiobjectiveDSE).

## License
[MIT](license)
