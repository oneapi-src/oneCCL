# Contributing guidelines

If you have improvements to the oneCCL code, please send us your pull
requests! To get started with pull requests, see GitHub
[howto](https://help.github.com/en/articles/about-pull-requests).

The current guidelines are work in progress.

## Pull request checklist

TBD

### RFC pull requests

It is strongly advised to open an RFC (request for comments) pull request when contributing new
primitives. Please provide the following details:

* The definition of the operation as an oneCCL primitive. It should also include interface and semantics. It is OK to have sketches for the interface, but the semantics should be fairly well-defined.

* Use case, including the model and parallelism scenario.

## Code contribution guidelines

The code must be:

* *Tested*: oneCCL uses gtests for lightweight functional testing.

* *Documented*: oneCCL uses Doxygen for inline comments in public header
  files that are used to build reference manual and markdown (also processed by
  Doxygen) for user guide.

* *Portable*: oneCCL supports CPU and GPU
  architectures, as well as different compilers and run-times. The new code should be complaint
  with the [System Requirements](README.md#system-requirements).

## Coding style

The general principle is to follow the style of existing / surrounding code.

TBD

## Unit tests

Be sure to extend the existing tests when fixing an issue.
