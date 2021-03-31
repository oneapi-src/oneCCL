# Contributing guidelines

We welcome community contributions to oneCCL. You can:

- Submit your changes directly with a [pull request](https://github.com/oneapi-src/oneCCL/pulls).
- Log a bug or feedback with an [issue](https://github.com/oneapi-src/oneCCL/issues).

Refer to our guidelines on [pull requests](#pull-requests) and [isssues](#issues) before you proceed.

## Issues

Use [GitHub issues]((https://github.com/oneapi-src/oneCCL/issues)) to:
- report an issue
- provide feedback
- make a feature request

**Note**: To report a vulnerability, refer to [Intel vulnerability reporting policy](https://www.intel.com/content/www/us/en/security-center/default.html).

## Pull requests

Before you submit a pull request, make sure that:

- You follow our [code contribution guidelines](#code-contribution-guidelines) and our [coding style](#coding-style).
- You provided the [requested details](#rfc-pull-requests) for new primitives or extended the existing [unit tests](#unit-tests) when fixing an issue.
- You [signed-off](#sign-your-work) your work. 

**Note**: This project follows the
[GitHub flow](https://guides.github.com/introduction/flow/index.html). To get started with pull requests, see [GitHub howto](https://help.github.com/en/articles/about-pull-requests).

### RFC pull requests

It is strongly advised to open an RFC (request for comments) pull request when contributing new
primitives. Please provide the following details:

* The definition of the operation as a oneCCL primitive. It should include an interface and semantics. We welcome sketches for the interface, but the semantics should be fairly well-defined.

* A use case, including a model and parallelism scenario.

### Code contribution guidelines

The code must be:

* *Tested*: oneCCL uses `gtests` for lightweight functional testing.

* *Documented*: oneCCL uses `Doxygen` for inline comments in public header
  files that are used to build the API reference and  `reStructuredText` for the Developer Guide. See [oneCCL documentation](https://oneapi-src.github.io/oneCCL/) for reference.

* *Portable*: oneCCL supports CPU and GPU
  architectures as well as different compilers and run-times. The new code should be complaint
  with the [System Requirements](README.md#prerequisites).

### Coding style

The general principle is to follow the style of existing or surrounding code.

### Unit tests

Be sure to extend the existing tests when fixing an issue.

### Sign your work

Use the sign-off line at the end of the patch. Your signature certifies
that you wrote the patch or otherwise have the right to pass it on as an
open-source patch. If you can certify
the below (from [developercertificate.org](http://developercertificate.org/)):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
660 York Street, Suite 102,
San Francisco, CA 94110 USA

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```
Then you add a line to every git commit message:

    Signed-off-by: Kris Smith <kris.smith@email.com>

**Note**: Use your real name.

If you set your `user.name` and `user.email` git configs, you can sign your
commit automatically with `git commit -s`.
