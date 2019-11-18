# Documentation Generation

## Description ##

The documentation is written using the restructured text markup language (also referred to as reST) and can be built with Doxygen and Sphinx. 

## Software Requirements ##

* Doxygen 1.8.16
* Python 3.7.1 (may or may not work with older Python*, untested)

Once you have the software requirements set up, go to the `doc` directory and run the commmands described in the next section.

## Building Documentation ##

1. Install the required Python* libraries using `pip`:
  ```bash
  pip install -r requirements.txt
  ```
2. Run `build_docs.sh` to generate documentation.

  The `build_docs.sh` script generates the Doxygen XML tree from the source code, parses the generated output, converts it into reST format, and then generates the HTML output from all the `.rst` files in the `doc/rst/sources` directory.

## Building Specifications ##

1. Install the required Python* libraries using `pip`: 
  ```bash
  pip install -r requirements.txt
  ```
2. Run `build_spec.sh` to generate oneCCL specifications. 

  The `build_spec.sh` script copies the relevant topics from `doc/rst/source` and then generates the HTML output from the source files in `doc/spec/source` folder. 

## Configure Doxygen ##

The Doxygen configuration lives in the `docs/Doxyfile` file.
Please refer to the [Doxygen configuration reference](http://www.doxygen.nl/manual/config.html) for more information.


## Configure Sphinx ##

You can create and modify Sphinx settings in the `docs/rst/source/conf.py` file.

For more details, please refer to the [Sphinx configuration reference](https://www.sphinx-doc.org/en/master/usage/configuration.html).
