# How to generate documentation:

## Environment preparation

	1. Make sure the following packages are installed:
		- doxygen
		- python3
	2. Install python deps: `python3 -m pip install -r requirements.txt`

## Doc generation

	1. Generate docs by calling `./merge_doc.sh`
	2. The following files should be re-generated:
		- man3/OneCCL.3
		- OneCCL.md
