Overview

We use the reStructuredText (rST) lightweight markup language to update documentation.

rST files can be edited in any editor, but it is recommended to use Visual Studio Code or Sublime because of rST-specific extensions and packages.
To add a new topic, create a new .rst file to the source folder and add it to the TOC (index.rst).  

rST syntax guidelines:
	https://docs22.readthedocs.io/en/latest/rst-markup.html
	http://docutils.sourceforge.net/docs/user/rst/quickref.html

Generate rST documentation

When changes to docs are made, you can generate them locally to see how the look like:

1.	Make sure to have Python 3.X installed and the pip package manager.
2.	Install Sphnix to build rst documentation and Read the Docs theme that gives documentation a nice look: 
	pip install sphinx
	pip install sphinx_rtd_theme
3.	Generate documentation using the makefile (makefile.bat on Windows or ./Makefile on Linux). The docs are generated into ../source/html folder.

TODO: Update this file

To update GitHub Pages:
1.  got to doc/rst
2.  remove build folder if exist
3.  run makefile.bat on Windows or "make html" on Linux
4.  download gh-pages branch to another directory
5.  remove content from gh-pages branch
6.  copy build/html/* to gh-pages branch
7.  push gh-pages branch to GitHub.
