.. _`Machine Learning Control`: https://github.com/rickstaa/machine-learning-control

=====================
Release documentation
=====================

.. contents:: Table of Contents

Install requirements
--------------------

Building the `Machine Learning Control`_'s `HTML`_ documentation requires `sphinx`_,
the `Machine Learning Control`_ package and several plugins. All of the above can be
installed using the following `pip`_ command:

.. code-block:: bash

    pip install -e .[docs]

.. _`sphinx`: http://www.sphinx-doc.org/en/master
.. _`pip`: https://pypi.org/project/pip/

If you also want to build the `LATEX`_ documentation, you have to install the `texlive-full`_
package.

.. _`texlive-full`: https://tug.org/texlive/

Build the documentation
-----------------------

Build HTML documentation
~~~~~~~~~~~~~~~~~~~~~~~~

To build the `HTML`_ documentation go into the `docs/`_ directory and run the
``make html`` command. This command will generate the html documentation
inside the ``docs/build/html`` directory.

.. note::

    Make sure you are in the Conda environment in which you installed the Machine Learning Control package
    with it's dependencies.

.. _`HTML`: https://www.w3schools.com/html/

Build LATEX documentation
~~~~~~~~~~~~~~~~~~~~~~~~~

To build the `LATEX`_ documentation go into the `docs/`_ directory and run the
``make latex`` command. This command will generate the html documentation
inside the ``docs/build/latex`` directory.

.. _`LATEX`: https://www.latex-project.org/help/documentation/

Deploying
---------

To deploy documentation to the Github Pages site for the repository,
push the documentation to the `main`_ branch and run the
``make gh-pages`` command inside the `docs/`_ directory.

.. warning::

    Please make sure you are on the `main`_ branch while building the documentation. Otherwise,
    you will be greeted by errors.

.. _`docs/`: https://github.com/rickstaa/machine-learning-control/tree/main/docs
.. _`main`: https://github.com/rickstaa/machine-learning-control/tree/main