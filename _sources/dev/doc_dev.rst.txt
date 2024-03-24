=======================
Build the documentation
=======================

.. contents:: Table of Contents

Install requirements
--------------------

Building the :slc:`SLC <>` documentation requires `sphinx`_ and several sphinx plugins, the
``stable_learning_control`` python package, and some `LATEX`_ system dependencies. Most of the above
can be installed using the following `pip`_ command inside the ``./stable_learning_control`` folder:

.. code-block:: bash

    pip install .[docs]

To install the `LATEX`_ system dependencies, you can use the following command:

.. code-block:: bash

    sudo apt-get install texlive texlive-latex-extra texlive-science

.. _LATEX: https://www.tug.org/texlive/
.. _sphinx: https://www.sphinx-doc.org/en/master
.. _pip: https://pypi.org/project/pip/

Build the documentation
-----------------------

To build the `HTML`_ documentation, go into the :slc:`docs/ <tree/main/docs>`
directory and run the ``make html`` command. This command will generate the html documentation inside
the ``docs/build/html`` directory. If the documentation is successfully created, you can also use the
``make linkcheck`` command to check for broken links.

.. attention::
    Ensure you are in the Conda environment where you installed the :slc:`stable_learning_control <>`
    package with its dependencies.

.. note::
    Sometimes the ``make linkcheck`` command doesn't show the results on the stdout. You can also find the results
    in the ``docs/build/linkcheck`` folder. 

.. _HTML: https://www.w3schools.com/html/

Deploying
---------

The documentation is automatically built and deployed to the Github Pages site by the `Docs workflow`_
when a new version is released. You must `create a new release`_ to deploy documentation to the Github
Pages. Additionally, you can manually deploy the documentation through the `GitHub action interface`_
by running the `Docs workflow`_.

.. _`create a new release`: https://rickstaa.dev/stable-learning-control/dev/contributing.html#release-guidelines
.. _`Docs workflow`: https://github.com/rickstaa/stable-learning-control/actions/workflows/documentation.yml
.. _`GitHub action interface`: https://docs.github.com/en/actions/using-workflows/triggering-a-workflow#defining-inputs-for-manually-triggered-workflows
