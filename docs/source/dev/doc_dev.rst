=====================
Release documentation
=====================

.. contents:: Table of Contents

The BLC framework contains two `Github actions`_ that automatically check and
deploy new documentation:

    * The `docs_check_ci`_ action checks your changes to see if the documentation still builds.
    * The `docs_publish_ci`_ action deploys your documentation if a new version of the BLC framework is released.

Automatic build instructions
============================

To successfully deploy your new documentation, you have to follow the following development steps:

#. Create a new branch for the changes you want to make to the documentation (e.g. ``docs_change`` branch).
#. Make your changes to this branch.
#. Commit your changes. This will trigger the `docs_check_ci`_ action to run.
#. Create a pull request into the main branch if this action ran without errors.
#. Add a version bump label (``bump:patch``, ``bump:minor`` or ``bump:major``) to the pull request.
#. Merge the pull request into the main branch. The documentation will now be deployed using the `docs_publish_ci`_ action.

.. _`Github actions`: https://github.com/features/actions
.. _`docs_check_ci`: https://github.com/rickstaa/bayesian-learning-control/blob/main/.github/workflows/docs_check_ci.yml
.. _`docs_publish_ci`: https://github.com/rickstaa/bayesian-learning-control/blob/main/.github/workflows/docs_publish_ci.yml

.. tip::

    It is a good idea to `manually build the documentation <#build-the-documentation>`_ before pushing your changes to
    your branch. This way, you spot syntax errors early on in the development process.

Manual build instructions
=========================

Install requirements
--------------------

Building the BLC's `HTML`_ documentation requires `sphinx`_,
the BLC package and several plugins. All of the above can be
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

To build the `HTML`_ documentation, go into the `docs/`_ directory and run the
``make html`` command. This command will generate the html documentation
inside the ``docs/build/html`` directory.

.. note::
    Make sure you are in the Conda environment in which you installed the BLC package
    with it's dependencies.

.. _`HTML`: https://www.w3schools.com/html/

Build LATEX documentation
~~~~~~~~~~~~~~~~~~~~~~~~~

To build the `LATEX`_ documentation, go into the `docs/`_ directory and run the
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
    errors will greet you.

.. _`docs/`: https://github.com/rickstaa/bayesian-learning-control/tree/main/docs
.. _`main`: https://github.com/rickstaa/bayesian-learning-control/tree/main