.. _`Machine Learning Control`: https://github.com/rickstaa/machine-learning-control

===============
Release package
===============

.. contents:: Table of Contents

Code formatting guidelines
--------------------------

To increase code quality, readability and usability, we use several language guidelines:

Python guidelines:
    * **Linting:** Please make sure your python code doesn't contain any errors by checking it with the `flake8 python linter`_.
    * **Formatting:** Please format all your scripts using the `black python formatter`_.

.. _`flake8 python linter`: https://flake8.pycqa.org/en/latest/
.. _`black python formatter`: https://github.com/psf/black

Markdown guidelines:
    * **Linting and formatting:** Please make sure your markdown code contains no errors and is formatter according to the `remark-lint`_ style guidelines.

.. _`remark-lint`: https://github.com/remarkjs/remark-lint

The `Machine Learning Control`_ package contains several `GitHub actions`_, which check code changes
against these coding guidelines. As a result, when the above guidelines are not met, you will
receive an error/warning when you create a pull request. If you think a code guideline is not correct
or your code structure doesn't allow you to respect the guideline, please state so in the
pull request.


.. _`Github Actions`: https://github.com/rickstaa/machine-learning-control/actions

General guidelines
------------------

Before releasing the package, make sure the following steps are performed:

    #. Create a pull request to pull the changes of your development branch onto the master branch.
    #. Make sure that all the `pull request checks`_ were successful.
    #. Squash and merge your branch with the main branch.
    #. Update the documentation according to :doc:`doc_dev` if needed.
    #. Bump the version using the `bump2version tool <https://pypi.org/project/bump2version/>`_.
    #. Check the version of the current branch using the ``bumpversion --list`` command.
    #. Add a tag equal to the version specified in the last step (Check versioning guidelines below).
    #. Update the changelog using the `auto-changelog <https://github.com/CookPete/auto-changelog>`_ tool.
    #. Commit and push the changes to the remote.
    #. Create a release using the GitHub draft release tool.

.. _`pull request checks`: https://github.com/rickstaa/machine-learning-control/actions

Versioning guidelines
---------------------

Additionally please use the `versioning guidelines specified at semver.org <https://semver.org/>`_.
