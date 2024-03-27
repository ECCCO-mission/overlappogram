Development
============
We encourage all contributions. Please see our `contribution guide first <https://github.com/eccco-mission/eccco-mission/blob/main/contributing.md>`_.

We recommend working in a virtual environment.
This can be created by running ``python -m venv venv``. Then, activate the environment with ``source venv/bin/activate``.
You can then install the required packages with ``pip install -e ".[dev]"``.

If at any time you run into issues, please contact us by :doc:`following the guidelines here <help>`.

Setting up pre-commit
----------------------

The first time you develop code, you'll need to install the pre-commit. This checks that our style is consistent.
It gets installed when you do ``pip install -r ".[dev]"`` but then requires you to activate them by
running ``pre-commit install``. Now every time you commit, our checks will run first.

Building the docs
------------------
The docs are built using ``sphinx``. First, you must install it and the other documentation requirements with ::

    pip install -r ".[docs]"

Then, navigate to the ``docs`` directory and run ``make html`` to build the docs.

Running tests
-------------
To run the tests for this package, run ``pytest`` in the repository base directory.
