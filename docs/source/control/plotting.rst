================
Plotting Results
================

MLC ships with a simple plotting utility that can be used to plot diagnostics from experiments. You can run it with:

.. parsed-literal::

    python -m machine_learning_control.run plot [path/to/output_directory ...] [--legend [LEGEND ...]]
        [--xaxis XAXIS] [--value [VALUE ...]] [--count] [--smooth S]
        [--select [SEL ...]] [--exclude [EXC ...]]

For more information on using this utility, see the :ref:`plot utility <plot>` documentation or code :ref:`api`.

.. figure:: ../images/plots/lac/example_lac_performance_plot.svg
    :align: center

    Example plot that displays the performance of the LAC algorithm.