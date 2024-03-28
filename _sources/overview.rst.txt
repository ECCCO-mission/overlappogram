.. _overview:

Overview
=========

.. image:: _static/overlappogram.png
  :width: 800
  :alt: overlappogram image

``overlappogram`` works by accepting an overlappogram (an image of the Sun captured by a slitless spectrogram)
and the corresponding response
matrix for the instrument. It then utilizes row-by-row linear regression using an ElasticNet
to recover an emission measure cube that could have created the observation. To verify
this emission measure, it then generates spectrally pure maps which can be validated with a
ground truth observation.

Installation
-------------

``pip install overlappogram``


Running
--------

The most direct way to run ``overlappogram`` is using the ``unfold`` terminal program that gets installed.
It is executed simply: ``unfold config.toml`` where ``config.toml`` is a configuration file
defined according to :ref:`config` documentation.


.. _modes:

Optimization modes
--------------------

There are three possible optimization modes: "row,", "chunked," and "hybrid."

Row mode
+++++++++
"row" is the simplest optimization mode and is recommended for newcomers. In this optimization mode, the overlappogram
is divided into rows. Each row gets its own ElasticNet model to use when carrying out the inversion. Thus, each row is
inverted independently.


Chunked mode
+++++++++++++
"chunked" is the next simplest optimization row. The image is divided into a number of *chunks*
or sets of contiguous, non-overlapping rows.
The number of chunks is set by the *num_threads* parameter in the **execution** section of the configuration file.
Each chunk is given one ElasticNet model and a single thread for optimization. Thus, a single ElasticNet is used for
multiple rows of the inversion. This allows us to use warm starting. Warm starting is when the previous row's solution
is used as a starting point for inverting the next row in a chunk. To turn this feature on, set the *warm_start*
parameter in the **model** section of the configuration to true when used the "chunked" mode.

.. warning::
    Sometimes the chunked mode will result in *chunk artifacts* in the resulting inversions.
    These appear as discontinuous boundaries in the output spectrally pure images.
    We do not yet understand when this happens and doesn't happen.
    Thus, it is recommended to avoid chunked optimization unless you are confident.

Hybrid mode
++++++++++++
"hybrid" is a combination of the "chunked" and "row" optimization modes. The optimization begins in chunked mode but
switches to the row mode to optimize CPU performance. When inverting an overlappogram, some rows are harder to invert
than others and thus take more time. These rows tend to be adjacent and thus in the same chunk. Consequently, we noticed
that when using the chunked mode, we would finish most of the chunks but be waiting on a few rows of a chunk to complete.
Because chunks are done sequentially, there may only be a few rows left to invert.
Thus, the "hybrid" mode will stop doing chunked inversion and instead use row-based inversion when fewer than a set number
of rows remain to invert.
This number is defined in the "mode_switch_thread_count" parameter of the **execution** section of the configuration file.
