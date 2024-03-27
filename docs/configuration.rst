.. _config:

Configuration
==============

Each run of `unfold` requires a configuration file in the `TOML format <https://toml.io/en/>`. The configuration file
is divided into sections:

- **paths**: provides the input file paths.
- **output**: describes the properties of the output.
- **inversion**: defines inversion parameters, how the inversion operation is defined.
- **model**: defines the parameters for the ``sklearn`` ElasticNet that is used.
- **execution**: describes the execution of the inversion, including computer specific properties like the number of threads.

.. note::
    All sections and parameters are expected in the configuration file. There are no optional parameters with defaults.

`We provide an example configuration file here. <https://github.com/ECCCO-mission/overlappogram/blob/main/example_config.toml>`

**paths** section
------------------

There are four configurables for this section:

- *overlappogram*: path to the overlappogram image to be inverted.
- *weights*: path to the accompanying weights used in the inversion.
- *response*: path to the instrument response.
- *gnt*: path to the file containing atomic physics values from Chianti, the *G(n, t)* function.

**output** section
--------------------

There are three configurables for this section:

- *make_spectral*: if true, makes spectrally pure images as output. otherwise, these files are not made.
- *directory*: path to the directory where output files are written.
- *overwrite*: if true, output files will be overwritten. otherwise, the program will fail writing if a file already exists.

**inversion** section
----------------------

There are six configurables for this section:

- *solution_fov_width*: an integer for the field-of-view width in pixels used in the solution. We suggest 2.
- *detector_row_range*: a list of two integers defining the range of detector rows to invert. For example, [10, 35] would run between 10 and 35.
- *field_angle_range*: a list of two integers defining the range of field angles to use in inversion. Units are arc seconds.
- *response_dependency_name*: for now, only "logt" is supported.
- *response_dependency_list*: a list of floats defining the logarithm of the temperature used in the response dependency.
- *smooth_over*: ?

**model** section
-------------------

This section defines the parameters used by `Scikit-Learn's ElasticNet <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html>`.
You can find more exhaustive descriptions of the parameters at that link in the `sklearn` documentation.

There are six configurables for this section:

- *alphas*: A list of floating point numbers defining the various values of alpha to iterate over. Alpha is a constant in ``sklearn`` that multiplies a penalty term.
- *rhos*: A list of floating point numbers defining the various values of rho to iterate over. This corresponds to ``sklearn``'s `l1_ratio` parameter.
- *warm_start*: Boolean indicating whether a warm start should be used when training the model. Note that if the *mode* (see the **execution** section) is "row" this has no effect.
- *tol*: The tolerance for the optimization.
- *max_iter*: The maximum number of iterations.
- *selection*: Either set to "cyclic" or "random". If set to "random", a random coefficient is updated every iteration rather than looping over features sequentially by default.


**execution** section
-----------------------

There are three configurables for this section:

- *num_threads*: The number of threads to use when optimizing.
- *mode_switch_thread_count*: Only used if *mode* is set to "hybrid". In that case, when the number of remaining threads is less than this value, the optimization switches from "chunked" to "row".
- *mode*: The optimization mode can be set to three different values: "row", "chunked", or "hybrid". See :ref:`modes` for a description of what these do.
