.. _overview:


Overview
===========================

PyConTurb can be used in three main ways:  

1. Unconstrained simulation with default IEC 61400-1 parameters.  
2. Unconstrained simulation with custom functions for the mean wind speed,
   turbulence standard deviation and/or turbulence spectrum as function of
   spatial location and/or turbulence component.  
3. Simulation constrained on measured time series (e.g., met mast data).


Getting started
----------------

You will first need to specify the spatial locations and turbulence components
that you want to simulate. Look at :ref:`examples` to get ideas on how to do
this.

If you are constraining against time series, you will need to transform your
data into the correct format for PyConTurb before you can feed it to
``gen_turb``. Look at the constraining data in :ref:`examples` to check the
format. Currently, PyConTurb only supports constraining time series in the u,
v and/or w directions (i.e., no line-of-sight lidar measurements...yet).

If you want to model a custom spatial variation of the mean wind speed, 
turbulence standard deviation and/or turbulence spectrum (e.g., model a wake
deficit in the y-z plane), you will need to define the appropriate
``wsp_func``, ``sig_func`` and/or ``spec_func`` functions. Look at the related
:ref:`reference_guide` sections (namely, :ref:`profiles`, :ref:`sig_models` and
:ref:`spectral_models`).


Bugs/feature requests
------------------------------

This package is a labor of love, and it has by no means achieved perfection. If
you find a bug or you think it could benefit from a cool feature, please submit
an issue using GitLab's
`issue tracker <https://gitlab.windenergy.dtu.dk/pyconturb/pyconturb/issues>`_.
**NOTE!** Please be sure to use the "Bug" template for bugs and the
"Feature request" template for features you'd like to see implemented.


Contribution guidelines
-------------------------

You are very welcome to implement suggested changes to the code. Collaboration
is encouraged! Here is a quick-start guide to contributing:

#. Make sure your master branch is updated (``git pull origin master``).  
#. Switch to your feature branch (``git checkout -b branch_name``).  
#. Make your desired changes to the code.  
#. Add tests for your code in the relevant ``pyconturb/tests/`` file(s). See
   other test files to figure out how to do this.  
#. Run the tests locally to make sure they all pass (from repo,
   ``python -m pytest pyconturb/tests``).  
#. Add and commit your related files using git if you haven't already.  
#. Push your feature branch to GitLab (``git push origin branch_name``).  
#. Create a merge request in GitLab from your feature branch and assign to
   ``@rink``.  
#. Make any suggested changes based on the code review, then push those
   changes.  
#. After the merge request is accepted, pull the new changes to master
   (``git checkout master``, then ``git pull origin master``).


Theory
-------

The constraining method is based on the Veers simulation method.
The theory is presented in
`this paper <https://iopscience.iop.org/article/10.1088/1742-6596/1037/6/062032>`_,
which you can reference as follows:

    RINKER, Jennifer M. PyConTurb: an open-source constrained turbulence generator.
    In: Journal of Physics: Conference Series. IOP Publishing, 2018. p. 062032.

This section will be expanded as time permits.