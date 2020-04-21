.. _installation:


Installation
===========================

Before you can install the package, you need to have a suitable Python package
installed. If you don't have Python, follow these instructions before attempting
to install PyConTurb.

.. toctree::
    :maxdepth: 2

    install_python
    

Requirements
--------------------------------

PyConTurb requires Python 3.6+, so be sure that you are installing it into
a suitable environment (see above). For simple users, all dependencies will be
installed with PyConTurb, except for two optional packages for running the
examples and plotting. For developers, use the ``dev_reqs.txt`` requirements file
to install the optional dependencies related to testing and building documentation.


Normal user
--------------------------------

* Install the most recent version of the code::
  
    pip install git+https://gitlab.windenergy.dtu.dk/pyconturb/pyconturb.git

* Update an installation to the most recent version::

    pip install --upgrade git+https://gitlab.windenergy.dtu.dk/pyconturb/pyconturb.git

* To run the notebook examples on your machine (optional)::

    pip install jupyter matplotlib

* To install based off a particular tag (e.g., ``v2.0``), branch
  (e.g., ``master``) or commit hash (e.g., ``1ea5060453ee9ce26b265065333dd1d370b3e8b6``)::
  
    pip install git+https://gitlab.windenergy.dtu.dk/pyconturb/pyconturb.git@v2.0
    pip install git+https://gitlab.windenergy.dtu.dk/pyconturb/pyconturb.git@master
    pip install git+https://gitlab.windenergy.dtu.dk/pyconturb/pyconturb.git@1ea5060453ee9ce26b265065333dd1d370b3e8b6


Developer
------------------------------

We highly recommend developers install PyConTurb into its own environment
(instructions above). The commands to clone and install PyConTurb with developer
options into the current active environment in an Anaconda Prompt are as
follows::

   git clone https://gitlab.windenergy.dtu.dk/pyconturb/pyconturb.git
   cd PyConTurb
   pip install -r dev_reqs.txt
   pip install -e .
