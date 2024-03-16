# Project-XF-methods
Running scripts for the MQC and quantum dynamics, Jupyter notebooks for plotting, and selected data

recipes

  This folder contains recipes of each MQC method defining necessary parameters


0-EXACT

  exact-model.py - Script for running QD calculations

  QD_reference.tar.bz2 - Compressed data of QD


1-ITMQC_and_other_methods

  NAMD-model.py - Script for running various MQC methods

  MQC_quantities.tar.bz2 - Compressed data for trajectory-averaged positions, coherences, populations over time
  MSE.tar.bz2 - Compressed data containing mean-square errors of coherences and populations

  Plot_quantities.ipynb - Notebook for plotting trajectory-averaged data vs. QD reference
  Accuracy_assessment.ipynb - Notebook for plotting the accuracy metrics
  Plot_snapshots.ipynb - Notebook for plotting the snapshots of specific time steps, featuring the TDPES from MQC dynamics and QD


2-ITMQC_without_BC

  This is for the ITMQC dynamics without the BC algorithm. The structure is similar to `1-ITMQC_and_other_methods`.


3-ITMQC_TD_widths
  
  This is for the ITMQC dynamics within the td width approximation X, X = Subotnik, Schwarz_1 (w^2 = 1), Schwarz_2 (w^2 = 4), Schwarz_3 (w^2 = 9).
  The structure is similar to `1-ITMQC_and_other_methods`.

