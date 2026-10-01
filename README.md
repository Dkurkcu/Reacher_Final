# Energy-Aware Reinforcement Learning for Proprioceptive Stall Detection

Official repository for the Brief Research Report: **"Energy-Aware Reinforcement Learning for Proprioceptive Stall Detection"**.

This repository contains the complete simulation environment, trained policy models, and evaluation scripts required to reproduce the energy metrics and behavioral results presented in the paper. 

## Overview
This project investigates how incorporating energy awareness into a Reinforcement Learning (RL) agent influences learned behavior in a planar 3-DOF robotic manipulator. By augmenting the standard policy observation with cumulative mechanical work and a temporal memory of proprioceptive stall detection, the Energy-Aware RL (E-RL) agent learns to:
1. Preserve reaching efficiency in unobstructed space.
2. Maintain productive physical interaction with lightweight, pushable obstacles.
3. Successfully detect and disengage from ineffective effort (stalling) against heavyweight, blocking obstacles to find alternative routes.

## Videos

The evaluation videos demonstrating unobstructed reaching, pushing a lightweight obstacle, and stall recovery against a heavyweight obstacle can be found in the `media` folder of this repository.

## Repository Structure

```text
├── env/
│   └── reacher_latest.py       # Custom Gymnasium environment for the 3-DOF planar manipulator
├── scripts/
│   ├── record_data.py          # Script to evaluate policies over 650-step horizons
│   └── generate_plots.py       # Generates the 3x3 Results Grid (Distance, Mechanical Energy, Stall Index)
├── models/
│   ├── baseline_model.zip      # Trained standard PPO baseline
│   └── erl_model.zip           # Trained E-RL policy with Deadlock Mitigation Strategy (DMS)
├── media/                      # Directory containing the evaluation videos and output figures
├── README.md
└── requirements.txt            # Python package dependencies
