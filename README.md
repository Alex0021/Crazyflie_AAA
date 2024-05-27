# Crazyflie AAA

**Written By: Alex Hébert, Andrew Brown, Ovar Skjevdal, Jad Bahmdouni, and Sophie Lequeu
  Class: Aerial Robotics
  Code: MICRO-502
  Professor: Dr. Dario Floreano

## Table of Contents
- [Summary](#summary)
- [Results](#results)
- [Obstacle Avoidance Algorithm](#obstacle-avoidance-algorithm)
- [Video](#video)
- [File Structure](#file-structure)

## Summary
In this code we implemented a controller for the Crazyflie 2.0 open source drone to take off from a landing pad, navigate through obstacles, find a another landing pad, land, and then find its way back to the original landing pad. 

## Results
We were able to autonomously complete the obstacle course in 98.3 seconds.

## Obstacle Avoidance Algorithm
Our obstacle avoidance algorithm was made to be as simple and lightweight a possible. It is a version of a potential field, but in only takes into account the closest obstacle (or closest 2 obstacles if the 2 obstacles are separated enough). This made for efficient, clean code that was simple to program, debug, and adjust.

## Video
https://www.youtube.com/watch?v=nqIxoirHC4E

## File Structure
You can find our controller in my_control.py