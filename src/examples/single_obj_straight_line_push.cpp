/**
 * push one object from start to goal.
 * Steps:
 * 1. generate object trajectory (position)
 * 2. Option 1: CMGMP. Solve QP to find vel, force for a given contact point
                Assuming robot point contact is sticking.
                Contact point could be selected by heuristics
      Option 2: similar to CMGMP, but also optimize the (x,y,z) contact point location
 *
 */

