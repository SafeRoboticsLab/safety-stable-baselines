# Environments

A growing set of small, self-contained reference environments that ship with
`safety_sb3` (under `safety_sb3.testing`) — each trainable to convergence in
minutes on CPU, each exercising the library on a problem where the right
behavior is obvious. They're the fastest way to see the algorithms work and to
validate your own setup against a known-good baseline.

For the **mjlab / GPU robot** environments (Go2 gap-jumping, Digit, crawl), see
[robot-safety-sandbox](https://github.com/SafeRoboticsLab/robot-safety-sandbox)
and its own environment showreel.

<div class="grid cards" markdown>

-   ### [Bicycle5D](bicycle5d.md)

    ![bicycle5d](assets/nominal_sac.gif){ width="320" }

    A 5-D bicycle drives to a goal through circular obstacles. **avoid** sits
    still; **reach-avoid** reaches from anywhere (SAC 100%, PPO 97%). Includes
    the learned value-function certificate and a PPO-vs-SAC value comparison.

    `SafetyPPO` · `SafetySAC` · `ReachAvoidPPO` · `ReachAvoidSAC`

</div>

Each page follows the same shape: what it teaches, the observation / action /
margin contract, a few lines to run it, and the expected result (GIF + value
figure). To add one, drop a module in `safety_sb3/testing/` and a page here.
