# Model-layer changes under the reach-avoid work

**The full document lives in the private model-authority repository:**
`vault-controller/docs/ssb_handoff/CHANGES_TO_THE_MODEL_LAYER.md`
(assumed checked out as a sibling of this repo; `vault/model_release.py` resolves it
the same way).

It is kept there deliberately: this repository is public, and the document contains
the robot's mass properties, geometry, and certified bounds. Nothing in THIS
repository carries model values — every quantity is read at runtime from the private
sibling via the hash-pinned release loader (`vault/data/controller_lock.sha256`),
and the figures it references are likewise in `vault-controller/docs/ssb_handoff/`.

Contents of the private document, for orientation:
  §§1–7   corrections to the model layer and what is NOT established
  §§8–13  the governor fix, four characterized limitations, the entropy-term
          analysis, preconditions, candidate terminal set, consumer gaps
  §14     ownership tables (safety theory vs filter architecture), the open-item
          registry, and pick-up mechanics for a coding agent
