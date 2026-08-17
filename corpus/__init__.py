"""Coverage corpus and scoring harness for ChihuahuaBot.

This package is the fitness function, and it is deliberately independent of the
analyser it scores. Nothing under ``corpus/`` may import ``chihuahuabot`` except
through the resolver plug point in :mod:`corpus.score` — if the measuring
instrument shared its AST helpers with the system under test, a bug in those
helpers would silently agree with itself and the coverage number would be a
tautology.
"""

from __future__ import annotations
