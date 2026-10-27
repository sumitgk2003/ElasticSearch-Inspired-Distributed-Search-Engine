"""Text analysis shared by indexing and querying.

The same analyzer must run at index time (on shards) and at query time
(on the coordinator), otherwise terms will not line up with postings.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import List

from nltk.stem import PorterStemmer

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A compact English stop-word list (Lucene/Elasticsearch "english" set plus
# a few very common function words).
STOP_WORDS = frozenset(
    """
    a an and are as at be but by for if in into is it no not of on or such
    that the their then there these they this to was will with i you he she
    we me my our your his her its them from have has had do does did been
    being were am so than too very can could would should may might must
    what which who whom whose when where why how all any both each few more
    most other some own same just only also about above after again against
    below between during further here once over under until up down out off
    """.split()
)

_stemmer = PorterStemmer()


@lru_cache(maxsize=200_000)
def stem(token: str) -> str:
    return _stemmer.stem(token)


def analyze(text: str) -> List[str]:
    """Lowercase, tokenize on alphanumerics, drop stop words, Porter-stem."""
    return [stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOP_WORDS]
