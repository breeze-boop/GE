from __future__ import annotations


from .paper_zkp_backend import LNS20Parameters, PaperProofResourceLimit
from .ring import Ring
from .strict_relation import StrictRelation


PACKING_PROFILE = 'main-iii-d-no-copy-research-packing-v1'


def _dimensions(relation, parameters, max_messages):
    if not isinstance(relation, StrictRelation) or not isinstance(parameters,LNS20Parameters):
        raise TypeError('StrictRelation and explicit LNS20Parameters are required')
    if type(max_messages) is not int or max_messages <= 0:
        raise ValueError('max_messages must be a positive integer')
    parameters.validate()
    p, ell = relation.params, parameters.degree
    if (parameters.modulus != p.q or ell != p.ell or
            parameters.kappa1 != p.kappa1 or parameters.kappa2 != p.kappa2):
        raise ValueError('proof modulus, degree and ranks must match the public strict relation')
    relation.layout.validate_shapes()
    if relation.n % ell or relation.scalar_count % ell:
        raise ValueError('strict layout must have n and tn divisible by ell')
    if any(block.offset % ell or block.scalar_length % ell for block in relation.layout.witness):
        raise ValueError('strict witness block boundaries must align with ell')
    messages = relation.scalar_count//ell
    if messages > max_messages:
        raise PaperProofResourceLimit(
            f'original III-D layout needs {messages} messages; limit is {max_messages}; '
            'no constraints or private coordinates have been expanded')
    if parameters.commitment_crs is not None and parameters.commitment_crs.message_count != messages:
        raise ValueError('supplied research CRS must have exactly tn/ell messages')
    return Ring(ell,p.q), messages


