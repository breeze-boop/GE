from __future__ import annotations
import hashlib
import hmac
import json
from .paper_relation import (PaperPublicStatement, PaperPrivateWitness,
                             LinearEquation, Variable, CoefficientBound)
from .trapdoor import gadget_matrix, verify_trapdoor
from .zkp_test2 import PiProof


def public_statement(pp, A):
    from .scheme import _matrix_shape, _pack_matrix, _required_trapdoor_bound
    p,ring=pp.params,pp.ring_q
    rows,prefix,cols=p.alpha1,2*p.alpha1,5*p.alpha1
    bound = _required_trapdoor_bound(pp)
    if not _matrix_shape(A,rows,prefix+cols,ring):
        raise ValueError('invalid JOIN public-key shape')
    G=gadget_matrix(ring,rows,5,pp.q2)
    equations=[]
    for i in range(rows):
        for j in range(cols):
            equations.append(LinearEquation(f'join_trapdoor[{i},{j}]',
                tuple((A[i][h],Variable('R',h*cols+j)) for h in range(prefix)),
                G[i][j]-A[i][prefix+j]))
    param_bytes=json.dumps(vars(p),sort_keys=True,separators=(',',':')).encode()
    crs_seed=hashlib.sha256(b'LatGE-A5-CRS'+pp.parcom.encode()).digest()
    encoding=(b'LatGE-JOIN-short-trapdoor-v2'+len(param_bytes).to_bytes(8,'little')+
              param_bytes+_pack_matrix(A)+_pack_matrix(G)+crs_seed)
    profile=(('ell',p.ell),('kappa1',p.kappa1),('kappa2',p.kappa2),
             ('rejection_delta',getattr(p,'zk_rejection_delta',None)),
             ('conjugates',getattr(p,'zk_conjugates',4)),
             ('research_reference',getattr(p,'zk_research_reference',False)))
    return PaperPublicStatement(ring,encoding,tuple(equations),(),
        (CoefficientBound('R',prefix*cols,-bound,bound),),(('R',prefix*cols),),profile,crs_seed,pp.parcom)


def prove(pp, key):
    if not verify_trapdoor(key):
        raise ValueError('invalid trapdoor')
    public=public_statement(pp,key.A)
    witness=PaperPrivateWitness({'R':[x for row in key.R for x in row]})
    from .join_streaming import prove_join
    transcript=prove_join(public,witness)
    return PiProof(public.digest(),transcript)


def verify(pp,A,proof):
    if type(proof) is not PiProof or type(proof.statement_digest) is not bytes or len(proof.statement_digest)!=32:
        return False
    if type(proof.transcript) is not bytes or not proof.transcript:
        return False
    try:
        public=public_statement(pp,A)
    except (ValueError,TypeError,IndexError,AttributeError):
        return False
    if not hmac.compare_digest(public.digest(),proof.statement_digest):
        return False
    from .join_streaming import verify_join
    return verify_join(public,proof.transcript)
