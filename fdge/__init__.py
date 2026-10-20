_PAPER_ALGORITHMS = (
    "SETUPinit", "SETUPGM", "SETUPOA", "UKGEN", "JOIN", "ISSUE", "GUPDATE",
    "Gr", "sampleR", "ENC", "DEC", "OPEN", "P", "V",
    "TSetup", "TAcc", "TWitness", "TVerify", "TUpdate",
)
_PROTOCOL_HELPERS = ("JOIN_CONTEXT", "PROOF_CONTEXT", "OPEN_CONTEXT", "OPEN_IDENTITY", "VERIFY_OPEN",
                     "EXPORT_GM_PUBLIC_CONTEXT", "BIND_GM_PUBLIC_CONTEXT")
__all__ = ["FDGEParams", *_PAPER_ALGORITHMS, *_PROTOCOL_HELPERS]


def __getattr__(name):
    if name == "FDGEParams":
        from .params import FDGEParams
        return FDGEParams
    if name in _PAPER_ALGORITHMS or name in _PROTOCOL_HELPERS:
        from . import paper_api
        return getattr(paper_api, name)
    raise AttributeError(name)
