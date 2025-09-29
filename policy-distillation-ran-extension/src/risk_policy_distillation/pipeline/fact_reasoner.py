import os

from dotenv import load_dotenv

from fm_factual.atom_reviser import AtomReviser
from fm_factual.custom.custom_atom_extractor import CustomAtomExtractor
from fm_factual.custom.custom_retriever import CustomRetriever
from fm_factual.fact_reasoner import FactReasoner
from fm_factual.nli_extractor import NLIExtractor


load_dotenv()
CACHE_DIR = os.getenv("CACHE_DIR")
MERLIN_PATH = os.getenv("MERLIN_PATH")


def build_custom_reasoner_pipeline(model, nli_prompt_version, atoms, contexts):
    context_retriever = CustomRetriever(contexts=contexts)
    atom_extractor = CustomAtomExtractor(atoms=atoms)
    atom_reviser = AtomReviser(model)
    nli_extractor = NLIExtractor(model, prompt_version=nli_prompt_version)

    # Create the FactReasoner pipeline
    pipeline = FactReasoner(
        context_retriever=context_retriever,
        atom_extractor=atom_extractor,
        atom_reviser=atom_reviser,
        nli_extractor=nli_extractor,
        merlin_path=MERLIN_PATH,
        debug_mode=False,
        use_priors=False,
    )

    pipeline.build(
        response="just a placeholder here",
        has_atoms=False,
        revise_atoms=False,
        has_contexts=False,
        rel_atom_context=True,
        rel_context_context=False,
        contexts_per_atom_only=False,
        remove_duplicates=True,
    )

    results, marginals = pipeline.score()

    return results, pipeline.fact_graph
