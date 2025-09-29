from fm_factual.atom_extractor import AtomExtractor


class CustomAtomExtractor(AtomExtractor):

    def __init__(self, atoms):
        self.atoms = atoms

    def run(self, response):
        return {
            "num_atoms": len(self.atoms),
            "atoms": self.atoms,
            "all_atoms": [
                {"label": "claim", "atom": a} for i, a in enumerate(self.atoms)
            ],
            "all_facts": [
                {"label": "claim", "atom": a} for i, a in enumerate(self.atoms)
            ],
        }
