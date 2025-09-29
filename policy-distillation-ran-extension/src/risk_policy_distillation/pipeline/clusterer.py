import json
import logging
import random
from json import JSONDecodeError

import numpy as np
from sentence_transformers import SentenceTransformer, util

from risk_policy_distillation.models.components.labeller import Labeller
from risk_policy_distillation.pipeline.fact_reasoner import (
    build_custom_reasoner_pipeline,
)


logger = logging.getLogger("logger")


class Clusterer:

    def __init__(
        self,
        llm_component,
        criterion,
        label_names,
        fr_treshold=0.90,
        start_threshold=0.7,
        end_threshold=0.4,
        min_community_size=2,
        n_labels=10,
        n_iter=200,
    ):
        self.llm_component = llm_component
        self.criterion = criterion
        self.label_names = label_names

        self.fact_reasoner_threshold = fr_treshold
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.min_community_size = min_community_size
        self.n_labels = n_labels
        self.n_iterations = n_iter

        self.labeller = Labeller(llm_component)

    def cluster(self, clustering_input, threshold=0.75, min_community_size=2):
        logger.info("Clustering {} instances".format(len(clustering_input)))

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(clustering_input)
        clusters = util.community_detection(
            embeddings, min_community_size=min_community_size, threshold=threshold
        )

        cluster_sentences = []
        for cluster in clusters:
            cs = []
            for sentence_id in cluster:
                cs.append(clustering_input[sentence_id])

            cluster_sentences.append(cs)

        return cluster_sentences, clusters

    def cleanup(self, graph, side=0):
        concept_nodes = graph.get_nodes(side)
        concept_values = [n.value for n in concept_nodes]

        # cluster in very similar groups
        cluster_sentences, _ = self.cluster(
            concept_values, threshold=0.95, min_community_size=self.min_community_size
        )

        logger.info(
            "Cleaned up {} clusters with {} total concepts".format(
                len(cluster_sentences), sum([len(cs) for cs in cluster_sentences])
            )
        )
        for cs in cluster_sentences:
            logger.info(cs)
            representative = random.choice(cs)

            merge_ids = [n.id for n in concept_nodes if n.value in cs]
            graph.merge_nodes(
                merge_ids, representative, probability=1.0, side=side, cleanup=True
            )

        return graph

    def run_fact_reasoner(self, name, contexts, context_prob, decision):
        # Set up the components of the pipeline
        model = "llama-3.1-8b-instruct"
        nli_prompt_version = "v1"

        atoms = [
            "The text was classified as {} because it contains the following concept: {}".format(
                decision, name
            )
        ]
        contexts = [
            (
                "The text was classified as {} because it contains the following concept: {} ".format(
                    decision, c
                ),
                context_prob[i],
            )
            for i, c in enumerate(contexts)
        ]

        r, g = build_custom_reasoner_pipeline(
            model, nli_prompt_version, atoms, contexts
        )
        return r, g

    def evaluate_graph(self, graph, cluster, results):
        direct_ent = [
            e
            for e in graph.edges
            if (e.target.startswith("a0") and e.type == "entailment")
        ]
        direct_equ = [
            e
            for e in graph.edges
            if (e.target.startswith("a0") and e.type == "equivalence")
        ]  # contexts directly entailed in the atom

        covered = [
            e.source
            for e in direct_equ + direct_ent
            if e.probability >= self.fact_reasoner_threshold
        ]
        score = len(covered)

        atom_prob = results["marginals"][0]["probabilities"][1]

        # nodes that are not covered by entailment or equivalence
        remaining = [
            cluster[int(graph.nodes[n].id.split("_")[-1])]
            for n in graph.nodes
            if (n not in covered and n.startswith("c"))
        ]

        return score, atom_prob, remaining

    def label_cluster(self, cluster, previous_names, best_name):
        additional_context = ""
        if len(previous_names):
            additional_context = """The following common reasons have previously been tried: {}. 
                 Generate a novel and creative common reason that has not
                 been tried before.""".format(
                previous_names, best_name
            )
        cluster_name = self.labeller.label(
            additional_context, str(cluster), temperature=1.0
        )

        return cluster_name

    def iterative_naming(self, cluster, context_prob, decision="harmful", use_fr=True):
        max_score = 0
        best_name = ""
        best_name_prob = 0.0
        remaining = []

        best_possible_coverage = len(cluster)
        previous_names = []
        for i in range(self.n_labels):
            try:
                name = self.label_cluster(cluster, previous_names, best_name)
                if (
                    not use_fr
                ):  # return the first option if fact reasoner is not being used
                    return name, 1.0, []
            except JSONDecodeError:  # the generated name is not in json format
                continue

            res, g = self.run_fact_reasoner(name, cluster, context_prob, decision)
            score, prob, r = self.evaluate_graph(g, cluster, res)

            previous_names.append(name)

            if score > max_score:
                max_score = score
                remaining = r
                best_name = name
                best_name_prob = prob

                if max_score == best_possible_coverage:
                    break

        return best_name, best_name_prob, remaining

    def run_clustering(self, graph, labels, use_fr=True, verbose=False):
        logger.info("Clustering...")

        sides = labels
        iter = 0

        thresholds = list(
            np.arange(
                self.start_threshold,
                self.end_threshold,
                step=((self.end_threshold - self.start_threshold) / self.n_iterations),
            )
        )

        while iter < self.n_iterations:
            side = sides[int(iter % len(labels))]

            for l in labels:
                graph = self.cleanup(graph, side=l)

            concept_nodes = graph.get_nodes(side)
            concept_values = [n.value for n in concept_nodes]
            concept_probs = [n.probability for n in concept_nodes]

            cluster_sentences, cluster_indices = self.cluster(
                concept_values, threshold=thresholds[iter]
            )
            if verbose:
                logger.info("Iteration = {}".format(iter))
                logger.info(
                    "\tNumber of nodes in graph: {}".format(
                        {l: len(graph.nodes[l]) for l in graph.labels}
                    )
                )
                logger.info("\tUsing threshold = {}".format(thresholds[iter]))
                logger.info("Generated {} clusters".format(len(cluster_sentences)))

            if not len(cluster_sentences):
                iter += 1
                continue

            for cluster_id, cs in enumerate(cluster_sentences):
                cs_prob = [concept_probs[i] for i in cluster_indices[cluster_id]]

                # get the label
                decision = self.label_names[side]
                best_name, probability, remaining = self.iterative_naming(
                    cs, cs_prob, decision=decision, use_fr=use_fr
                )

                if best_name == "":
                    iter += 1
                    continue

                # remove remaining from the cluster
                for r in remaining:
                    cs.remove(r)

                if len(cs) > 1:
                    # merge nodes
                    merge_ids = [n.id for n in concept_nodes if n.value in cs]
                    graph.merge_nodes(merge_ids, best_name, probability, side)

                    # print the merged
                    if verbose:
                        logger.info("Named new cluster = \t{}".format(best_name))
                        for s in cs:
                            logger.info("\t\t{}".format(s))

            iter += 1

        return graph
