"""Public molecular identity and Reactome topology, independent of patient outcomes."""
from collections import defaultdict
import csv
import gzip

import numpy as np

from common import MODALITIES, RELATIONS


def table(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def unique_values(pairs):
    values = defaultdict(set)
    for key, value in pairs:
        if key and value and key != "-" and value != "-":
            values[key].add(value)
    return {key: next(iter(items)) for key, items in values.items() if len(items) == 1}


def gene_symbols(path):
    return unique_values((r["Symbol"], r["GeneID"]) for r in table(path) if r["#tax_id"] == "9606")


def protein_symbols(path):
    return unique_values((r["accession"], r["gene_symbol"]) for r in table(path))


def verified_site_symbols(path):
    candidates = []
    for r in table(path):
        if (r["mapping_status"] != "validated" or r["ncbi_tax_id"] != "9606"
                or r["modification"] != "phosphorylation" or int(r["supporting_peptide_count"]) < 1
                or int(r["conflicting_peptide_count"]) != 0 or len(r["sequence_sha256"]) != 64
                or r["feature_id"] != f'{r["refseq_protein"]}_{r["residue"]}{r["position"]}'):
            raise ValueError("Unexpected row in verified PTM asset")
        if r["Gene"] and r["Gene"] == r["substrate_genesymbol"]:
            candidates.append((r["feature_id"], (r["substrate"], r["Gene"])))
    return {key: value[1] for key, value in unique_values(candidates).items()}


def reactome(files, wanted_genes):
    if files["version.txt"].read_text().strip() != "97":
        raise ValueError("Expected pinned Reactome release 97")
    with files["ReactomePathways.txt"].open(encoding="utf-8") as stream:
        human = {row[0] for row in csv.reader(stream, delimiter="\t")
                 if len(row) == 3 and row[0].startswith("R-HSA-") and row[2] == "Homo sapiens"}
    members = defaultdict(set)
    with files["NCBI2Reactome_All_Levels.txt"].open(encoding="utf-8") as stream:
        for row in csv.reader(stream, delimiter="\t"):
            if len(row) == 6 and row[5] == "Homo sapiens" and row[0] in wanted_genes and row[1] in human:
                members[row[0]].add(row[1])
    with files["ReactomePathwaysRelation.txt"].open(encoding="utf-8") as stream:
        parents = {(a, b) for a, b in csv.reader(stream, delimiter="\t") if a in human and b in human and a != b}
    return members, parents


def degree_shuffle(edges, rng, count):
    """Directed two-edge switches, preserving each node's in/out degree."""
    original = set(edges)
    shuffled = sorted(original)
    if len(shuffled) < 2:
        raise ValueError("Insufficient edges to construct a degree-preserving control")
    edge_set = set(shuffled)
    accepted = 0
    for i, j in rng.integers(0, len(shuffled), size=(8 * len(shuffled), 2)):
        a, b = shuffled[i]
        c, d = shuffled[j]
        if a == c or b == d or a == d or c == b or (a, d) in edge_set or (c, b) in edge_set:
            continue
        edge_set.remove((a, b))
        edge_set.remove((c, d))
        shuffled[i], shuffled[j] = (a, d), (c, b)
        edge_set.update((shuffled[i], shuffled[j]))
        accepted += 1
    if original == edge_set:
        raise ValueError("Degree-preserving shuffle left a knowledge relation unchanged")
    for axis in (0, 1):
        np.testing.assert_array_equal(np.bincount(np.asarray(sorted(original))[:, axis], minlength=count),
                                      np.bincount(np.asarray(shuffled)[:, axis], minlength=count))
    return sorted(shuffled), dict(edges=len(original), accepted_switches=accepted,
                                  changed_edges=len(edge_set - original))


def build_graph(assays, gene_map, members, parents, seed=42):
    # The identity join is fixed in all graph arms. An unmapped assay identity
    # has its own isolated node and shares the same two self updates.
    group = {modality: [(f"NCBIGene:{gene_map[symbol]}" if symbol in gene_map
                         else f"unmapped:{modality}:{feature}") for feature, symbol in assays[modality]]
             for modality in MODALITIES}
    genes = sorted({node for nodes in group.values() for node in nodes if node.startswith("NCBIGene:")})
    orphans = sorted({node for nodes in group.values() for node in nodes if node.startswith("unmapped:")})
    paths = sorted({p for node in genes for p in members.get(node.split(":")[1], [])})
    ids = genes + orphans + paths
    index = {node: i for i, node in enumerate(ids)}
    gene_edges = [(index[node], index[p]) for node in genes for p in sorted(members.get(node.split(":")[1], []))]
    path_edges = sorted((index[a], index[b]) for a, b in parents if a in index and b in index)
    rng = np.random.default_rng(seed)
    shuffled_genes, gene_stats = degree_shuffle(gene_edges, rng, len(ids))
    shuffled_paths, path_stats = degree_shuffle(path_edges, rng, len(ids))
    out = dict(node_ids=np.asarray(ids), node_type=np.repeat(np.arange(3), [len(genes), len(orphans), len(paths)]))
    stats = dict(nodes=dict(genes=len(genes), isolated_assay=len(orphans), pathways=len(paths)),
                 real_edges=dict(member=len(gene_edges), parent=len(path_edges)),
                 shuffled=dict(member=gene_stats, parent=path_stats), mappings={})
    for modality in MODALITIES:
        if not assays[modality]:
            raise ValueError(f"Missing assay features: {modality}")
        out[f"{modality}_ids"] = np.asarray([feature for feature, _ in assays[modality]])
        out[f"{modality}_node"] = np.asarray([index[node] for node in group[modality]], dtype=np.int64)
        stats["mappings"][modality] = dict(total=len(group[modality]),
                                            mapped=sum(node.startswith("NCBIGene:") for node in group[modality]))
    for arm, m, h in (("real", gene_edges, path_edges), ("shuffled", shuffled_genes, shuffled_paths)):
        for rel, edge in zip(RELATIONS, (m, [(b, a) for a, b in m], h, [(b, a) for a, b in h])):
            out[f"{arm}_{rel}"] = np.asarray(edge, dtype=np.int64).reshape(-1, 2)
    return out, stats
