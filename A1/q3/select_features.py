#  script made with help of llm


import sys
import networkx as nx
from networkx.algorithms import isomorphism


def gaston_patterns_load_karo(file_path):
    """
    Gaston output padhta hai
    format dekha hai: # support, t id, v, e, x: tids
    """
    saare_patterns = []
    current_pattern = None
    current_support = 0

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue

            words = line.split()

            # support wali line → # 616
            if words[0] == '#':
                if len(words) > 1:
                    current_support = int(words[1])
                continue

            # naya graph shuru → t 1
            elif words[0] == 't':
                if current_pattern is not None:
                    saare_patterns.append(current_pattern)

                pattern_id = words[1] if len(words) > 1 else "0"

                current_pattern = {
                    'id': pattern_id,
                    'support': current_support,
                    'lines': [],
                    'graph': nx.Graph(),
                    'tids': set()           # transaction ids rakhenge
                }

            # node wali line → v 0 0
            elif words[0] == 'v':
                if current_pattern:
                    current_pattern['lines'].append(line)
                    current_pattern['graph'].add_node(int(words[1]), label=words[2])

            # edge wali line → e 0 1 0
            elif words[0] == 'e':
                if current_pattern:
                    current_pattern['lines'].append(line)
                    current_pattern['graph'].add_edge(int(words[1]), int(words[2]), label=words[3])

            # transaction ids → x: 0 1 5 7 ...
            elif words[0] == 'x:':
                if current_pattern:
                    current_pattern['tids'].update(words[1:])

    # aakhri pattern bhi add kar do
    if current_pattern:
        saare_patterns.append(current_pattern)

    return saare_patterns



def redundant_hai_kya(naya_pattern, selected_patterns):
    """
    dekhta hai ki naya pattern pehle walo mein se kisi se bohot similar to nahi
    fast → jaccard on tids, slow → isomorphism
    """
    node_match = isomorphism.categorical_node_match('label', None)
    edge_match = isomorphism.categorical_edge_match('label', None)

    for purana in selected_patterns:

        # FAST check → tids se jaccard
        if len(naya_pattern['tids']) > 0 and len(purana['tids']) > 0:
            common = len(naya_pattern['tids'].intersection(purana['tids']))
            total = len(naya_pattern['tids'].union(purana['tids']))

            if total == 0: 
                continue

            jaccard = common / total

            if jaccard > 0.90:
                return True
            else:
                continue   # low overlap → alag hi samjho

        # SLOW check → structural similarity
        sup_naya = naya_pattern['support']
        sup_purana = purana['support']

        if sup_purana == 0: 
            continue

        ratio = min(sup_naya, sup_purana) / max(sup_naya, sup_purana)

        if ratio < 0.90:
            continue

        # subgraph isomorphism check
        gm = isomorphism.GraphMatcher(
            purana['graph'], naya_pattern['graph'],
            node_match=node_match, edge_match=edge_match
        )

        if gm.subgraph_is_isomorphic():
            return True

        gm_rev = isomorphism.GraphMatcher(
            naya_pattern['graph'], purana['graph'],
            node_match=node_match, edge_match=edge_match
        )

        if gm_rev.subgraph_is_isomorphic():
            return True

    return False



def main():
    if len(sys.argv) < 5:
        print("Usage: python3 select_features.py <gaston_out> <output_file> <k> <db_size>")
        sys.exit(1)

    gaston_file = sys.argv[1]
    output_file = sys.argv[2]
    kitne_features_chahiye = int(sys.argv[3])
    total_database_size = int(sys.argv[4])

    print(f"patterns load kar raha hoon from {gaston_file}...")
    patterns = gaston_patterns_load_karo(gaston_file)
    print(f"{len(patterns)} patterns mile.")

    # 1. acchi quality wale patterns chuno (information gain type score)
    achhe_patterns = []

    for p in patterns:
        # chhote tukde skip
        if p['graph'].number_of_edges() < 1:
            continue

        support = p['support']
        probability = support / total_database_size

        p['score'] = probability * (1.0 - probability)
        achhe_patterns.append(p)

    # score ke hisaab se best pehle
    achhe_patterns.sort(key=lambda x: x['score'], reverse=True)

    # 2. top-k select karo (redundancy hata ke)
    final_selected = []

    print(f"top {kitne_features_chahiye} discriminative features select kar raha hoon...")

    for candidate in achhe_patterns:
        if len(final_selected) >= kitne_features_chahiye:
            break

        if not redundant_hai_kya(candidate, final_selected):
            final_selected.append(candidate)

        if len(final_selected) % 10 == 0:
            print(f"\r{len(final_selected)}/{kitne_features_chahiye} selected...", end="", file=sys.stderr)

    print(f"\nfinal selection: {len(final_selected)} features.")

    # 3. output file mein likh do
    with open(output_file, 'w') as f:
        for idx, pattern in enumerate(final_selected):
            f.write(f"t # {idx}\n")
            for line in pattern['lines']:
                f.write(line + "\n")


if __name__ == "__main__":
    main()