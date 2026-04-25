import numpy as np

def solve(base_vectors, query_vectors, k, K, time_budget):
    import faiss

    base_arr = np.array(base_vectors, dtype=np.float32, copy=True)
    query_arr = np.array(query_vectors, dtype=np.float32, copy=True)

    num_base, dim = base_arr.shape
    use_lrge_mode = num_base > 800100

    if use_lrge_mode:
        num_lists = 1024 if num_base < 2_000_000 else 2048
        corse_quntizr = faiss.IndexFlatL2(dim)
        srch_indx = faiss.IndexIVFFlat(
            corse_quntizr,
            dim,
            num_lists,
            faiss.METRIC_L2,
        )

        random_state = np.random.RandomState(42)
        trin_size = min(num_base, 200000)
        trin_ids = random_state.choice(num_base, trin_size, replace=False)
        trin_vectors = base_arr[trin_ids]

        srch_indx.train(trin_vectors)
        srch_indx.add(base_arr)

        if time_budget >= 60:
            srch_indx.nprobe = 64
        elif time_budget >= 20:
            srch_indx.nprobe = 32
        else:
            srch_indx.nprobe = 16

        distnces, neighbrs = srch_indx.search(query_arr, k)

    else:
        srch_indx = faiss.IndexFlatL2(dim)
        srch_indx.add(base_arr)
        distnces, neighbrs = srch_indx.search(query_arr, k)

    all_indcs = neighbrs.ravel()
    valid_mask = all_indcs >= 0
    all_indcs = all_indcs[valid_mask]
    scors = np.bincount(all_indcs, minlength=num_base).astype(np.float32)

    ranking = np.lexsort((np.arange(num_base), -scors))[:K]

    return ranking.astype(np.int64)
