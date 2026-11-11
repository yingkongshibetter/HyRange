"""One MARIOH run with the authors' code and settings, exactly in the order of their main.py.

Runs in the MARIOH environment (HYRANGE_MARIOH_PYTHON; Python 3.12, torch CPU).  Imports the authors' functions from
the MARIOH clone (HYRANGE_MARIOH) and calls them as main.py does:
  set_random_seed(seed) -> preprocess_data(_preserved)(train), (test) -> select_edges_by_weight_condition_* on both
  -> maximal cliques of the filtered graphs -> prepare_sns_data(m_gt, 4, Gtrain_cliques) -> extract_features
  -> normalize_features -> train_classifier_net(params.py best_params of the dataset and mode)
  -> generate_pred_hyperedges(G_test, ..., best_thres, ratio_prob) -> evaluate_graph.
Differences from main.py, none of which changes a computed value: the two files are loaded one after the other
instead of by joblib Parallel(n_jobs=2); the unused m_gt_test is not computed (the filtered test graph is);
torch uses one thread; inference runs under torch.no_grad() (main.py never
calls .eval(), so dropout stays active at inference, as in main.py); the prediction is saved as json.

--query Q.json  (MARIOH on residual pair counts) replaces the test graph G_test by the weighted graph given as [[u, v, w], ...]
                (the residual pair counts W - W(F) after the degree-aware forcing); training is unchanged.

usage: python -I -B -X utf8 experiments/marioh_runner.py --data enron --mode preserved --seed 0 --out OUT.json --pred PRED.json
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from hyrange import bench as MB  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--mode", choices=["preserved", "reduced"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--query", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pred", required=True)
    a = ap.parse_args()

    sys.path.insert(0, str(MB.MAR))
    os.chdir(MB.MAR)          # main.py reads data/<ds>/...; nothing is written there
    import networkx as nx
    import numpy as np
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    from params import params_dict_preserved, params_dict_reduced
    from utils.data_processing import (normalize_features, prepare_sns_data, preprocess_data,
                                       preprocess_data_preserved, set_random_seed)
    from utils.evaluation import jaccard_similarity_hyperedges, multiset_jaccard_similarity, sanity_check
    from utils.feature_extraction import extract_features
    from utils.graph_operations import (generate_pred_hyperedges, select_edges_by_weight_condition_preserved,
                                        select_edges_by_weight_condition_reduced)
    from utils.training import CliqueDataset, train_classifier_net

    preserved = a.mode == "preserved"
    device = torch.device("cpu")
    set_random_seed(a.seed)
    rec = dict(data=a.data, mode=a.mode, seed=a.seed, query=bool(a.query), torch=torch.__version__,
               networkx=nx.__version__, numpy=np.__version__)
    t0 = time.perf_counter()
    # load_and_preprocess_data(data_, preserved), sequential
    pre = preprocess_data_preserved if preserved else preprocess_data
    sel = select_edges_by_weight_condition_preserved if preserved else select_edges_by_weight_condition_reduced
    G, gt = pre(str(MB.train_file(a.data, a.mode)))
    G_test, gt_test = pre(str(MB.target_file(a.data, a.mode)))
    if a.query:
        with open(a.query, "r", encoding="utf-8") as f:
            q = json.load(f)
        G_test = nx.Graph()
        G_test.add_edges_from(((int(u), int(v), {"weight": int(w)}) for u, v, w in q))
        rec["query_pairs"] = len(q)
        rec["query_weight_sum"] = int(sum(w for _, _, w in q))
    m_gt, m_G = sel(gt, G.copy())
    # main.py also computes m_gt_test, which it never uses; the filtered graph m_G_test does not depend on the gt
    # argument, so the gt argument is left empty here (saves the O(|gt| x #filtered pairs) list rebuild).
    _, m_G_test = sel([] if preserved else set(), G_test.copy())
    Gtrain_cliques = {tuple(sorted(c)) for c in nx.find_cliques(m_G)}
    Gtest_cliques = {tuple(sorted(c)) for c in nx.find_cliques(m_G_test)}
    rec["t_load_s"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    X, y = prepare_sns_data(m_gt, 4, Gtrain_cliques)
    features = extract_features(X, m_G, Gtrain_cliques)
    true_features = features[(np.array(y) == 1)]
    false_features = features[(np.array(y) == 0)]
    true_features, false_features, X_mean, X_std = normalize_features(true_features, false_features)
    X_train = torch.tensor((true_features.tolist() + false_features.tolist()), dtype=torch.float32)
    y_train = torch.tensor([1] * len(true_features) + [0] * len(false_features), dtype=torch.long)
    train_dataset = CliqueDataset(X_train, y_train)
    params_dict = params_dict_preserved if preserved else params_dict_reduced
    best_thres = params_dict[a.data]["best_thres"]
    ratio_prob = params_dict[a.data]["ratio_prob"]
    best_params = params_dict[a.data]["best_params"].copy()
    del best_params["loss_type"]
    rec["t_sampling_features_s"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    clf = train_classifier_net(train_dataset, device, **best_params, criterion=nn.CrossEntropyLoss())
    rec["t_train_s"] = time.perf_counter() - t2
    rec.update(best_thres=best_thres, ratio_prob=ratio_prob,
               best_params={k: (list(v) if isinstance(v, tuple) else v) for k, v in best_params.items()},
               n_train_pos=int(len(true_features)), n_train_neg=int(len(false_features)))

    t3 = time.perf_counter()
    with torch.no_grad():
        pred = generate_pred_hyperedges(G_test, gt_test, clf, X_mean, X_std, device, "test", best_thres, ratio_prob,
                                        Gtrain_cliques, Gtest_cliques)
    rec["t_reconstruct_s"] = time.perf_counter() - t3
    rec["t_total_s"] = time.perf_counter() - t0
    pred = [tuple(sorted(int(x) for x in e)) for e in pred]
    rec["sanity_projection_reproduced"] = bool(sanity_check(G_test, pred))
    if not a.query:
        # MARIOH's own evaluate_graph numbers
        rec["marioh_eval_jaccard"] = jaccard_similarity_hyperedges(pred, gt_test)
        rec["marioh_eval_multiset_jaccard"] = multiset_jaccard_similarity(pred, gt_test)
        rec["metrics"] = MB.metrics(pred, Counter(gt_test))
    rec["n_pred_copies"] = len(pred)
    MB.save_hyperedges(Counter(pred), a.pred)
    MB.save_json(rec, a.out)
    print("RESULT " + json.dumps({k: v for k, v in rec.items() if k != "best_params"}), flush=True)


if __name__ == "__main__":
    main()
