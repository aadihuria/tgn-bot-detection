import numpy as np

from detectors.burst_detector import BurstCoordinationDetector


def test_cofollow_graph_ignores_single_incidental_overlaps():
    # two accounts that each follow the same target exactly once, an hour
    # apart, shouldn't get an edge -- that's just coincidence at this
    # population size, not coordination
    src = np.array([0, 1])
    dst = np.array([100, 100])
    t = np.array([0.0, 3600.0])

    det = BurstCoordinationDetector(window_hours=4.0, min_shared_events=2)
    G = det.build_cofollow_graph(src, dst, t)
    assert not G.has_edge(0, 1)


def test_cofollow_graph_connects_repeated_coincidences():
    # accounts 0 and 1 both hit target 100 and target 101 close together
    # in time, twice -- that should draw an edge
    src = np.array([0, 1, 0, 1])
    dst = np.array([100, 100, 101, 101])
    t = np.array([0.0, 60.0, 5000.0, 5100.0])

    det = BurstCoordinationDetector(window_hours=4.0, min_shared_events=2)
    G = det.build_cofollow_graph(src, dst, t)
    assert G.has_edge(0, 1)


def test_burst_cluster_is_flagged_and_organic_is_not():
    rng = np.random.default_rng(0)

    # bot cluster: 10 accounts, all fire at 3 shared targets within minutes
    bot_src, bot_dst, bot_t = [], [], []
    activation_times = [0.0, 5 * 86400, 10 * 86400]
    for activation in activation_times:
        for node in range(10):
            for target in [500, 501, 502]:
                bot_src.append(node)
                bot_dst.append(target)
                bot_t.append(activation + rng.uniform(0, 3600))

    # organic accounts: spread out over weeks, mostly distinct targets
    organic_src, organic_dst, organic_t = [], [], []
    for node in range(200, 260):
        n_follows = rng.integers(2, 6)
        for _ in range(n_follows):
            organic_src.append(node)
            organic_dst.append(int(rng.integers(0, 400)))
            organic_t.append(float(rng.uniform(0, 15 * 86400)))

    src = np.array(bot_src + organic_src)
    dst = np.array(bot_dst + organic_dst)
    t = np.array(bot_t + organic_t)

    order = np.argsort(t)
    src, dst, t = src[order], dst[order], t[order]

    det = BurstCoordinationDetector(window_hours=4.0, min_cluster_size=5, min_shared_events=2)
    G = det.build_cofollow_graph(src, dst, t)
    clusters = det.detect_coordinated_clusters(G, src, t)

    flagged_node_sets = [set(c["node_ids"]) for c in clusters]
    bot_nodes = set(range(10))
    assert any(len(bot_nodes & fs) / len(bot_nodes) > 0.7 for fs in flagged_node_sets)

    organic_nodes = set(range(200, 260))
    for fs in flagged_node_sets:
        organic_frac = len(fs & organic_nodes) / len(fs)
        assert organic_frac < 0.5


def test_coordination_score_is_zero_for_too_few_events():
    det = BurstCoordinationDetector()
    assert det.coordination_score(np.array([1.0, 2.0])) == 0.0
