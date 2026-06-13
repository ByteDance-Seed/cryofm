SEARCHER_REGISTRY = {
    "coarse": CoarseSearcher,
    "fine": FineSearcher,
}

cls = SEARCHER_REGISTRY[state.schedule.searcher_name]