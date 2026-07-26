
PROFILES ={
    "naive":{
        "search_type": "dense",
        "enabled_hyde": False,
        "enabled_reranking":False,
        "enabled_crag": False,
        "enabled_self_rag": False,
        "top_k": 4

    },
    "sparse_search":{
        "search_type": "sparse",
        "enabled_hyde": False,
        "enabled_reranking":False,
        "enabled_crag": False,
        "enabled_self_rag": False,
        "top_k": 4

    },
    "hybrid":{
        "search_type": "hybrid",
        "enabled_hyde": False,
        "enabled_reranking":False,
        "enabled_crag": False,
        "enabled_self_rag": False,
        "top_k": 4

    },
    "hybrid+rerank":{
        "search_type": "hybrid",
        "enabled_hyde": False,
        "enabled_reranking":True,
        "enabled_crag": False,
        "enabled_self_rag": False,
        "top_k": 4

    },
    "hybrid+rerank+hyde":{
        "search_type": "hybrid",
        "enabled_hyde": True,
        "enabled_reranking":True,
        "enabled_crag": False,
        "enabled_self_rag": False,
        "top_k": 4

    },
    "hybrid+rerank+crag":{
        "search_type": "hybrid",
        "enabled_hyde": False,
        "enabled_reranking":True,
        "enabled_crag": True,
        "enabled_self_rag": False,
        "top_k": 4

    },
    "all":{
        "search_type": "hybrid",
        "enabled_hyde": True,
        "enabled_reranking":True,
        "enabled_crag": True,
        "enabled_self_rag": True,
        "top_k": 4

    }
}