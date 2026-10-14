"""
discovery — finding which models an option can use, from the providers' own
model lists rather than hard-coded IDs (free models get renamed and retired).

    openrouter  — GET /api/v1/models, filtered to models CONDUCTOR can use
    nvidia      — the catalog's allowlist, validated against GET /v1/models
    cache       — the local cache both use (and the tool-calling test results)
"""
