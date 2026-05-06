"""Test fixture — registered by test_enrichment_plugin_loaded_at_startup."""
LOADED = False


def register():
    global LOADED
    LOADED = True
