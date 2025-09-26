# __init__.py
# Re-export the node mappings so ComfyUI can discover them.

from .real_skin_post_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# (Optional) Some loaders also look for NODES_LIST; not required, but harmless.
NODES_LIST = list(NODE_CLASS_MAPPINGS.keys())