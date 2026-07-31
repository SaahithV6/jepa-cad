# Source this to make the local .venv usable for Modal + torch.
# The uv-built .venv was missing a few pure-Python deps (typing_extensions,
# pyparsing, multidict, yarl, protobuf); they live in ./pylibs as a shim so we
# don't have to mutate the (sometimes read-only) venv. Remote Modal images
# install the full requirements.txt, so this shim is only for the local client.
export PYTHONPATH="/home/best/jepa-cad/pylibs${PYTHONPATH:+:$PYTHONPATH}"
