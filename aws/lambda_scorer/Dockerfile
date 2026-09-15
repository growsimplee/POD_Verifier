FROM public.ecr.aws/lambda/python:3.12

# ---------------------------------------------------------------------------
# CPU-only PyTorch + torchvision, PINNED for reproducible builds.
# torch/torchvision come from the PyTorch CPU index (no CUDA); their pure-python
# deps come from PyPI. numpy pinned <2 because torch 2.2.x's numpy bridge is
# built against numpy 1.x (numpy 2 breaks torch.from_numpy at runtime).
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
      torch==2.2.2 torchvision==0.17.2 "numpy<2"

COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Application code + trained checkpoint (ImageNet-normalized inference is the
# default in handler.py — the validated, correct preprocessing).
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY model/best.pt /opt/model/best.pt
COPY handler.py ${LAMBDA_TASK_ROOT}/
COPY sarathy_client.py ${LAMBDA_TASK_ROOT}/

# Fail the build early if the runtime stack is broken, and confirm the
# checkpoint loads into the architecture with no key mismatches.
RUN python -c "import torch, timm, cv2, numpy, boto3; print('deps ok', torch.__version__, 'numpy', numpy.__version__)" \
 && python -c "import torch; from src.model import MultiHeadEfficientNet; \
m=MultiHeadEfficientNet(4, pretrained=False); \
ck=torch.load('/opt/model/best.pt', map_location='cpu', weights_only=False); \
st=ck.get('model_state_dict', ck) if isinstance(ck, dict) else ck; \
miss,unexp=m.load_state_dict(st, strict=False); \
assert not miss and not unexp, ('checkpoint/arch mismatch', len(miss), len(unexp)); \
print('checkpoint loads cleanly: 0 missing / 0 unexpected')"

CMD ["handler.handler"]
