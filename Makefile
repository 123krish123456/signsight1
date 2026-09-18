# SignSight. PRD §7 M7 asks for results reproducible from a single `make evaluate`.
#
# Every target is safe to re-run: preparation skips archives it has already extracted,
# and training refuses to start unless the manifest passes the signer-disjoint check.

ARCHIVES ?= E:/datasets/INCLUDE
PACK     ?= backend/vocab/isl_v1.json
ARCH     ?= bilstm
EPOCHS   ?= 60

PY := python

.PHONY: help install test data manifest train export evaluate all clean serve app ci

help:
	@echo "make install    - python + node dependencies"
	@echo "make test       - the test suite and module self-checks"
	@echo "make data       - extract archives and ingest into the manifest (ARCHIVES=...)"
	@echo "make manifest   - assign signer-disjoint splits and check the M2 gate"
	@echo "make train      - train the classifier (ARCH=bilstm|transformer EPOCHS=60)"
	@echo "make export     - Keras -> ONNX, with the equivalence and latency gates"
	@echo "make evaluate   - accuracy, confusion matrix, per-signer, ablations"
	@echo "make all        - data -> manifest -> train -> export -> evaluate"
	@echo "make serve      - run the backend"
	@echo "make app        - run the speaker app"

install:
	pip install -e ".[dev,ml,train]"
	cd app && npm install && npm run fetch-assets

test:
	pytest -q
	$(PY) -m ml.features.extract
	$(PY) -m backend.pipeline.buffer
	$(PY) -m backend.mock

data:
	$(PY) -m ml.data.prepare --archives "$(ARCHIVES)"

manifest:
	$(PY) -m ml.data.manifest --assign --check

train:
	$(PY) -m ml.train --arch $(ARCH) --epochs $(EPOCHS) --pack $(PACK)

export:
	$(PY) -m ml.export_onnx

evaluate:
	$(PY) -m ml.evaluate --split test
	$(PY) -m ml.evaluate --split test --ablate face

# The whole path from downloaded archives to reported numbers.
all: data manifest train export evaluate

serve:
	uvicorn backend.main:app --reload

app:
	cd app && npm run dev

ci:
	pytest -q

clean:
	rm -rf ml/reports ml/data/cache
	@echo "left models and the manifest alone - delete those by hand if you mean it"
