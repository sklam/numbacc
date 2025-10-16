setup-workspace: deps/sealir deps/spy


deps/sealir:
	bash scripts/checkout.sh https://github.com/sklam/sealir wip/updates deps/sealir

deps/spy:
	bash scripts/checkout.sh https://github.com/sklam/spy wip/numbacc_tensor deps/spy


build:
	pip install -e ./deps/sealir
	pip install -e './deps/spy[dev]'
	make -C ./deps/spy/spy/libspy
	pip install -e .