all: mavros instrumentation

instrumentation:
	docker build -f Dockerfile.instrumentation -t dskatz/debgrind .

mavros:
	docker build \
	  -f Dockerfile.mavros \
	  -t dskatz/mavros:example \
	  .

.PHONY: mavros
