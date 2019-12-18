all: mavros instrumentation

instrumentation:
	docker build -f Dockerfile.instrumentation -t dskatz/debgrind .

mavros:
	docker build \
	  -f Dockerfile.mavros \
	  -t dskatz/mavros:example \
	  .

arducopter:
	docker build \
	-f Dockerfile.ardu \
	-t dskatz/ardu:copter

.PHONY: mavros
