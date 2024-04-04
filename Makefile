.PHONY: metal
metal:
	go build -ldflags=-s -o metal cmd/metal/*

.PHONY: info
info:
	go build -ldflags=-s -o info cmd/info/*

.PHONY: opencl
opencl:
	go build -ldflags=-s -o opencl cmd/opencl/*