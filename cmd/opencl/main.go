package main

import (
	"fmt"
	"log"
	"strings"

	"github.com/opengs/go-opencl/opencl"
)

const (
	deviceType = opencl.DeviceTypeAll

	dataSize = 128

	programCode = `
kernel void kern(global float* out, global float* a, global float* b)
{
	size_t i = get_global_id(0);
	out[i] = a[i] + b[i];
}
`
)

func printHeader(name string) {
	fmt.Println(strings.ToUpper(name))
	for _ = range name {
		fmt.Print("=")
	}
	fmt.Println()
}

func printInfo(platform opencl.Platform, device opencl.Device) {
	var platformName string
	err := platform.GetInfo(opencl.PlatformName, &platformName)
	if err != nil {
		log.Fatal(err)
	}

	var vendor string
	err = device.GetInfo(opencl.DeviceVendor, &vendor)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println()
	printHeader("Using")
	fmt.Println("Platform:", platformName)
	fmt.Println("Vendor:  ", vendor)
}

func main() {
	platforms, err := opencl.GetPlatforms()
	if err != nil {
		log.Fatal(err)
	}

	printHeader("Platforms")

	foundDevice := false

	var platform opencl.Platform
	var device opencl.Device
	var name string
	for _, curPlatform := range platforms {
		err = curPlatform.GetInfo(opencl.PlatformName, &name)
		if err != nil {
			log.Fatal(err)
		}

		var devices []opencl.Device
		devices, err = curPlatform.GetDevices(deviceType)
		if err != nil {
			log.Fatal(err)
		}

		// Use the first available device
		if len(devices) > 0 && !foundDevice {
			var available bool
			err = devices[0].GetInfo(opencl.DeviceAvailable, &available)
			if err == nil && available {
				platform = curPlatform
				device = devices[0]
				foundDevice = true
			}
		}

		version := curPlatform.GetVersion()

		fmt.Printf("Name: %v, devices: %v, version: %v\n", name, len(devices), version)
	}

	if !foundDevice {
		panic("No device found")
	}

	printInfo(platform, device)

	var context opencl.Context
	context, err = device.CreateContext()
	if err != nil {
		log.Fatal(err)
	}
	defer context.Release()

	var commandQueue opencl.CommandQueue
	commandQueue, err = context.CreateCommandQueue(device)
	if err != nil {
		log.Fatal(err)
	}
	defer commandQueue.Release()

	var program opencl.Program
	program, err = context.CreateProgramWithSource(programCode)
	if err != nil {
		log.Fatal(err)
	}
	defer program.Release()

	var logMsg string
	err = program.Build(device, &logMsg)
	if err != nil {
		fmt.Println(logMsg)
		log.Fatal(err)
	}

	kernel, err := program.CreateKernel("kern")
	if err != nil {
		log.Fatal(err)
	}
	defer kernel.Release()

	buffer, err := context.CreateBuffer([]opencl.MemFlags{opencl.MemWriteOnly}, dataSize*4)
	if err != nil {
		log.Fatal(err)
	}
	defer buffer.Release()

	buffer1, err := context.CreateBuffer([]opencl.MemFlags{opencl.MemReadOnly}, dataSize*4)
	if err != nil {
		log.Fatal(err)
	}
	defer buffer1.Release()

	buffer2, err := context.CreateBuffer([]opencl.MemFlags{opencl.MemReadOnly}, dataSize*4)
	if err != nil {
		log.Fatal(err)
	}
	defer buffer2.Release()

	err = kernel.SetArg(0, buffer.Size(), &buffer)
	if err != nil {
		log.Fatal(err)
	}

	err = kernel.SetArg(1, buffer1.Size(), &buffer1)
	if err != nil {
		log.Fatal(err)
	}

	err = kernel.SetArg(2, buffer2.Size(), &buffer2)
	if err != nil {
		log.Fatal(err)
	}

	writeData := make([]float32, dataSize)
	for i := 0; i < dataSize; i++ {
		writeData[i] = float32(i)
	}

	err = commandQueue.EnqueueWriteBuffer(buffer1, true, writeData)
	if err != nil {
		log.Fatal(err)
	}
	err = commandQueue.EnqueueWriteBuffer(buffer2, true, writeData)
	if err != nil {
		log.Fatal(err)
	}

	err = commandQueue.EnqueueNDRangeKernel(kernel, 1, []uint64{dataSize})
	if err != nil {
		log.Fatal(err)
	}

	commandQueue.Flush()
	commandQueue.Finish()

	data := make([]float32, dataSize)

	err = commandQueue.EnqueueReadBuffer(buffer, true, data)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println()
	printHeader("Output")
	for _, item := range data {
		fmt.Printf("%v ", item)
	}
	fmt.Println()
}
