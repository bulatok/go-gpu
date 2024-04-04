package main

import (
	"flag"
	"fmt"
	"github.com/hupe1980/go-mtl"
	"log"
	"math/rand"
	"sync"
	"time"
	"unsafe"
)

// source msl language
const source = `#include <metal_stdlib>

using namespace metal;

kernel void dot_product(device const float* inA,
                        device const float* inB,
                        device float* result,
                        uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] * inB[index];
}
`

func dotProductGPU(dataA, dataB []float32) float32 {
	// Create a Metal device.
	device, err := mtl.CreateSystemDefaultDevice()
	if err != nil {
		log.Fatal(err)
	}

	// Create a Metal library from the provided source code.
	lib, err := device.NewLibraryWithSource(source)
	if err != nil {
		log.Fatal(err)
	}

	// Retrieve the Metal function named "dot_product" from the library.
	dotProduct, err := lib.NewFunctionWithName("dot_product")
	if err != nil {
		log.Fatal(err)
	}

	// Create a Metal compute pipeline state with the function.
	pipelineState, err := device.NewComputePipelineStateWithFunction(dotProduct)
	if err != nil {
		log.Fatal(err)
	}

	// Create a Metal command queue to submit commands for execution.
	q := device.NewCommandQueue()

	// Set the length of the arrays.
	arrLen := uint(4)

	// Create Metal buffers for input and output data.
	// b1 and b2 represent the input arrays, and r represents the output array.
	b1 := device.NewBufferWithBytes(unsafe.Pointer(&dataA[0]), unsafe.Sizeof(dataA), mtl.ResourceStorageModeShared)
	b2 := device.NewBufferWithBytes(unsafe.Pointer(&dataB[0]), unsafe.Sizeof(dataB), mtl.ResourceStorageModeShared)
	r := device.NewBufferWithLength(unsafe.Sizeof(arrLen), mtl.ResourceStorageModeShared)

	// Create a Metal command buffer to encode and execute commands.
	cb := q.CommandBuffer()

	// Create a compute command encoder to encode compute commands.
	cce := cb.ComputeCommandEncoder()

	// Set the compute pipeline state to specify the function to be executed.
	cce.SetComputePipelineState(pipelineState)

	// Set the input and output buffers for the compute function.
	cce.SetBuffer(b1, 0, 0)
	cce.SetBuffer(b2, 0, 1)
	cce.SetBuffer(r, 0, 2)

	// Specify threadgroup size
	tgs := pipelineState.MaxTotalThreadsPerThreadgroup
	if tgs > arrLen {
		tgs = arrLen
	}

	// Dispatch compute threads to perform the calculation.
	cce.DispatchThreads(mtl.Size{Width: arrLen, Height: 1, Depth: 1}, mtl.Size{Width: tgs, Height: 1, Depth: 1})

	// End encoding the compute command.
	cce.EndEncoding()

	// Commit the command buffer for execution.
	cb.Commit()

	// Wait until the command buffer execution is completed.
	cb.WaitUntilCompleted()

	// Read the results from the output buffer
	result := (*[1 << 30]float32)(r.Contents())[:arrLen]

	// Calculate the dot product
	var dotProductResult float32
	for _, value := range result {
		dotProductResult += value
	}

	return dotProductResult
}

func dotProductNaive(s1, s2 []float32) float32 {
	var result float32
	for i := range s1 {
		result += s1[i] * s2[i]
	}

	return result
}

func dotProductCPU(s1, s2 []float32, gn int) float32 {
	size := len(s1)
	if size <= gn {
		gn = size
	}

	var result float32

	chunkSize := (size + gn - 1) / gn

	var wg sync.WaitGroup

	results := make(map[int]float32, gn)
	mx := sync.Mutex{}
	for i := 0; i < gn; i++ {
		start := i * chunkSize
		end := (i + 1) * chunkSize
		if end > size {
			end = size
		}
		i := i
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			var partialResult float32
			for j := start; j < end; j++ {
				partialResult += s1[j] * s2[j]
			}
			mx.Lock()
			results[i] = partialResult
			mx.Unlock()
		}(start, end)
	}

	wg.Wait()

	for _, res := range results {
		result += res
	}
	return result
}

func randomVectors(size int) ([]float32, []float32) {
	var exp float32 = 10
	s1 := make([]float32, 0, size)
	s2 := make([]float32, 0, size)

	for i := 0; i < size; i++ {
		s1 = append(s1, rand.Float32()*exp)
		s2 = append(s2, rand.Float32()*exp)
	}

	return s1, s2
}

func withTiming(name string, f func()) {
	s := time.Now()
	f()
	fmt.Printf("%s (%v sec)\n", name, time.Now().Sub(s).Seconds())
}

// make metal && ./metal -n 1000000000
func main() {
	n := flag.Int("n", 10, "vectors size")
	flag.Parse()

	s1, s2 := randomVectors(*n)

	withTiming("dotProductNaive", func() {
		dotProductNaive(s1, s2)
	})

	withTiming("dotProductCPU", func() {
		dotProductCPU(s1, s2, 200)
	})

	withTiming("dotProductGPU", func() {
		dotProductGPU(s1, s2)
	})
}
