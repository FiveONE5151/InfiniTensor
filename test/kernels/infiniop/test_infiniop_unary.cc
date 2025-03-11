#include "core/graph.h"
#include "core/runtime.h"
#include <iostream>
#include <optional>
#include <ostream>
#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#endif
#include "operators/unary.h"

#include "test.h"

namespace infini {
template <class T>
void testUnaryCpu(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &shape, const DataType &dataType) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shape, dataType);

    auto op = g->addOp<T>(input, nullptr);
    g->dataMalloc();
    input->setData(generator);

    runtime->run(g);
    // op->getOutput()->print();
    // op->getOutput()->printData();
    EXPECT_TRUE(1);
}

void testClipCpu(const std::function<void(void *, size_t, DataType)> &generator,
                 const Shape &shape, const DataType &dataType,
                 std::optional<float> min, std::optional<float> max) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shape, dataType);

    auto op = g->addOp<ClipObj>(input, nullptr, min, max);

    g->dataMalloc();
    input->setData(generator);
    runtime->run(g);

    EXPECT_TRUE(1);
}

#ifdef USE_CUDA
template <class T>
void testUnaryCuda(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &shape, const DataType &dataType) {
    // cpu
    auto cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Graph cpuG = make_ref<GraphObj>(cpuRuntime);
    auto cpuInput = cpuG->addTensor(shape, dataType);

    auto cpuOp = cpuG->addOp<T>(cpuInput, nullptr);
    cpuG->dataMalloc();
    cpuInput->setData(generator);

    cpuRuntime->run(cpuG);
    auto cpuOutput = cpuOp->getOutput();

    // cuda
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    Graph cudaG = make_ref<GraphObj>(cudaRuntime);
    auto cudaInput = cudaG->addTensor(shape, dataType);

    auto cudaOp = cudaG->addOp<T>(cudaInput, nullptr);
    cudaG->dataMalloc();
    cudaInput->setData(generator);

    cudaRuntime->run(cudaG);
    auto cudaOutput = cudaOp->getOutput()->clone(cpuRuntime);

    EXPECT_TRUE(cudaOutput->equalData(cpuOutput));
}
#endif

TEST(ElementWise, Cpu) {
    testUnaryCpu<ReluObj>(IncrementalGenerator(), Shape{1, 2, 2, 3},
                          DataType::Float32);
    testUnaryCpu<ReluObj>(IncrementalGenerator(), Shape{1, 2, 2, 3},
                          DataType::Float16);
}

TEST(Clip, Cpu) {
    testClipCpu(IncrementalGenerator(), Shape{1, 2, 2, 3}, DataType::Float32,
                0.5, 0.8);

    // testClipCpu(IncrementalGenerator(), Shape{1, 2, 2, 3}, DataType::UInt32,
    //             0.5, 0.8);

    testClipCpu(IncrementalGenerator(), Shape{1, 2, 2, 3}, DataType::Float32,
                0.5, std::nullopt);

    testClipCpu(IncrementalGenerator(), Shape{1, 2, 2, 3}, DataType::Float32,
                std::nullopt, 0.8);

    testClipCpu(IncrementalGenerator(), Shape{1, 2, 2, 3}, DataType::Float32,
                std::nullopt, std::nullopt);
}
#ifdef USE_CUDA
TEST(ElementWise, Cuda) {
    testUnaryCuda<ReluObj>(IncrementalGenerator(), Shape{1, 2, 2, 3},
                           DataType::Float32);
    testUnaryCuda<ReluObj>(IncrementalGenerator(), Shape{1, 2, 2, 3},
                           DataType::Float16);
}
#endif

} // namespace infini
