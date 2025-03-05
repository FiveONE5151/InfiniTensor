#include "core/graph.h"
#include "core/runtime.h"
#include "core/tensor.h"
#include "utils/data_generator.h"

#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#endif
#include "operators/where.h"

#include "test.h"

namespace infini {

void testWhereCpu(
    const std::function<void(void *, size_t, DataType)> &input_generator,
    const std::function<void(void *, size_t, DataType)> &cond_generator,
    const Shape &shapeCon, const Shape &shapeX, const Shape &shapeY,
    const DataType &dataType) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto inputX = g->addTensor(shapeX, dataType);
    auto inputY = g->addTensor(shapeY, dataType);
    auto condition = g->addTensor(shapeCon, DataType::Float32);
    auto op = g->addOp<WhereObj>(inputX, inputY, condition, nullptr);

    g->dataMalloc();
    inputX->setData(input_generator);
    inputY->setData(input_generator);
    condition->setData(cond_generator);

    runtime->run(g);

    EXPECT_TRUE(1);
}

TEST(Where, Cpu) {
    testWhereCpu(IncrementalGenerator(), OneGenerator(), Shape{1, 2, 2, 3},
                 Shape{1, 2, 2, 3}, Shape{1, 2, 2, 3}, DataType::Float32);
}

} // namespace infini
