#include "operators/where.h"
#include "tensor/tensor_descriptor.h"
#include "utils/infiniop_utils.h"
#include "utils/operator_utils.h"
#include <iostream>

namespace infini {

WhereObj::WhereObj(GraphObj *graph, Tensor inputX, Tensor inputY,
                   Tensor condition, Tensor output)
    : OperatorObj(OpType::Where, TensorVec{inputX, inputY, condition},
                  {output}) {
    IT_ASSERT(checkValid(graph));
}
WhereObj::~WhereObj() {
    if (opDesc) {
        try {
            CHECK_ERROR(infiniopDestroyWhereDescriptor(
                (infiniopWhereDescriptor_t)opDesc));
        } catch (const std::exception &e) {
            std::cerr << "Error in ~WhereObj: " << e.what() << std::endl;
        }
    }
}
optional<vector<Shape>> WhereObj::inferShape(const TensorVec &inputs) {
    auto shapeX = inputs[0]->getDims();
    auto shapeY = inputs[1]->getDims();
    auto shapeCon = inputs[2]->getDims();
    auto retXY = infer_broadcast(shapeX, shapeY);
    auto ret = infer_broadcast(retXY, shapeCon);
    return {{ret}};
}

std::string WhereObj::toString() const {
    std::ostringstream os;
    os << "Where[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[2]->getDims()) << ",";
    os << "inputX=" << inputs[0]->getGuid() << ",";
    os << "inputY=" << inputs[1]->getGuid() << ",";
    os << "condition=" << inputs[2]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> WhereObj::getWorkloadVector() const {
    vector<int> ret = getOutput()->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> WhereObj::getOpAttrVector() const { return {type.underlying()}; }

void WhereObj::initInfiniOp(const Runtime context) {
    auto inputX_dim = inputs[0]->getDims();
    auto inputY_dim = inputs[1]->getDims();
    auto cond_dim = inputs[2]->getDims();
    auto output_dim = outputs[0]->getDims();

    auto inputX_shape = toInfiniopShape(inputX_dim);
    auto inputY_shape = toInfiniopShape(inputY_dim);
    auto cond_shape = toInfiniopShape(cond_dim);
    auto output_shape = toInfiniopShape(output_dim);

    // create tensor descriptor
    infiniopTensorDescriptor_t inputX_tensor;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &inputX_tensor, inputX_dim.size(), inputX_shape.data(), nullptr,
        toInfiniopDataLayout(inputs[0]->getDType().getIndex())));
    infiniopTensorDescriptor_t inputY_tensor;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &inputY_tensor, inputY_dim.size(), inputY_shape.data(), nullptr,
        toInfiniopDataLayout(inputs[1]->getDType().getIndex())));
    infiniopTensorDescriptor_t cond_tensor;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &cond_tensor, cond_dim.size(), cond_shape.data(), nullptr,
        toInfiniopDataLayout(inputs[2]->getDTypeIndex())));
    infiniopTensorDescriptor_t output_tensor;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &output_tensor, output_dim.size(), output_shape.data(), nullptr,
        toInfiniopDataLayout(outputs[0]->getDTypeIndex())));

    // create op descriptor
    CHECK_ERROR(infiniopCreateWhereDescriptor(
        context->opHandle(), (infiniopWhereDescriptor_t *)&opDesc,
        inputX_tensor, inputY_tensor, cond_tensor, output_tensor));

    // destroy tensor descriptor
    CHECK_ERROR(infiniopDestroyTensorDescriptor(inputX_tensor));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(inputY_tensor));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(cond_tensor));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(output_tensor));
}
} // namespace infini
