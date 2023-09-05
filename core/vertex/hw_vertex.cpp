// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

#ifdef __IPU__
// Use the IPU intrinsics
#include <ipu_memory_intrinsics>
#include <ipu_vector_math>
#define NAMESPACE ipu
#else
// Use the std functions
#include <cmath>
#define NAMESPACE std
#endif

#ifdef __IPU__
#define SUPERVISOR_TARGET __attribute__((target("supervisor")))
#else
#define SUPERVISOR_TARGET
#endif

using namespace poplar;

/**
 * @brief Cycle count vertex, with a data barrier.
 *
 * NOTE: the data barrier mechanism is an attempt to avoid XLA and Poplar to
 * re-organize too much the program and as a consequence get inacurrate cycle
 * count measurements. It is by far not perfect and 100% reliable (e.g. comms
 * can still be inserted in the middle), but probably good enough for most
 * cases.
 *
 * TODO: support multiple data barrier in/out vectors.
 *
 * @tparam T Type of the data barrier.
 */
template <typename T>
class CycleCountBarrier : public SupervisorVertex {
  static const bool needsAlignWorkers = false;

 public:
  InOut<Vector<T, VectorLayout::ONE_PTR>> data;
  Output<Vector<unsigned, VectorLayout::ONE_PTR>> out;

  SUPERVISOR_TARGET bool compute() {
#ifdef __IPU__
    out[0] = __builtin_ipu_get_scount_l();
    out[1] = __builtin_ipu_get_scount_u();
#else
    out[0] = 0;
    out[1] = 0;
#endif
    return true;
  }
};

// explicit instantiations
template class CycleCountBarrier<bool>;
template class CycleCountBarrier<unsigned char>;
template class CycleCountBarrier<signed char>;
template class CycleCountBarrier<unsigned short>;
template class CycleCountBarrier<short>;
template class CycleCountBarrier<unsigned>;
template class CycleCountBarrier<int>;
template class CycleCountBarrier<float>;
template class CycleCountBarrier<half>;
