#ifndef _PHYS_QUANTUM
#define _PHYS_QUANTUM

#include <math/auto_diff.h>
#include <math/algebra.h>

#include <phys/constant.h>

/**
 * 量子力学相关定义。
 * 包含波动力学和矩阵力学
 */
namespace phys
{
namespace quantum
{

/**
 * @brief 根据时空维度计算空间维度
 */
constexpr size_t space_dimension(size_t spacetime_dimension)
{
	return spacetime_dimension - 1;
}

constexpr size_t spacetime_dimension(size_t space_dimension)
{
	return space_dimension + 1;
}

template<typename _T, size_t _SpacetimeDim>
using spacetime_vector = math::vector<_T, _SpacetimeDim>;

template<typename _T, size_t _SpacetimeDim>
using spacetime_matrix = math::matrix<_T, _SpacetimeDim, _SpacetimeDim>;

template<typename _T, size_t _ProjectionDim>
using spacetime_point = math::ad_point<_T, _ProjectionDim, _ProjectionDim>;

}
}

#endif//_PHYS_QUANTUM
