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

template<size_t _SpacetimeDim, typename _T>
using spacetime_vector = math::vector<_SpacetimeDim, _T>;

template<size_t _SpacetimeDim, typename _T>
using spacetime_matrix = math::matrix<_SpacetimeDim, _SpacetimeDim, _T>;

template<size_t _ProjectionDim, typename _T>
using spacetime_point = math::ad_point<_ProjectionDim, _ProjectionDim, _T>;

}
}

#endif//_PHYS_QUANTUM
