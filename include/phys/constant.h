#ifndef _PHYS_CONSTANT
#define _PHYS_CONSTANT

#include <math/complex.h>

/**
 * 数学常数定义
 */
namespace phys
{
/**
 * @brief 高精度原始数值
 */
struct __constant_float128
{
	/**
	 * @brief 普朗克常数.2019年SI新定义后为精确有理数
	 */
	static constexpr __float128 h = 6.62607015e-34;

	/**
	 * @brief 约化普朗克常数ℏ=h/2π，无理数。此值为CODATA推荐精度的有理数值
	 */
	static constexpr __float128 hbar = 1.054571817e-34;

	/**
	 * @brief iℏ
	 */
	static constexpr math::complex<__float128> ihbar = math::iz(hbar);

	static constexpr __float128 _2pihbar = h;
};

/**
 * @brief 根据模板精度转换
 */
template<typename _T>
struct constant
{
	static constexpr _T h = (_T)__constant_float128::h;

	static constexpr _T hbar = (_T)__constant_float128::hbar;

	static constexpr math::complex<_T> ihbar = math::iz(hbar);

	static constexpr _T _2pihbar = h;
};
}

#endif//_PHYS_CONSTANT
