#ifndef _PHYS_CANONICAL
#define _PHYS_CANONICAL

#include <math/auto_diff.h>
#include <math/algebra.h>
#include <math/integral.h>

/**
 * 正则方程相关定义。
 * 如广义坐标、广义动量、正则方程等。
 */
namespace phys
{
namespace canonical
{
/**
 * @brief 相空间的维数
 * @param s 自由度
 * @param time_dependent H是否是含时的
 */
constexpr size_t phase_space_dim(size_t s, bool time_dependent)
{
	return time_dependent ? (2 * s + 1) : (2 * s);
}

/**
 * @brief 根据相空间的维数计算系统自由度
 */
constexpr size_t degree_of_freedom(size_t phase_space_dim)
{
	return phase_space_dim / 2;
}

template<size_t _PhaseSpaceDim, typename _T>
using phase_vector = math::vector<_PhaseSpaceDim, _T>;

template<size_t _PhaseSpaceDim, typename _T>
using phase_matrix = math::matrix<_PhaseSpaceDim, _PhaseSpaceDim, _T>;

/**
 * @brief 相空间中的点，输出维度、输入维度相等。
 * @param _PhaseSpaceDim 相空间维数
 * @param _T 数值类型
 */
template<size_t _PhaseSpaceDim, typename _T>
using phase_point = math::ad_point<_PhaseSpaceDim, _PhaseSpaceDim, _T>;

/**
 * @brief _PhaseSpaceDim维相空间系统哈密顿量H
 * 		  H(p, q)为2N或2N+1元标量函数
 */
template<size_t _PhaseSpaceDim, typename _T>
using Hamiltonian = math::ad_point<1, _PhaseSpaceDim, _T>;

#define __assert_phase_space_dim__(dim) static_assert(dim >= 2, "phase space dimension is not lower than 2")

template<size_t _PhaseSpaceDim, typename _T>
inline math::vector<degree_of_freedom(_PhaseSpaceDim), _T>& q(math::vector<_PhaseSpaceDim, _T>& qp)
{
	__assert_phase_space_dim__(_PhaseSpaceDim);
	constexpr size_t dof = degree_of_freedom(_PhaseSpaceDim);
	return qp.template slice < dof > (0);
}

template<size_t _PhaseSpaceDim, typename _T>
inline const math::vector<degree_of_freedom(_PhaseSpaceDim), _T>& q(const math::vector<_PhaseSpaceDim, _T>& qp)
{
	__assert_phase_space_dim__(_PhaseSpaceDim);
	constexpr size_t dof = degree_of_freedom(_PhaseSpaceDim);
	return qp.template slice < dof > (0);
}

template<size_t _PhaseSpaceDim, typename _T>
inline math::vector<degree_of_freedom(_PhaseSpaceDim), _T>& p(math::vector<_PhaseSpaceDim, _T>& qp)
{
	__assert_phase_space_dim__(_PhaseSpaceDim);
	constexpr size_t dof = degree_of_freedom(_PhaseSpaceDim);
	return qp.template slice < dof > (dof);
}

template<size_t _PhaseSpaceDim, typename _T>
inline const math::vector<degree_of_freedom(_PhaseSpaceDim), _T>& p(const math::vector<_PhaseSpaceDim, _T>& qp)
{
	__assert_phase_space_dim__(_PhaseSpaceDim);
	constexpr size_t dof = degree_of_freedom(_PhaseSpaceDim);
	return qp.template slice < dof > (dof);
}

template<size_t _PhaseSpaceDim, typename _T>
inline _T& t(const math::vector<_PhaseSpaceDim, _T>& qp)
{
	__assert_phase_space_dim__(_PhaseSpaceDim);
	return qp[_PhaseSpaceDim - 1];
}

/**
 * @brief N自由度系统的广义坐标、广义动量初始值
 * 		  ξ(q, p)对应Jacobi矩阵是恒等矩阵
 * 		  [q] [Is 0s]
 * 		  [p] [0s Is]
 * 		  其中Is、0s分别为s维恒等矩阵、零矩阵。
 */
template<size_t _PhaseSpaceDim, typename _T>
inline phase_point<_PhaseSpaceDim, _T> xi(const math::vector<_PhaseSpaceDim, _T>& qp)
{
	__assert_phase_space_dim__(_PhaseSpaceDim);
	return phase_point<_PhaseSpaceDim, _T>(qp, math::identity<phase_matrix<_PhaseSpaceDim, _T>, math::mul>());
}

/**
 * @brief 提取广义坐标
 */
template<size_t _PhaseSpaceDim, typename _T>
inline phase_point<degree_of_freedom(_PhaseSpaceDim), _T> q(const math::ad_point<_PhaseSpaceDim, _PhaseSpaceDim, _T>& qp)
{
	__assert_phase_space_dim__(_PhaseSpaceDim);
	constexpr size_t dof = degree_of_freedom(_PhaseSpaceDim);
	return qp.template slice < dof > (0);
}

template<size_t, size_t _PhaseSpaceDim, typename _T>
inline phase_point<1, _T> qi(const math::ad_point<_PhaseSpaceDim, _PhaseSpaceDim, _T>& qp, size_t i)
{
	__assert_phase_space_dim__(_PhaseSpaceDim);
	return qp.template slice < 1 > (i);
}

/**
 * @brief 提取广义动量
 */
template<size_t _PhaseSpaceDim, typename _T>
inline phase_point<degree_of_freedom(_PhaseSpaceDim), _T> p(const math::ad_point<_PhaseSpaceDim, _PhaseSpaceDim, _T>& qp)
{
	__assert_phase_space_dim__(_PhaseSpaceDim);
	constexpr size_t dof = degree_of_freedom(_PhaseSpaceDim);
	return qp.template slice < dof > (dof);
}

template<size_t, size_t _PhaseSpaceDim, typename _T>
inline phase_point<1, _T> pi(const math::ad_point<_PhaseSpaceDim, _PhaseSpaceDim, _T>& qp, size_t i)
{
	__assert_phase_space_dim__(_PhaseSpaceDim);
	constexpr size_t dof = degree_of_freedom(_PhaseSpaceDim);
	return qp.template slice < 1 > (dof + i);
}

/**
 * @brief 提取时间
 */
template<size_t _PhaseSpaceDim, typename _T>
inline phase_point<1, _T> t(const math::ad_point<_PhaseSpaceDim, _PhaseSpaceDim, _T>& qp)
{
	__assert_phase_space_dim__(_PhaseSpaceDim);
	static_assert(_PhaseSpaceDim % 2, "phase space is not time dependent");
	return qp.template slice < 1 > (_PhaseSpaceDim - 1);
}

/**
 * @brief 广义速度。
 * 		  根据正则方程导出：dq/dt=∂H/∂p
 */
template<size_t _PhaseSpaceDim, typename _T>
inline math::vector<degree_of_freedom(_PhaseSpaceDim), _T> dq_dt(const math::ad_point<1, _PhaseSpaceDim, _T>& H)
{
	constexpr size_t dof = degree_of_freedom(_PhaseSpaceDim);
	return H.derivative[0].template slice < dof > (dof);
}

template<size_t _PhaseSpaceDim, typename _T>
inline _T dq_dt(const math::ad_point<1, _PhaseSpaceDim, _T>& H, size_t i)
{
	constexpr size_t dof = degree_of_freedom(_PhaseSpaceDim);
	return H.derivative[0][dof + i];
}

/**
 * @brief 广义力。
 * 		  根据正则方程导出：dp/dt=-∂H/∂q
 */
template<size_t _PhaseSpaceDim, typename _T>
inline math::vector<degree_of_freedom(_PhaseSpaceDim), _T> dp_dt(const math::ad_point<1, _PhaseSpaceDim, _T>& H)
{
	constexpr size_t dof = degree_of_freedom(_PhaseSpaceDim);
	return -H.derivative[0].template slice < dof > (0);
}

template<size_t _PhaseSpaceDim, typename _T>
inline _T dp_dt(const math::ad_point<1, _PhaseSpaceDim, _T>& H, size_t i)
{
	return -H.derivative[0][i];
}

/**
 * @brief 构建动能。将H置于右边，左边的偏导数全补0
 * TODO 时间维总是应该在最右侧
 */
template<size_t _PhaseSpaceDim, size_t _DOF, typename _T>
inline Hamiltonian<_PhaseSpaceDim, _T> Hp(const math::ad_point<1, _DOF, _T>& H)
{
	__assert_phase_space_dim__(_PhaseSpaceDim);
	return math::cat_derivative(math::matrix<1, _PhaseSpaceDim - _DOF, _T>::zero(), H);
}

/**
 * @brief 构建势能。将H置于左边，右边的偏导数全补0
 */
template<size_t _PhaseSpaceDim, size_t _DOF, typename _T>
inline Hamiltonian<_PhaseSpaceDim, _T> Hq(const math::ad_point<1, _DOF, _T>& H)
{
	__assert_phase_space_dim__(_PhaseSpaceDim);
	return math::cat_derivative(H, math::matrix<1, _PhaseSpaceDim - _DOF, _T>::zero());
}

/**
 * @brief 用于数值积分的dq/dt、dp/dt计算函数
 */
template<size_t _PhaseSpaceDim, typename _T, typename _HamiltonianType>
struct F_qp
{
	static_assert(
			tplmp::type_equal<
			decltype(tplmp::decl<_HamiltonianType>::val()(tplmp::decl<phase_vector<_PhaseSpaceDim, _T> >::val())),
			Hamiltonian<_PhaseSpaceDim, _T>
			>::value,
			"auto diff Hamiltonian expr expected");

	_HamiltonianType H;

	/**
	 * H的函数签名为Hamiltonian<_PhaseSpaceDim, _T> H(const phase_point<_PhaseSpaceDim, _T> &qp)
	 */
	inline F_qp(_HamiltonianType&& H_expr) :
			H(tplmp::forward < _HamiltonianType > (H_expr))
	{
	}

	inline math::vector<_PhaseSpaceDim, _T> operator()(_T t, const math::vector<_PhaseSpaceDim, _T>& qp) const
	{
		Hamiltonian<_PhaseSpaceDim, _T> _H = H(xi(qp));
		math::vector<_PhaseSpaceDim, _T> F_qp; //自动微分计算dq/dt、 dp/dt
		q(F_qp) = dq_dt(_H);
		p(F_qp) = dp_dt(_H);
		return F_qp;
	}
};

/**
 * @brief 自由粒子动能
 */
template<size_t _PhaseSpaceDim, typename _T>
inline Hamiltonian<_PhaseSpaceDim, _T> T_free_particle(_T k, const math::ad_point<_PhaseSpaceDim, _PhaseSpaceDim, _T>& qp)
{
	phase_point<degree_of_freedom(_PhaseSpaceDim), _T> _p = p(qp);
	return Hp<_PhaseSpaceDim>((_p * _p) * k);
}

/**
 * @brief 谐振子势
 */
template<size_t _PhaseSpaceDim, typename _T>
inline Hamiltonian<_PhaseSpaceDim, _T> V_harmonic_oscillator(_T k, const math::ad_point<_PhaseSpaceDim, _PhaseSpaceDim, _T>& qp)
{
	phase_point<degree_of_freedom(_PhaseSpaceDim), _T> _q = q(qp);
	return Hq<_PhaseSpaceDim>((_q * _q) * k);
}

/**
 * @brief 库伦势/引力势
 */
template<size_t _PhaseSpaceDim, typename _T>
inline Hamiltonian<_PhaseSpaceDim, _T> V_coulomb(_T k, const math::ad_point<_PhaseSpaceDim, _PhaseSpaceDim, _T>& qp)
{
	return Hq<_PhaseSpaceDim>(-k / q(qp).norm());
}

/**
 * @brief 反平方势
 */
template<size_t _PhaseSpaceDim, typename _T>
inline Hamiltonian<_PhaseSpaceDim, _T> V_inverse_square(_T k, const math::ad_point<_PhaseSpaceDim, _PhaseSpaceDim, _T>& qp)
{
	phase_point<degree_of_freedom(_PhaseSpaceDim), _T> _q = q(qp);
	return Hq<_PhaseSpaceDim>(-k / (_q * _q));
}

}
}

#endif//_PHYS_CANONICAL
