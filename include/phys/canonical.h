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
template<typename _T, size_t _PhaseSpaceDim>
using phase_vector = math::vector<_T, _PhaseSpaceDim>;

template<typename _T, size_t _PhaseSpaceDim>
using phase_matrix = math::matrix<_T, _PhaseSpaceDim, _PhaseSpaceDim>;

/**
 * @brief 相空间中的点，输出维度、输入维度相等。
 * @param _PhaseSpaceDim 相空间维数
 * @param _T 数值类型
 */
template<typename _T, size_t _PhaseSpaceDim>
using phase_point = math::ad_point<_T, 1, _PhaseSpaceDim, _PhaseSpaceDim>;

/**
 * @brief _PhaseSpaceDim维相空间系统哈密顿量H，标量函数
 * 		  H(p, q)为2N或2N+1元标量函数
 */
template<typename _T, size_t _PhaseSpaceDim>
using Hamiltonian = math::ad_point<_T, 0, _PhaseSpaceDim>;

#define __assert_phase_space_dim__(dim) static_assert(dim >= 2, "phase space dimension is not lower than 2")

template<typename _T, size_t _PhaseSpaceDim>
inline math::vector<_T, degree_of_freedom(_PhaseSpaceDim)>& q(math::tensor<_T, _PhaseSpaceDim>& qp)
{
	__assert_phase_space_dim__(_PhaseSpaceDim);
	constexpr size_t dof = degree_of_freedom(_PhaseSpaceDim);
	return qp.template slice < dof > (0);
}

template<typename _T, size_t _PhaseSpaceDim>
inline const math::vector<_T, degree_of_freedom(_PhaseSpaceDim)>& q(const math::tensor<_T, _PhaseSpaceDim>& qp)
{
	__assert_phase_space_dim__(_PhaseSpaceDim);
	constexpr size_t dof = degree_of_freedom(_PhaseSpaceDim);
	return qp.template slice < dof > (0);
}

template<typename _T, size_t _PhaseSpaceDim>
inline math::vector<_T, degree_of_freedom(_PhaseSpaceDim)>& p(math::tensor<_T, _PhaseSpaceDim>& qp)
{
	__assert_phase_space_dim__(_PhaseSpaceDim);
	constexpr size_t dof = degree_of_freedom(_PhaseSpaceDim);
	return qp.template slice < dof > (dof);
}

template<typename _T, size_t _PhaseSpaceDim>
inline const math::vector<_T, degree_of_freedom(_PhaseSpaceDim)>& p(const math::tensor<_T, _PhaseSpaceDim>& qp)
{
	__assert_phase_space_dim__(_PhaseSpaceDim);
	constexpr size_t dof = degree_of_freedom(_PhaseSpaceDim);
	return qp.template slice < dof > (dof);
}

template<typename _T, size_t _PhaseSpaceDim>
inline _T& t(const math::tensor<_T, _PhaseSpaceDim>& qp)
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
template<typename _T, size_t _PhaseSpaceDim>
inline phase_point<_T, _PhaseSpaceDim> xi(const math::tensor<_T, _PhaseSpaceDim>& qp)
{
	__assert_phase_space_dim__(_PhaseSpaceDim);
	return phase_point<_T, _PhaseSpaceDim>(qp, math::identity<phase_matrix<_T, _PhaseSpaceDim>, math::mul>());
}

/**
 * @brief 提取广义坐标
 */
template<typename _T, size_t _PhaseSpaceDim>
inline phase_point<_T, degree_of_freedom(_PhaseSpaceDim)> q(const math::ad_point<_T, 1, _PhaseSpaceDim, _PhaseSpaceDim>& qp)
{
	__assert_phase_space_dim__(_PhaseSpaceDim);
	constexpr size_t dof = degree_of_freedom(_PhaseSpaceDim);
	return math::slice < dof > (qp, 0);
}

template<typename _T, size_t, size_t _PhaseSpaceDim>
inline phase_point<_T, 1> qi(const math::ad_point<_T, 1, _PhaseSpaceDim, _PhaseSpaceDim>& qp, size_t i)
{
	__assert_phase_space_dim__(_PhaseSpaceDim);
	return math::slice < 1 > (qp, i);
}

/**
 * @brief 提取广义动量
 */
template<typename _T, size_t _PhaseSpaceDim>
inline phase_point<_T, degree_of_freedom(_PhaseSpaceDim)> p(const math::ad_point<_T, 1, _PhaseSpaceDim, _PhaseSpaceDim>& qp)
{
	__assert_phase_space_dim__(_PhaseSpaceDim);
	constexpr size_t dof = degree_of_freedom(_PhaseSpaceDim);
	return math::slice < dof > (qp, dof);
}

template<typename _T, size_t, size_t _PhaseSpaceDim>
inline phase_point<_T, 1> pi(const math::ad_point<_T, 1, _PhaseSpaceDim, _PhaseSpaceDim>& qp, size_t i)
{
	__assert_phase_space_dim__(_PhaseSpaceDim);
	constexpr size_t dof = degree_of_freedom(_PhaseSpaceDim);
	return math::slice < 1 > (qp, dof + i);
}

/**
 * @brief 提取时间
 */
template<typename _T, size_t _PhaseSpaceDim>
inline phase_point<_T, 1> t(const math::ad_point<_T, 1, _PhaseSpaceDim, _PhaseSpaceDim>& qp)
{
	__assert_phase_space_dim__(_PhaseSpaceDim);
	static_assert(_PhaseSpaceDim % 2, "phase space is not time dependent");
	return math::slice < 1 > (qp, _PhaseSpaceDim - 1);
}

/**
 * @brief 广义速度。
 * 		  根据正则方程导出：dq/dt=∂H/∂p
 */
template<typename _T, size_t _PhaseSpaceDim>
inline math::vector<_T, degree_of_freedom(_PhaseSpaceDim)> dq_dt(const math::ad_point<_T, 0, _PhaseSpaceDim>& H)
{
	constexpr size_t dof = degree_of_freedom(_PhaseSpaceDim);
	return H.derivative.template slice < dof > (dof);
}

template<typename _T, size_t _PhaseSpaceDim>
inline _T dq_dt(const math::ad_point<_T, 0, _PhaseSpaceDim>& H, size_t i)
{
	constexpr size_t dof = degree_of_freedom(_PhaseSpaceDim);
	return H.derivative[dof + i];
}

/**
 * @brief 广义力。
 * 		  根据正则方程导出：dp/dt=-∂H/∂qyf
 */
template<typename _T, size_t _PhaseSpaceDim>
inline math::vector<_T, degree_of_freedom(_PhaseSpaceDim)> dp_dt(const math::ad_point<_T, 0, _PhaseSpaceDim>& H)
{
	constexpr size_t dof = degree_of_freedom(_PhaseSpaceDim);
	return -H.derivative.template slice < dof > (0);
}

template<typename _T, size_t _PhaseSpaceDim>
inline _T dp_dt(const math::ad_point<_T, 0, _PhaseSpaceDim>& H, size_t i)
{
	return -H.derivative[i];
}

/**
 * @brief 构建动能。将H置于右边，左边的偏导数全补0
 * TODO 时间维总是应该在最右侧
 */
template<size_t _PhaseSpaceDim, typename _T, size_t _DOF>
inline Hamiltonian<_T, _PhaseSpaceDim> Hp(const math::ad_point<_T, 0, _DOF>& H)
{
	__assert_phase_space_dim__(_PhaseSpaceDim);
	return math::cat_derivative(math::matrix<_T, 1, _PhaseSpaceDim - _DOF>::zero(), H);
}

/**
 * @brief 构建势能。将H置于左边，右边的偏导数全补0
 */
template<size_t _PhaseSpaceDim, typename _T, size_t _DOF>
inline Hamiltonian<_T, _PhaseSpaceDim> Hq(const math::ad_point<_T, 0, _DOF>& H)
{
	__assert_phase_space_dim__(_PhaseSpaceDim);
	return math::cat_derivative(H, math::matrix<_T, 1, _PhaseSpaceDim - _DOF>::zero());
}

/**
 * @brief 用于数值积分的dq/dt、dp/dt计算函数
 */
template<typename _T, size_t _PhaseSpaceDim, typename _HamiltonianType>
struct F_qp
{
	/**
	 * 如果H参数类型不是phase_point<_T, _PhaseSpaceDim>，会报错类似
	 * error: invalid initialization of reference of type 'const math::ad_point<double, 1, 6, 6>&' from expression of type 'tplmp::__decl_impl_base::__decl_impl<XXX>::type'
	 */
	static_assert(
			tplmp::type_equal<
			typename tplmp::callable_ret<_HamiltonianType, phase_point<_T, _PhaseSpaceDim> >::type,
			Hamiltonian<_T, _PhaseSpaceDim>
			>::value,
			"auto diff Hamiltonian expr expected");

	_HamiltonianType H;

	/**
	 * H的函数签名为Hamiltonian<_T, _PhaseSpaceDim> H(const phase_point<_T, _PhaseSpaceDim> &qp)
	 */
	inline F_qp(_HamiltonianType&& H_expr) :
			H(tplmp::forward < _HamiltonianType > (H_expr))
	{
	}

	inline math::vector<_T, _PhaseSpaceDim> operator()(_T t, const math::vector<_T, _PhaseSpaceDim>& qp) const
	{
		Hamiltonian<_T, _PhaseSpaceDim> _H = H(xi(qp));
		math::vector<_T, _PhaseSpaceDim> F_qp; //自动微分计算dq/dt、 dp/dt
		q(F_qp) = dq_dt(_H);
		p(F_qp) = dp_dt(_H);
		return F_qp;
	}
};

/**
 * @brief 自由粒子动能
 */
template<typename _T, size_t _PhaseSpaceDim>
inline Hamiltonian<_T, _PhaseSpaceDim> T_free_particle(_T k, const math::ad_point<_T, 1, _PhaseSpaceDim, _PhaseSpaceDim>& qp)
{
	phase_point<_T, degree_of_freedom(_PhaseSpaceDim)> _p = p(qp);
	return Hp<_PhaseSpaceDim, _T>((_p * _p) * k);
}

/**
 * @brief 谐振子势
 * @param eq_pos 平衡位置
 */
template<typename _T, size_t _PhaseSpaceDim>
inline Hamiltonian<_T, _PhaseSpaceDim> V_harmonic_oscillator(_T k, const math::ad_point<_T, 1, _PhaseSpaceDim, _PhaseSpaceDim>& qp,
		const math::vector<_T, degree_of_freedom(_PhaseSpaceDim)>& eq_pos = math::vector<_T, degree_of_freedom(_PhaseSpaceDim)>::zero())
{
	phase_point<_T, degree_of_freedom(_PhaseSpaceDim)> _q = q(qp);
	_q.value -= eq_pos;
	return Hq<_PhaseSpaceDim, _T>((_q * _q) * k);
}

/**
 * @brief 库伦势/非相对论引力势
 */
template<typename _T, size_t _PhaseSpaceDim>
inline Hamiltonian<_T, _PhaseSpaceDim> V_coulomb(_T k, const math::ad_point<_T, 1, _PhaseSpaceDim, _PhaseSpaceDim>& qp,
		const math::vector<_T, degree_of_freedom(_PhaseSpaceDim)>& source = math::vector<_T, degree_of_freedom(_PhaseSpaceDim)>::zero())
{
	phase_point<_T, degree_of_freedom(_PhaseSpaceDim)> _q = q(qp);
	_q.value -= source;
	return Hq<_PhaseSpaceDim, _T>(-k / math::frobenius_norm(_q));
}

/**
 * @brief 反平方势
 */
template<typename _T, size_t _PhaseSpaceDim>
inline Hamiltonian<_T, _PhaseSpaceDim> V_inverse_square(_T k, const math::ad_point<_T, 1, _PhaseSpaceDim, _PhaseSpaceDim>& qp,
		const math::vector<_T, degree_of_freedom(_PhaseSpaceDim)>& source = math::vector<_T, degree_of_freedom(_PhaseSpaceDim)>::zero())
{
	phase_point<_T, degree_of_freedom(_PhaseSpaceDim)> _q = q(qp);
	_q.value -= source;
	return Hq<_PhaseSpaceDim, _T>(-k / (_q * _q));
}

}
}

#endif//_PHYS_CANONICAL
