#include "extended_kalman_filter.hpp"

#include <numeric>


namespace tools
{
ExtendedKalmanFilter::ExtendedKalmanFilter(
  const Eigen::VectorXd & x0, const Eigen::MatrixXd & P0,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add)
: x(x0), P(P0), I(Eigen::MatrixXd::Identity(x0.rows(), x0.rows())), x_add(x_add)
{
  data["residual_yaw"] = 0.0;
  data["residual_pitch"] = 0.0;
  data["residual_distance"] = 0.0;
  data["residual_angle"] = 0.0;
      // 初始化各种残差，残差是模型预测值和实际值的偏差，判断拟合效果
  data["nis"] = 0.0;
  data["nees"] = 0.0;
  data["nis_fail"] = 0.0;
  data["nees_fail"] = 0.0;
  data["recent_nis_failures"] = 0.0;
}

Eigen::VectorXd ExtendedKalmanFilter::predict(const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q)//传入了线性状态转移矩阵F（默认状态转移函数为x->F*x），Q为过程噪声协方差
{
  return predict(F, Q, [&](const Eigen::VectorXd & x) { return F * x; });  //lamba表达式，调用重载函数
}

Eigen::VectorXd ExtendedKalmanFilter::predict(
  const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f)    //重载函数，多穿入了一个非线性状态转移函数f
{
  P = F * P * F.transpose() + Q;    //P是预测协方差
  x = f(x);   //预测状态
  return x;
}

//此处只传线性化后雅可比矩阵或者本就是线性的H，h，F，f，用雅可比矩阵计算函数求线性化的状态转移矩阵和观测矩阵不在此处
Eigen::VectorXd ExtendedKalmanFilter::update(
  const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)//依次是观测值z，线性化后的观测矩阵H，测量噪声协方差R，
{
  return update(z, H, R, [&](const Eigen::VectorXd & x) { return H * x; }, z_subtract); //调用重载函数，用线性话后的H求出线性化后的观测函数h
}

Eigen::VectorXd ExtendedKalmanFilter::update(
  const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> h,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)//z_subtract：测量残差减法函数：计算测量值z与预测测量值h(x)的差（残差），即residual = z_subtract(z, h(x))。特殊用途：处理角度等 “循环量” 的减法
{
  Eigen::VectorXd x_prior = x;
  Eigen::MatrixXd K = P * H.transpose() * (H * P * H.transpose() + R).inverse();

  // Stable Compution of the Posterior Covariance
  // https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb
  P = (I - K * H) * P * (I - K * H).transpose() + K * R * K.transpose();

  x = x_add(x, K * z_subtract(z, h(x)));

  /// 卡方检验
  Eigen::VectorXd residual = z_subtract(z, h(x));
  // 新增检验
  Eigen::MatrixXd S = H * P * H.transpose() + R;
  double nis = residual.transpose() * S.inverse() * residual;//NIS：测量残差的归一化二次型，服从自由度为 “测量维度” 的卡方分布；，若 NIS 频繁超过阈值：测量模型（h或H）或测量噪声R设置不合理（如R过小，低估了测量噪声）；
  double nees = (x - x_prior).transpose() * P.inverse() * (x - x_prior);//NEES：状态估计误差的归一化二次型，服从自由度为 “状态维度” 的卡方分布。，若 NEES 频繁超过阈值：状态转移模型（f或F）或过程噪声Q设置不合理（如Q过小，低估了系统模型的不确定性）。

  // 卡方检验阈值（自由度=4，取置信水平95%）
  constexpr double nis_threshold = 0.711;
  constexpr double nees_threshold = 0.711;

  if (nis > nis_threshold) nis_count_++, data["nis_fail"] = 1;
  if (nees > nees_threshold) nees_count_++, data["nees_fail"] = 1;
  total_count_++;
  last_nis = nis;

  recent_nis_failures.push_back(nis > nis_threshold ? 1 : 0);

  if (recent_nis_failures.size() > window_size) {
    recent_nis_failures.pop_front();
  }

  int recent_failures = std::accumulate(recent_nis_failures.begin(), recent_nis_failures.end(), 0);
  double recent_rate = static_cast<double>(recent_failures) / recent_nis_failures.size();

  data["residual_yaw"] = residual[0];
  data["residual_pitch"] = residual[1];
  data["residual_distance"] = residual[2];
  data["residual_angle"] = residual[3];
  data["nis"] = nis;
  data["nees"] = nees;
  data["recent_nis_failures"] = recent_rate;

  return x;
}

}  // namespace tools