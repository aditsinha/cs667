
#include <cassert>
#include <cmath>
#include <random>
#include <fstream>
#include <iostream>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

// Generate a random point on the surface of the d-dimensional unit
// hypersphere
template <class URNG>
Eigen::Hyperplane<double, Eigen::Dynamic> GenerateRandomHyperplane(int d, URNG& g) {
  Eigen::VectorXd vec(d);

  std::normal_distribution<double> normal;

  // generate a random vector
  for (int i = 0; i < d; i++) {
    vec(i) = normal(g);
  }

  // normalize so it lies on the unit ball
  vec = vec * (1 / vec.norm());
  return Eigen::Hyperplane<double, Eigen::Dynamic>(vec, 0);
}

template <class URNG>
Eigen::MatrixXd GenerateFeatureMatrix(int n, int d, URNG& g) {
  Eigen::MatrixXd mat(n, d);

  std::normal_distribution<double> normal(0.0, 1.0);
  std::exponential_distribution<double> exponential(1.0);

  // generate a uniform random point in the unit cube.

  for (int i = 0; i < n; i++) {
    // http://mathworld.wolfram.com/BallPointPicking.html
    for (int j = 0; j < d; j++) {
      mat(i, j) = normal(g);
    }
    mat.row(i) = mat.row(i) * (1 / (sqrt(exponential(g) + mat.row(i).squaredNorm())));
  }

  return mat;
}

template <class URNG>
Eigen::VectorXi CalculateLabelVector(Eigen::Hyperplane<double, Eigen::Dynamic> separator, Eigen::MatrixXd features, URNG& g) {
  Eigen::VectorXi labels(features.rows());

  std::normal_distribution<double> normal (0.0, 0.1);

  for (int i = 0; i < features.rows(); i++) {
    // Add some noise to the distance for fun
    labels(i) = (separator.signedDistance(features.row(i)) + normal(g) >= 0) ? 1 : 0;
    // labels(i) = (separator.signedDistance(features.row(i)) >= 0) ? 1 : 0;
  }

  return labels;
}

void ExportData(Eigen::VectorXi labels, Eigen::MatrixXd features, std::ofstream& out) {
  assert(out.is_open());

  for (int i = 0; i < labels.rows(); i++) {
    out << labels(i) << ",";
    for (int j = 0; j < features.cols(); j++) {
      out << features(i,j) << ",";
    }
    out << std::endl;
  }
}


int main(int argc, char** argv) {
  std::default_random_engine generator;

  // choose random hyperplane that goes through the origin
  Eigen::Hyperplane<double, Eigen::Dynamic> separator = GenerateRandomHyperplane(10, generator);

  Eigen::MatrixXd features = GenerateFeatureMatrix(100000, 10, generator);
  Eigen::VectorXi labels = CalculateLabelVector(separator, features, generator);

  std::ofstream out_file("data.csv");
  ExportData(labels, features, out_file);
}
