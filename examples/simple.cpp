#include <iomanip>
#include <iostream>

#include "../xsum/xsum.hpp"

int main() {
  // Large superaccumulator
  xsum_large_accumulator lacc;
  double const a = 0.7209e-5;
  double s = 0;
  for (int i = 0; i < 10000; ++i) {
    xsum_add(&lacc, a);
    s += a;
  }
  std::cout << std::setprecision(20) << xsum_round(&lacc) << "\n"
            << std::setprecision(20) << s << "\n";
}
