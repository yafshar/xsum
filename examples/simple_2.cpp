#include <iomanip>
#include <iostream>

#include "../xsum/xsum.hpp"

using namespace xsum;

int main() {
  // Large superaccumulator
  xsum_large lacc;
  double const a = 0.7209e-5;
  double s = 0;
  for (int i = 0; i < 10000; ++i) {
    lacc.add(a);
    s += a;
  }
  std::cout << std::setprecision(20) << lacc.round() << "\n"
            << std::setprecision(20) << s << "\n";
}
