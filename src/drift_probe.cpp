// drift_probe.cpp — test helper: reads real state vectors (10 numbers per line) on
// stdin, prints drift(x, E) per line. Used by validate.py to compare against Python.
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "model.hpp"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: drift_probe <E>  (states on stdin)\n";
    return 2;
  }
  const double E = std::strtod(argv[1], nullptr);
  const brillouin::Params p;
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
    std::istringstream ss(line);
    brillouin::State x{};
    for (int i = 0; i < brillouin::DIM; ++i) {
      if (!(ss >> x[i])) {
        std::cerr << "error: expected " << brillouin::DIM << " numbers per line\n";
        return 1;
      }
    }
    const brillouin::State d = brillouin::drift(x, E, p);
    std::cout << std::setprecision(17);
    for (int i = 0; i < brillouin::DIM; ++i) std::cout << (i ? " " : "") << d[i];
    std::cout << "\n";
  }
  return 0;
}
