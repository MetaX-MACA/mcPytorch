#include <torch/torch.h>
#include <iostream>

int main() {
  const at::Device device("cuda");
  torch::Tensor a = torch::rand({2, 4}).to(device);
  torch::Tensor b = torch::rand({1, 4}).to(device);
  torch::Tensor c  = a + b;
  std::cout << "==> a\n " << a << std::endl;
  std::cout << "==> b\n " << b << std::endl;
  std::cout << "==> c\n " << c << std::endl;
  return 0;
}
