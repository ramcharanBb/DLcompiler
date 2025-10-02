
// Test nova.matmul
func.func @test_matmul(%a: tensor<2x3xf32>, %b: tensor<3x4xf32>) -> tensor<2x4xf32> {
  %0 = nova.matmul %a, %b : tensor<2x4xf32>, tensor<3x4xf32>
  return %0 : tensor<2x4xf32>
}