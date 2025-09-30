//test file
func.func @test_add(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
<<<<<<< HEAD
  %0 = nova.add %arg0, %arg1 : tensor<2x2xf32>, tensor<2x2xf32>
=======
  %0 = nova.add %arg0, %arg1 : tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
>>>>>>> 08c39335bccfc6f99a9b5e6b29485ac525e18e91
  return %0 : tensor<2x2xf32>
}

