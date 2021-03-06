
?
?void convolve_common_engine_float_NHWC<__half, __half, 128, 6, 7, 3, 3, 5, true, false, false, false, false>(int, int, int, __half const*, __half const*, int, __half*, conv_kernel_common_params, unsigned long long, unsigned long, float, float, int, __half const*, __half const*, bool)C?2* 28???@??H??bPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_dwconv/depthwiseh?u  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 
8??A@??AH??AbZModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_expand_bn/FusedBatchNormV3hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?n8͊?@͊?H͊?bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2b_se_excite/mulhu  ?B
?
?void convolve_common_engine_float_NHWC<__half, __half, 128, 5, 5, 3, 3, 3, true, false, false, false, false>(int, int, int, __half const*, __half const*, int, __half*, conv_kernel_common_params, unsigned long long, unsigned long, float, float, int, __half const*, __half const*, bool)T?*2?b8??:@??:H??:XbHModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/stem_conv/Conv2Dhu  zB
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?b8??7@??7H??7bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block1a_se_excite/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?I8??5@??5H??5bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_expand_activation/mulhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?28??-@??-H??-bOModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block1a_se_squeeze/Meanhu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_128x128_nn_align8>(cutlass_75_tensorop_f16_s1688gemm_f16_128x128_nn_align8::Params)? ??*?2?8??+@??+H??+XbRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_expand_conv/Conv2Dhu  ?A
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?I8??)@??)H??)bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_se_excite/mulhu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 5, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)P*?2?-8??)@??)H??)bPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3b_dwconv/depthwisehu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)(*?2(8??)@??)H??)bNModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_dwconv_pad/Padhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??(@??(H??(bYModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_expand_activation/Sigmoidhu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 3, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)0*?2?n8??$@??$H??$bPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2b_dwconv/depthwisehu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 5, 2, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)d*?2?8??$@??$H??$bPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3a_dwconv/depthwisehu  HB
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 5, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)P*?2? 8??@??H??bPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5b_dwconv/depthwisehu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 5, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)P*?2? 8??@??H??bPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5c_dwconv/depthwisehu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 3, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)0*?2?b8??@??H??bPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block1a_dwconv/depthwisehu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 3, 2, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct):*?2?I8??@??H??bPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_dwconv/depthwisehu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?-8??@??H??bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3b_se_excite/mulhu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8::Params)` ??*?2?18??@??H??XbSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block1a_project_conv/Conv2Dhu  HB
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?	*?2 8??@??H??bSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2b_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)(*?2(8??@??H??bIModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/stem_conv_pad/Padhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?	*?2 8ٌ@ٌHٌbZModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2b_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?	*?2 8??@??H??bZModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3a_expand_bn/FusedBatchNormV3hu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 5, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)P*?2?8??@??H??bPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5a_dwconv/depthwisehu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/stem_bn/FusedBatchNormV3hu  ?B
?
Div_GPU_DT_HALF_DT_HALF_kernel*?2?$8??@??H??bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/normalization/truedivhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block1a_bn/FusedBatchNormV3hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8ڶ@ڶHڶbUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3a_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bNModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2b_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8ڮ@ڮHڮbUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2b_expand_activation/mulhu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 5, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)P*?2?8??@??H??bPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6d_dwconv/depthwisehu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 5, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)P*?2?8??@??H??bPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6b_dwconv/depthwisehu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8::Params)` ??*?2?18??@??H??XbRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3a_expand_conv/Conv2Dhu  HB
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 5, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)P*?2?8ړ@ړHړbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6c_dwconv/depthwisehu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8::Params)` ??*?2?18??@??H??XbRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2b_expand_conv/Conv2Dhu  HB
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2? 8ڽ@ڽHڽbMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5b_se_excite/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2? 8ڽ@ڽHڽbMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5c_se_excite/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bKModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/stem_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8ۑ@ۑHۑbNModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block1a_activation/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)(*?2(8??@??H??bNModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3a_dwconv_pad/Padhu  ?B
?
Sub_GPU_DT_HALF_DT_HALF_kernel*?2?$8??@??H??bIModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/normalization/subhu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x128_nn_align8>(cutlass_75_tensorop_f16_s1688gemm_f16_64x128_nn_align8::Params)? ??*?2?8??@??H??XbSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2b_project_conv/Conv2Dhu  ?A
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2b_activation/Sigmoidhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 
8??@??H??bSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bYModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3a_expand_activation/Sigmoidhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3a_se_excite/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8۴@۴H۴bYModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2b_expand_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bOModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/stem_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block1a_activation/Sigmoidhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bNModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4b_se_excite/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4c_se_excite/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5a_se_excite/mulhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2$8??@??H??bOModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2b_se_squeeze/Meanhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?28??@??H??bOModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_se_squeeze/Meanhu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8::Params)` ??*?2?8??@??H??XbSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_project_conv/Conv2Dhu  HB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??
@??
H??
bRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_activation/Sigmoidhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??
@??
H??
b[Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block1a_project_bn/FusedBatchNormV3hu  ?B
?
 Cast_GPU_DT_HALF_DT_FLOAT_kernel*?2??8??
@??
H??
bJModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/normalization/Casthu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??
@??
H??
bZModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4a_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??
@??
H??
bZModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3b_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??	@??	H??	bSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3b_bn/FusedBatchNormV3hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2??8??	@??	H??	bLModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/normalization/Cast_1hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2??8??	@??	H??	bFModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/rescaling/Casthu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4a_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bNModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3b_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3b_expand_activation/mulhu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 3, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)0*?2?8??@??H??bPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4c_dwconv/depthwisehu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 3, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)0*?2?8??@??H??bPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4b_dwconv/depthwisehu  ?B
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?218??@??H??PXbRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5b_expand_conv/Conv2Dhu  ?A
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?218ޒ@ޒHޒPXbRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5c_expand_conv/Conv2Dhu  ?A
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8ދ@ދHދbMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_se_excite/mulhu  ?B
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?218??@??H??PXbRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_expand_conv/Conv2Dhu  ?A
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?**?2 	8??@??H??bZModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?**?2 	8??@??H??bSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5c_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?**?2 	8??@??H??bZModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5b_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?**?2 	8??@??H??bSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5b_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?**?2 	8??@??H??bZModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5c_expand_bn/FusedBatchNormV3hu  ?B
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?2?8??@??H??PXbRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3b_expand_conv/Conv2Dhu  ?A
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6b_se_excite/mulhu  ?B
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?2?8޿@޿H޿PXbRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4a_expand_conv/Conv2Dhu  ?A
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6d_se_excite/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6c_se_excite/mulhu  ?B
?
:turing_fp16_s1688gemm_fp16_128x256_ldg8_f2f_stages_32x1_nn??? ??*?2
8??@??H??PXbGModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/top_conv/Conv2Dh
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3b_activation/Sigmoidhu  ?B
?
7turing_fp16_s1688gemm_fp16_64x128_sliced1x2_ldg8_f2f_nn???*?2?8??@??H??PXbSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3b_project_conv/Conv2Dhu  ?A
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)(*?2(8??@??H??bNModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4a_dwconv_pad/Padhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bYModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3b_expand_activation/Sigmoidhu  ?B
?
.turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_nn???*?218??@??H??PXbSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5c_project_conv/Conv2Dhu  ?A
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bYModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4a_expand_activation/Sigmoidhu  ?B
?
.turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_nn???*?218??@??H??PXbSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5b_project_conv/Conv2Dhu  ?A
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 3, 2, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct):*?2?8??@??H??bPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4a_dwconv/depthwisehu  ?B
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nt???*?2
8??@??H??PXbjgradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/top_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nt???*?2	8??@??H??PXbvgradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_project_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)(*?2(8??@??H??bNModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_dwconv_pad/Padhu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 3, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)0*?2?8??@??H??bPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_dwconv/depthwisehu  ?B
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_tn???*?28??@??H??PXbigradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/top_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8ޭ@ޭHޭbUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5b_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_expand_activation/mulhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?	*?2 8??@??H??bSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3a_bn/FusedBatchNormV3hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5c_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bNModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5b_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bNModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5c_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4a_se_excite/mulhu  ?B
?
Div_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bWgradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/global_average_pooling2d/truedivhu  ?B
?
.turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_tn???*?2	8??@??H??PXbugradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_project_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5a_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bZModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4c_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bZModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5a_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4b_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bZModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4b_expand_bn/FusedBatchNormV3hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8߯@߯H߯bNModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3a_activation/mulhu  ?B
?
.turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_nn???*?218??@??H??PXbSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5a_project_conv/Conv2Dhu  ?A
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?28??@??H??PXbSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_project_conv/Conv2Dhu  ?A
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4c_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::OuterReductionKernel<16, Eigen::TensorReductionEvaluatorBase<Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<0l>> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<long, Eigen::type2index<1l> > const, Eigen::TensorReshapingOp<Eigen::IndexList<Eigen::type2index<1l>, long> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, Eigen::internal::SumReducer<float>, long>(Eigen::internal::SumReducer<float>, Eigen::TensorReductionEvaluatorBase<Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<0l>> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<long, Eigen::type2index<1l> > const, Eigen::TensorReshapingOp<Eigen::IndexList<Eigen::type2index<1l>, long> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long, long, Eigen::TensorReductionEvaluatorBase<Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<0l>> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<long, Eigen::type2index<1l> > const, Eigen::TensorReshapingOp<Eigen::IndexList<Eigen::type2index<1l>, long> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>::CoeffReturnType*)<*?2?8??@??H??bagradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/top_bn/FusedBatchNormGradV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5c_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bYModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_expand_activation/Sigmoidhu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8::Params)` ??*?2?8??@??H??XbSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3a_project_conv/Conv2Dhu  HB
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?	8??@??H??bEModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/rescaling/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bYModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5c_expand_activation/Sigmoidhu  ?B
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?218??@??H??PXbSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4c_project_conv/Conv2Dhu  ?A
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?218??@??H??PXbSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4b_project_conv/Conv2Dhu  ?A
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5b_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bYModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5b_expand_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<long, Eigen::type2index<1l> > const, Eigen::TensorReshapingOp<Eigen::IndexList<Eigen::type2index<1l>, long> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<long, Eigen::type2index<1l> > const, Eigen::TensorReshapingOp<Eigen::IndexList<Eigen::type2index<1l>, long> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?2(8??@??H??bagradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/top_bn/FusedBatchNormGradV3hu  ?B
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?218??@??H??PXbRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4b_expand_conv/Conv2Dhu  ?A
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bNModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4c_activation/mulhu  ?B
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?218??@??H??PXbRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4c_expand_conv/Conv2Dhu  ?A
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?218??@??H??PXbRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5a_expand_conv/Conv2Dhu  ?A
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4c_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bNModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5a_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5a_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4b_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bNModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4b_activation/mulhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2<8߻@߻H߻bOModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3b_se_squeeze/Meanhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_se_excite/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3a_activation/Sigmoidhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 
8߬@߬H߬b[Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2b_project_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?P*?2 8??@??H??bOModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/top_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 
8??@??H??b[Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_project_bn/FusedBatchNormV3hu  ?B
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?2	8ߏ@ߏHߏPXbRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6d_expand_conv/Conv2Dhu  ?A
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?2	8ߏ@ߏHߏPXbRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_expand_conv/Conv2Dhu  ?A
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?2	8ޏ@ޏHޏPXbRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6c_expand_conv/Conv2Dhu  ?A
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?2	8??@??H??PXbRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6b_expand_conv/Conv2Dhu  ?A
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2$8??@??H??bOModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3a_se_squeeze/Meanhu  ?B
?
7turing_fp16_s1688gemm_fp16_64x128_sliced1x2_ldg8_f2f_nn???*?28??@??H??PXbSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6b_project_conv/Conv2Dhu  ?A
?
7turing_fp16_s1688gemm_fp16_64x128_sliced1x2_ldg8_f2f_nn???*?28??@??H??PXbSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6c_project_conv/Conv2Dhu  ?A
?
7turing_fp16_s1688gemm_fp16_64x128_sliced1x2_ldg8_f2f_nn???*?28??@??H??PXbSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6d_project_conv/Conv2Dhu  ?A
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?H*?2 8??@??H??bZModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6d_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?H*?2 8??@??H??bZModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_expand_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bYModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4c_expand_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bYModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5a_expand_activation/Sigmoidhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?H*?2 8??@??H??bZModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6c_expand_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bYModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4b_expand_activation/Sigmoidhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?H*?2 8??@??H??bZModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6b_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?H*?2 8??@??H??bSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6c_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?H*?2 8??@??H??bSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6b_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4b_activation/Sigmoidhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?H*?2 8??@??H??bSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6d_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4c_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5a_activation/Sigmoidhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?H*?2 8??@??H??bSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_bn/FusedBatchNormV3hu  ?B
?
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bGModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2b_add/addhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?2(8??@??H??b;cond_1/then/_10/cond_1/Adam/Adam/update_7/ResourceApplyAdamhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorGeneratorOp<tensorflow::generator::SparseXentLossGenerator<float, long>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorGeneratorOp<tensorflow::generator::SparseXentLossGenerator<float, long>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)>*?28??@??H??bgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshu  ?B
U
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8ߟ@ߟHߟbAdam/gradients/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bJModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/top_activation/mulhu  ?B
?
.turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_nn???*?218??@??H??PXbSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4a_project_conv/Conv2Dhu  ?A
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?2(8??@??H??b;cond_1/then/_10/cond_1/Adam/Adam/update_4/ResourceApplyAdamhu  ?B
W
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bAdam/gradients/mul_1hu  ?B
W
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8ߏ@ߏHߏbAdam/gradients/mul_2hu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2?8??@??H??bOModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5b_se_squeeze/Meanhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2?8߃@߃H߃bOModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5c_se_squeeze/Meanhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4a_bn/FusedBatchNormV3hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6c_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6d_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bNModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6c_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bagradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_se_excite/mul/Mul_1hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bNModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6b_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6b_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bNModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6d_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bNModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_activation/mulhu  ?B
?
7turing_fp16_s1688gemm_fp16_64x128_sliced1x2_ldg8_f2f_nn???*?28??@??H??PXbSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_project_conv/Conv2Dhu  ?A
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bAdam/gradients/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bNModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/top_activation/Sigmoidhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8߾@߾H߾bNModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4a_activation/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)**?2(8߻@߻H߻b[gradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/global_average_pooling2d/BroadcastTohu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?**?2 8??@??H??bSModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6d_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bYModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_expand_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bYModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6b_expand_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6b_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bYModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6c_expand_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bYModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6d_expand_activation/Sigmoidhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2x8??@??H??bOModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5a_se_squeeze/Meanhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2x8??@??H??bOModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4c_se_squeeze/Meanhu  ?B
U
Sub_GPU_DT_HALF_DT_HALF_kernel*?2?8ߟ@ߟHߟbAdam/gradients/subhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2x8ߟ@ߟHߟbOModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4b_se_squeeze/Meanhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6c_activation/Sigmoidhu  ?B
W
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bAdam/gradients/addhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??b[Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3a_project_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??b[Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3b_project_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4a_activation/Sigmoidhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<Eigen::half*, Eigen::half*, tensorflow::functor::Sum<Eigen::half> >(Eigen::half*, Eigen::half*, int, int, int, tensorflow::functor::Sum<Eigen::half>)%*?2?8??@??H??bagradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_se_excite/mul/Sum_1hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??b[Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5a_project_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??b[Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5b_project_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??b[Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5c_project_bn/FusedBatchNormV3hu  ?B
?
?
void Eigen::internal::OuterReductionKernel<16, Eigen::TensorReductionEvaluatorBase<Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<0l>> const, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, Eigen::internal::SumReducer<float>, long>(Eigen::internal::SumReducer<float>, Eigen::TensorReductionEvaluatorBase<Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<0l>> const, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long, long, Eigen::TensorReductionEvaluatorBase<Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<0l>> const, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>::CoeffReturnType*)&*?2?8??@??H??bagradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/top_bn/FusedBatchNormGradV3hu  ?B
?
?void Eigen::internal::OuterReductionKernel<16, Eigen::TensorReductionEvaluatorBase<Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<0l>> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<long, Eigen::type2index<1l> > const, Eigen::TensorReshapingOp<Eigen::IndexList<Eigen::type2index<1l>, long> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, Eigen::internal::SumReducer<float>, long>(Eigen::internal::SumReducer<float>, Eigen::TensorReductionEvaluatorBase<Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<0l>> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<long, Eigen::type2index<1l> > const, Eigen::TensorReshapingOp<Eigen::IndexList<Eigen::type2index<1l>, long> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long, long, Eigen::TensorReductionEvaluatorBase<Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<0l>> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<long, Eigen::type2index<1l> > const, Eigen::TensorReshapingOp<Eigen::IndexList<Eigen::type2index<1l>, long> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>::CoeffReturnType*)<*?2{8??@??H??bmgradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_project_bn/FusedBatchNormGradV3hu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2::Params)? ??*?28??@??H??XbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4b_se_expand/Conv2Dhu  ?A
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2::Params)? ??*?28??@??H??XbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5a_se_expand/Conv2Dhu  ?A
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2::Params)? ??*?28??@??H??XbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4c_se_expand/Conv2Dhu  ?A
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2::Params)? ??*?28??@??H??XbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5c_se_expand/Conv2Dhu  ?A
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2::Params)? ??*?28??@??H??XbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_se_expand/Conv2Dhu  ?A
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2::Params)? ??*?28??@??H??XbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5b_se_expand/Conv2Dhu  ?A
?
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bGModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3b_add/addhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8߿@߿H߿bRModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)6*?28߷@߷H߷bgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2?8??@??H??bFModel0_FineTuned_10Layers_SixEpochsTotal/global_average_pooling2d/Meanhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?2(8??@??H??b<cond_1/then/_10/cond_1/Adam/Adam/update_10/ResourceApplyAdamhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??b[Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6c_project_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??b[Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4a_project_bn/FusedBatchNormV3hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bNModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_activation/mulhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2<8??@??H??bOModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4a_se_squeeze/Meanhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??b[Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6b_project_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??b[Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_project_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<long, Eigen::type2index<1l> > const, Eigen::TensorReshapingOp<Eigen::IndexList<Eigen::type2index<1l>, long> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<long, Eigen::type2index<1l> > const, Eigen::TensorReshapingOp<Eigen::IndexList<Eigen::type2index<1l>, long> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?2(8??@??H??bmgradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_project_bn/FusedBatchNormGradV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??b[Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_project_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long) *?28??@??H??bArgMaxhu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align1>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align1::Params)q ??*?2
8??@??H??Xb5Model0_FineTuned_10Layers_SixEpochsTotal/dense/MatMulhu  HB
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2?8??@??H??bOModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6c_se_squeeze/Meanhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2?8??@??H??bOModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6d_se_squeeze/Meanhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??b[Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6d_project_bn/FusedBatchNormV3hu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2?8??@??H??bOModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_se_squeeze/Meanhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2?8??@??H??bOModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6b_se_squeeze/Meanhu  ?B
?
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2?8ߍ@ߍHߍbGModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5b_add/addhu  ?B
?
 Cast_GPU_DT_HALF_DT_FLOAT_kernel*?2?8??@??H??bkgradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_project_conv/Conv2D/Cast/Casthu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2::Params)` ??*?28ߊ@ߊHߊXbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5b_se_reduce/Conv2Dhu  HB
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??b[Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4c_project_bn/FusedBatchNormV3hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8??@??H??bLModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/top_conv/Conv2D/Casthu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_tn_align1>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_tn_align1::Params)r ??*?28??@??H??XbCgradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/dense/MatMulhu  HB
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??b[Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4b_project_bn/FusedBatchNormV3hu  ?B
?
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2?8߂@߂H߂bGModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5c_add/addhu  ?B
?
 Cast_GPU_DT_HALF_DT_FLOAT_kernel*?2?8??@??H??b_gradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/top_conv/Conv2D/Cast/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?@?H?bXModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_project_conv/Conv2D/Casthu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2::Params)` ??*?28?@?H?XbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5c_se_reduce/Conv2Dhu  HB
?
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2{8?u@?uH?ubGModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4c_add/addhu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2::Params)` ??*?28?t@?tH?tXbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_se_reduce/Conv2Dhu  HB
?
?void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *?2 8?p@?pH?pbAll_10hu  ?B
?
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2{8?p@?pH?pbGModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4b_add/addhu  ?B
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2d8?p@?pH?pbmul_9hu  ?B
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2Z8?o@?oH?obmul_6hu  ?B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*?2?8?o@?oH?ob
IsFinite_7hu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2::Params)` ??*?28?m@?mH?mXbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4b_se_reduce/Conv2Dhu  HB
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?2(8?l@?lH?lb;cond_1/then/_10/cond_1/Adam/Adam/update_2/ResourceApplyAdamhu  ?B
?
dvoid tensorflow::BiasGradNHWC_SharedAtomics<Eigen::half>(int, Eigen::half const*, Eigen::half*, int) ?*?28?i@?iH?ibPgradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/dense/BiasAdd/BiasAddGradhu  ?B
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?i@?iH?iXbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4a_se_reduce/Conv2Dhu  HA
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2::Params)` ??*?28?g@?gH?gXbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4c_se_reduce/Conv2Dhu  HB
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8::Params)^ ??*?28?g@?gH?gXbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_se_reduce/Conv2Dhu  HA
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?2(8?f@?fH?fb9cond_1/then/_10/cond_1/Adam/Adam/update/ResourceApplyAdamhu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2::Params)` ??*?28?b@?bH?bXbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5a_se_reduce/Conv2Dhu  HB
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8::Params)^ ??*?28?a@?aH?aXbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6c_se_reduce/Conv2Dhu  HA
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8::Params)^ ??*?28?a@?aH?aXbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6d_se_reduce/Conv2Dhu  HA
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nt_align1>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nt_align1::Params)v ??*?2(8?a@?aH?abEgradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/dense/MatMul_1hu  HB
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8::Params)^ ??*?28?`@?`H?`XbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6b_se_reduce/Conv2Dhu  HA
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?`@?`H?`XbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2b_se_reduce/Conv2Dhu  HA
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_tn_align8>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_tn_align8::Params)^ ??*?28?`@?`H?`Xbrgradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_se_expand/Conv2D/Conv2DBackpropInputhu  HA
?
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2J8?`@?`H?`bGModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6d_add/addhu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nt_align8>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nt_align8::Params)` ??*?28?`@?`H?`Xbsgradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_se_reduce/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?`@?`H?`XbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3b_se_reduce/Conv2Dhu  HA
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?_@?_H?_bWModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_expand_conv/Conv2D/Casthu  ?B
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?_@?_H?_XbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3a_se_reduce/Conv2Dhu  HA
?
*volta_fp16_s884gemm_fp16_64x64_ldg8_f2f_nnj??*?28?_@?_H?_PXbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6c_se_expand/Conv2Dhu  B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?]@?]H?]bXModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6b_project_conv/Conv2D/Casthu  ?B
?
?void tensorflow::(anonymous namespace)::DynamicStitchKernel<int>(int, int, tensorflow::GpuDeviceArrayStruct<int, 8>, tensorflow::GpuDeviceArrayStruct<int const*, 8>, int*)*?28?]@?]H?]b]gradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/global_average_pooling2d/DynamicStitchhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?[@?[H?[bWModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6d_expand_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?Z@?ZH?ZbWModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6c_expand_conv/Conv2D/Casthu  ?B
?
*volta_fp16_s884gemm_fp16_64x64_ldg8_f2f_ntj??*?28?Z@?ZH?ZPXbsgradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_se_expand/Conv2D/Conv2DBackpropFilterhu  HB
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2?8?Y@?YH?YbOModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_se_squeeze/Meanhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?W@?WH?WbWModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6b_expand_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?V@?VH?VbXModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6d_project_conv/Conv2D/Casthu  ?B
?
*volta_fp16_s884gemm_fp16_64x64_ldg8_f2f_nnj??*?28?V@?VH?VPXbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_se_expand/Conv2Dhu  B
?
?
void Eigen::internal::OuterReductionKernel<16, Eigen::TensorReductionEvaluatorBase<Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<0l>> const, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, Eigen::internal::SumReducer<float>, long>(Eigen::internal::SumReducer<float>, Eigen::TensorReductionEvaluatorBase<Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<0l>> const, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long, long, Eigen::TensorReductionEvaluatorBase<Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<0l>> const, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>::CoeffReturnType*)&*?2{8?U@?UH?Ubmgradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_project_bn/FusedBatchNormGradV3hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?U@?UH?UbXModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6c_project_conv/Conv2D/Casthu  ?B
?
*volta_fp16_s884gemm_fp16_64x64_ldg8_f2f_nnj??*?28?T@?TH?TPXbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6b_se_expand/Conv2Dhu  B
?
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2J8?T@?TH?TbGModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6c_add/addhu  ?B
?
*volta_fp16_s884gemm_fp16_64x64_ldg8_f2f_nnj??*?28?R@?RH?RPXbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6d_se_expand/Conv2Dhu  B
?
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2J8?R@?RH?RbGModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6b_add/addhu  ?B
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?Q@?QH?QXbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_se_reduce/Conv2Dhu  HA
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?P@?PH?PbQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3b_se_expand/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2$8?P@?PH?PbQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6b_se_expand/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2$8?P@?PH?PbQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6d_se_expand/Sigmoidhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?O@?OH?ObPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_dwconv/depthwisehu  ?B
?
Kvoid Eigen::internal::ReductionInitKernel<float, long>(float, long, float*)*?28?O@? H?.bagradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/top_bn/FusedBatchNormGradV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2$8?N@?NH?NbQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_se_expand/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?M@?MH?MbQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5a_se_expand/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?M@?MH?MbQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5c_se_expand/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2$8?L@?LH?LbQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6c_se_expand/Sigmoidhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?L@?LH?LbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6b_dwconv/depthwisehu  ?B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*?2?8?K@?KH?Kb
IsFinite_4hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?K@?KH?KbQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4a_se_expand/Sigmoidhu  ?B
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?K@?KH?KXbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3b_se_expand/Conv2Dhu  HA
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?K@?KH?KbQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_se_expand/Sigmoidhu  ?B
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?K@?KH?KXbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2b_se_expand/Conv2Dhu  HA
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?J@?JH?JbQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4b_se_expand/Sigmoidhu  ?B
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?J@?JH?JXbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3a_se_expand/Conv2Dhu  HA
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?I@?IH?IbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6c_dwconv/depthwisehu  ?B
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?G@?GH?GXbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_se_expand/Conv2Dhu  HA
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?G@?GH?GXbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4a_se_expand/Conv2Dhu  HA
?
dvoid tensorflow::BiasGradNHWC_SharedAtomics<Eigen::half>(int, Eigen::half const*, Eigen::half*, int) ?*?28?F@?FH?Fbkgradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_se_reduce/BiasAdd/BiasAddGradhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?F@?FH?FbXModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_project_conv/Conv2D/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?E@?EH?EbQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4c_se_expand/Sigmoidhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?D@?DH?DbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6d_dwconv/depthwisehu  ?B
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8::Params)^ ??*?28?D@?DH?DXbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block1a_se_reduce/Conv2Dhu  HA
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?D@?DH?DbQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5b_se_expand/Sigmoidhu  ?B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2 8?C@?CH?Cbmul_12hu  ?B
?
?void cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And)+ *?228?C@?CH?CbAll_7hu  ?B
|
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?C@?CH?Cb:Model0_FineTuned_10Layers_SixEpochsTotal/dense/MatMul/Casthu  ?B
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8::Params)^ ??*?28?A@?AH?AXbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block1a_se_expand/Conv2Dhu  HA
?
Kvoid Eigen::internal::ReductionInitKernel<float, long>(float, long, float*)*?28?@@?H? bmgradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_project_bn/FusedBatchNormGradV3hu  ?B
?
?void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *?28?@@?@H?@bAll_8hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?@@?@H?@bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2b_se_expand/Sigmoidhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?@@?@H?@b;cond_1/then/_10/cond_1/Adam/Adam/update_3/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *?2 8??@??H??bAll_2hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28??@??H??bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3a_se_expand/Sigmoidhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28??@??H??b;cond_1/then/_10/cond_1/Adam/Adam/update_5/ResourceApplyAdamhu  ?B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*?28?>@?>H?>bIsFinite_10hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sigmoid_gradient_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sigmoid_gradient_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?2$8?=@?=H?=bkgradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_se_expand/Sigmoid/SigmoidGradhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *?28?<@?<H?<bAll_9hu  ?B
?
 Cast_GPU_DT_HALF_DT_FLOAT_kernel*?2?8?<@?<H?<bMgradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/dense/MatMul/Cast/Casthu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?<@?<H?<b;cond_1/then/_10/cond_1/Adam/Adam/update_8/ResourceApplyAdamhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?;@?;H?;bWModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5b_expand_conv/Conv2D/Casthu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?:@?:H?:b<cond_1/then/_10/cond_1/Adam/Adam/update_11/ResourceApplyAdamhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?:@?:H?:bWModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5c_expand_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?9@?9H?9bXModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5b_project_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?9@?9H?9bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6d_se_expand/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?9@?9H?9bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6c_se_reduce/Conv2D/Casthu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?9@?9H?9b;cond_1/then/_10/cond_1/Adam/Adam/update_6/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::RowReduceKernel<cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long>, float*, cub::Sum>(cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long>, float*, int, int, cub::Sum, std::iterator_traits<cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long> >::value_type)*?28?8@?8H?8b@Model0_FineTuned_10Layers_SixEpochsTotal/softmax_float32/Softmaxhu  ?B
?

?	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorReshapingOp<Eigen::IndexList<int, Eigen::type2index<1l> > const, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> > const> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorReshapingOp<Eigen::IndexList<int, Eigen::type2index<1l> > const, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> > const> const> const> const, Eigen::GpuDevice>, int)*?28?8@?8H?8bgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshu  ?B
?
?void cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And)+ *?2-8?8@?8H?8bAll_4hu  ?B
?
?void tensorflow::(anonymous namespace)::GenerateNormalizedProb<float, float, 4>(float const*, float const*, float const*, float*, int, int, bool)%*?28?8@?8H?8b@Model0_FineTuned_10Layers_SixEpochsTotal/softmax_float32/Softmaxhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?7@?7H?7bPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5a_dwconv/depthwisehu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?7@?7H?7bPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3b_dwconv/depthwisehu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?6@?6H?6bPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5c_dwconv/depthwisehu  ?B
?
?void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *?2 8?6@?6H?6bAllhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?6@?6H?6bPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5b_dwconv/depthwisehu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?5@?5H?5bPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4b_dwconv/depthwisehu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?5@?5H?5bPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4a_dwconv/depthwisehu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?4@?4H?4bWModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_expand_conv/Conv2D/Casthu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?4@?4H?4XbHModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/stem_conv/Conv2Dhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?4@?4H?4b;cond_1/then/_10/cond_1/Adam/Adam/update_9/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?4@?4H?4bPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_dwconv/depthwisehu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?3@?3H?3bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6b_se_reduce/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?2@?2H?2bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_se_reduce/Conv2D/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?2@?2H?2bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_se_expand/Sigmoidhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?2@?2H?2bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6c_se_expand/Conv2D/Casthu  ?B
?
?void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)2 *?28?2@?2H?2bAll_7hu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?2@?2H?2bPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_dwconv/depthwisehu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?2@?2H?2bPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2b_dwconv/depthwisehu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?1@?1H?1bPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3a_dwconv/depthwisehu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?1@?1H?1XbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5c_se_reduce/Conv2Dhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?1@?1H?1bPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4c_dwconv/depthwisehu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?0@?0H?0bXModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5c_project_conv/Conv2D/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?0@?0H?0bAdam/gradients/Sigmoid_1hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?0@?0H?0bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4a_se_reduce/Sigmoidhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?0@?0H?0b;cond_1/then/_10/cond_1/Adam/Adam/update_1/ResourceApplyAdamhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?0@?0H?0bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6d_se_reduce/Sigmoidhu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?0@?0H?0XbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_se_reduce/Conv2Dhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *?28?0@?0H?0bAll_3hu  ?B
?
?void tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)?*  2$8?0@?0H?0bkgradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_se_expand/BiasAdd/BiasAddGradhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28?0@?0H?0bBgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2q8?0@?0H?0bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6c_dwconv/depthwise/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?0@?0H?0bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6b_se_expand/Conv2D/Casthu  ?B
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?0@?0H?0bmul_2hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?0@?0H?0bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block1a_se_reduce/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?0@?0H?0bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_se_reduce/Sigmoidhu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?0@?0H?0Xb5Model0_FineTuned_10Layers_SixEpochsTotal/dense/MatMulhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?0@?0H?0bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6d_se_reduce/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?0@?0H?0bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4b_se_expand/BiasAddhu  ?B
?
?void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Max>(float const*, float*, int, int, cub::Max, std::iterator_traits<float const*>::value_type)**?28?0@?0H?0b@Model0_FineTuned_10Layers_SixEpochsTotal/softmax_float32/Softmaxhu  ?B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*?28?/@?/H?/b
IsFinite_9hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?/@?/H?/bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_se_reduce/Sigmoidhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?/@?/H?/bWModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4b_expand_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?/@?/H?/bXModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5a_project_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?/@?/H?/bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6d_se_reduce/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?/@?/H?/bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_se_expand/Conv2D/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?/@?/H?/bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4c_se_reduce/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?/@?/H?/bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6b_se_reduce/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?/@?/H?/bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6c_se_reduce/Sigmoidhu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?/@?/H?/XbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5a_se_reduce/Conv2Dhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?/@?/H?/bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5a_se_expand/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?/@?/H?/bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5b_se_expand/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?/@?/H?/bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5c_se_expand/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?/@?/H?/bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_se_expand/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?2$8?/@?/H?/bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6b_se_expand/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?2$8?/@?/H?/bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6c_se_expand/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?2$8?/@?/H?/bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6d_se_expand/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?2$8?/@?/H?/bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_se_expand/BiasAddhu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?/@?/H?/XbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5b_se_reduce/Conv2Dhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?/@?/H?/bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5c_se_reduce/BiasAddhu  ?B
?
Sqrt_GPU_DT_HALF_DT_HALF_kernel*?28?/@?/H?/bJModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/normalization/Sqrthu  ?B
?

?	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_rsqrt_op<float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_rsqrt_op<float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const, Eigen::GpuDevice>, long)*?28?/@?/H?/bmgradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_project_bn/FusedBatchNormGradV3hu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?/@?/H?/XbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4c_se_reduce/Conv2Dhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?/@?/H?/bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5b_se_reduce/BiasAddhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *?28?/@?/H?/bAll_5hu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?.@?.H?.bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6c_se_reduce/BiasAddhu  ?B
?
?void tensorflow::functor::CleanupSegments<bool*, bool*, tensorflow::functor::And>(bool*, bool*, int, int, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type)* 28?.@?.H?.bAll_10hu  HB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?.@?.H?.bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block1a_se_expand/Sigmoidhu  ?B
?
 Cast_GPU_DT_HALF_DT_FLOAT_kernel*?2?8?.@?.H?.bhgradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_se_reduce/Conv2D/Cast/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?.@?.H?.bdiv_no_nan_1hu  ?B
?
?void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)2 *?28?.@?.H?.bAll_4hu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?.@?.H?.XbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4b_se_reduce/Conv2Dhu  ?B
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?.@?.H?.bmul_4hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?.@?.H?.b3sparse_categorical_crossentropy/weighted_loss/valuehu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?.@?.H?.bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5a_se_reduce/BiasAddhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2J8?.@?.H?.bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_se_reduce/Conv2D/Casthu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?.@?.H?.bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4c_se_reduce/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28?.@?.H?.bgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorGeneratorOp<tensorflow::generator::SparseXentGradGenerator<float, long>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorGeneratorOp<tensorflow::generator::SparseXentGradGenerator<float, long>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28?.@?.H?.bgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshu  ?B
?
?void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *?28?.@?.H?.bAll_6hu  ?B
?
 Cast_GPU_DT_HALF_DT_FLOAT_kernel*?2?8?-@?-H?-bhgradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_se_expand/Conv2D/Cast/Casthu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?-@?-H?-bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6d_se_reduce/mulhu  ?B
S
Sub_GPU_DT_HALF_DT_HALF_kernel*?28?-@?-H?-bAdam/gradients/sub_1hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?-@?-H?-b
div_no_nanhu  ?B
?
?void tensorflow::functor::CleanupSegments<bool*, bool*, tensorflow::functor::And>(bool*, bool*, int, int, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type)* 28?-@?-H?-bAllhu  HB
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*?28?-@?-H?-bIsFinite_11hu  ?B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?-@?-H?-bmul_13hu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?-@?-H?-bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_se_expand/BiasAddhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?-@?-H?-bAssignAddVariableOp_2hu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?-@?-H?-bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3a_se_reduce/BiasAddhu  ?B
?
?void tensorflow::functor::CleanupSegments<bool*, bool*, tensorflow::functor::And>(bool*, bool*, int, int, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type)* 28?-@?-H?-bAll_2hu  HB
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?-@?-H?-bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4b_se_expand/BiasAdd/Casthu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?-@?-H?-bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4b_se_reduce/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?-@?-H?-bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4a_se_expand/BiasAddhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?,@?,H?,bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3b_se_reduce/mulhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?,@?,H?,bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_se_expand/BiasAdd/Casthu  ?B
D
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?,@?,H?,bMulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?,@?,H?,bLgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nanhu  ?B
X
"AddV2_GPU_DT_INT64_DT_INT64_kernel*?28?,@?,H?,bcond/then/_0/cond/addhu  ?B
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?,@?,H?,bmul_7hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?,@?,H?,bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6b_se_reduce/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?,@?,H?,bAssignAddVariableOphu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?,@?,H?,bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_se_reduce/BiasAddhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?,@?,H?,bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_se_reduce/mulhu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?,@?,H?,XbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6b_se_reduce/Conv2Dhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2J8?+@?+H?+bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5c_se_reduce/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2q8?+@?+H?+bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6b_dwconv/depthwise/Casthu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?+@?+H?+bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3a_se_reduce/mulhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?+@?+H?+bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3b_se_reduce/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?+@?+H?+bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4c_se_expand/BiasAddhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?+@?+H?+bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4b_se_reduce/Sigmoidhu  ?B
S
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?+@?+H?+bAdam/gradients/mul_4hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?+@?+H?+bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2b_se_reduce/mulhu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?+@?+H?+XbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_se_reduce/Conv2Dhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?+@?+H?+bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2b_se_reduce/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2B8?+@?+H?+bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5b_dwconv/depthwise/Casthu  ?B
?
!Cast_GPU_DT_FLOAT_DT_INT64_kernel*?28?+@?+H?+bbsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1hu  ?B
P
%LogicalAnd_GPU_DT_BOOL_DT_BOOL_kernel*?28?+@?+H?+b
LogicalAndhu  ?B
`
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?+@?+H?+bcond_1/then/_10/cond_1/Adam/Powhu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?+@?+H?+XbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6d_se_reduce/Conv2Dhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?+@?+H?+bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_se_reduce/Conv2D/Casthu  ?B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?+@?+H?+bmul_11hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?+@?+H?+bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5b_se_reduce/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?+@?+H?+bAssignAddVariableOp_1hu  ?B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*?268?*@?*H?*b
IsFinite_2hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?*@?*H?*bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5a_se_reduce/Sigmoidhu  ?B
S
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?*@?*H?*bAdam/gradients/mul_5hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?*@?*H?*bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6c_se_reduce/BiasAdd/Casthu  ?B
S
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?*@?*H?*bAdam/gradients/mul_3hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?*@?*H?*bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3a_se_expand/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?*@?*H?*bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6b_se_expand/BiasAdd/Casthu  ?B
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?*@?*H?*bmul_5hu  ?B
?
?void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *?28?*@?*H?*bAll_12hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?*@?*H?*bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2b_dwconv/depthwise/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2q8?*@?*H?*bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6d_dwconv/depthwise/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2J8?)@?)H?)bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5b_se_expand/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?)@?)H?)bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_se_expand/BiasAdd/Casthu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?)@?)H?)bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_se_reduce/mulhu  ?B
O
'Reciprocal_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?)@?)H?)btruedivhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2&8?)@?)H?)bWModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4a_expand_conv/Conv2D/Casthu  ?B
b
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?)@?)H?)b!cond_1/then/_10/cond_1/Adam/Pow_1hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?)@?)H?)bXModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4b_project_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?)@?)H?)bWModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5a_expand_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?)@?)H?)bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6b_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?)@?)H?)bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4b_dwconv/depthwise/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?)@?)H?)bAssignAddVariableOp_3hu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?)@?)H?)bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2b_se_reduce/BiasAddhu  ?B
?
 Cast_GPU_DT_HALF_DT_FLOAT_kernel*?28?)@?)H?)bNgradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/dense/BiasAdd/Cast/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2B8?)@?)H?)bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_dwconv/depthwise/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2&8?(@?(H?(bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4c_se_reduce/Conv2D/Casthu  ?B
d
!Cast_GPU_DT_INT64_DT_FLOAT_kernel*?28?(@?(H?(b"cond_1/then/_10/cond_1/Adam/Cast_1hu  ?B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*?28?(@?(H?(b
IsFinite_1hu  ?B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*?28?(@?(H?(b
IsFinite_3hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?(@?(H?(bWModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2b_expand_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?(@?(H?(bXModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2b_project_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_HALF_DT_FLOAT_kernel*?28?(@?(H?(byModel0_FineTuned_10Layers_SixEpochsTotal/softmax_float32/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Casthu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?(@?(H?(bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5c_se_reduce/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?(@?(H?(b>cond/then/_0/cond/cond/else/_195/cond/cond/AssignAddVariableOphu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?(@?(H?(bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block1a_se_reduce/BiasAddhu  ?B
?
 Cast_GPU_DT_HALF_DT_FLOAT_kernel*?28?(@?(H?(bigradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_se_reduce/BiasAdd/Cast/Casthu  ?B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?(@?(H?(bmul_10hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?(@?(H?(bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5c_se_reduce/Sigmoidhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?(@?(H?(bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_se_reduce/BiasAddhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?28?(@?(H?(b1sparse_categorical_crossentropy/weighted_loss/Sumhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2&8?'@?'H?'bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4b_se_reduce/Conv2D/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?'@?'H?'bagradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/top_bn/FusedBatchNormGradV3hu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?'@?'H?'bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2b_se_expand/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?'@?'H?'bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4a_se_reduce/BiasAddhu  ?B
?

?	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_rsqrt_op<float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_rsqrt_op<float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const, Eigen::GpuDevice>, long)*?28?'@?'H?'bagradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/top_bn/FusedBatchNormGradV3hu  ?B
?
"Maximum_GPU_DT_HALF_DT_HALF_kernel*?28?&@?&H?&bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/normalization/Maximumhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?&@?&H?&b6Model0_FineTuned_10Layers_SixEpochsTotal/dense/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?&@?&H?&bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3a_se_expand/BiasAddhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?&@?&H?&bPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block1a_dwconv/depthwisehu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2J8?&@?&H?&bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5b_se_reduce/Conv2D/Casthu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?&@?&H?&bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5b_se_reduce/mulhu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?&@?&H?&XbPModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6c_se_reduce/Conv2Dhu  ?B
U
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?28?%@?%H?%bAdam/gradients/add_1hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?%@?%H?%bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2b_se_reduce/Sigmoidhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?%@?%H?%bWModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4c_expand_conv/Conv2D/Casthu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?%@?%H?%Xbrgradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_se_expand/Conv2D/Conv2DBackpropInputhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?%@?%H?%bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block1a_se_expand/BiasAddhu  ?B
?
?void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Max>(float const*, float*, int, int, cub::Max, std::iterator_traits<float const*>::value_type)**?28?%@?%H?%bgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2&8?%@?%H?%bWModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3b_expand_conv/Conv2D/Casthu  ?B
F
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28?%@?%H?%bCasthu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?%@?%H?%bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_se_reduce/BiasAddhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?%@?%H?%bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5a_se_reduce/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?%@?%H?%bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4b_se_reduce/mulhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?%@?%H?%bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3b_se_expand/BiasAddhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *?28?%@?%H?%bAll_1hu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?28?%@?%H?%bSum_2hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2&8?%@?%H?%bXModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3b_project_conv/Conv2D/Casthu  ?B
g
(GreaterEqual_GPU_DT_INT64_DT_BOOL_kernel*?28?%@?%H?%bcond/then/_0/cond/GreaterEqualhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?%@?%H?%bmgradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_project_bn/FusedBatchNormGradV3hu  ?B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*?28?$@?$H?$b
IsFinite_5hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2B8?$@?$H?$bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5c_dwconv/depthwise/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?$@?$H?$bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3b_se_reduce/Sigmoidhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2J8?$@?$H?$bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5c_se_expand/Conv2D/Casthu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?$@?$H?$bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6b_se_reduce/BiasAddhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2
8?$@?$H?$bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3b_se_expand/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2
8?$@?$H?$bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4a_se_reduce/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?$@?$H?$bWModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3a_expand_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?$@?$H?$bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4c_dwconv/depthwise/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?$@?$H?$bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3a_se_reduce/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?$@?$H?$bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_se_reduce/Sigmoidhu  ?B
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?$@?$H?$bmul_8hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?$@?$H?$bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block1a_se_reduce/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?$@?$H?$bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4c_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?$@?$H?$bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/stem_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?$@?$H?$bWModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_expand_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2	8?$@?$H?$bXModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_project_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?#@?#H?#bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3b_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?#@?#H?#bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4a_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?#@?#H?#bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5a_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?#@?#H?#bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5c_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?#@?#H?#bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3b_dwconv/depthwise/Casthu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?#@?#H?#bUgradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mulhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?#@?#H?#bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?#@?#H?#bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2b_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?#@?#H?#bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5a_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2J8?#@?#H?#bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_se_expand/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?#@?#H?#bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block1a_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2&8?#@?#H?#bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4c_se_expand/Conv2D/Casthu  ?B
M
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*?268?#@?#H?#bIsFinitehu  ?B
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?#@?#H?#bmul_3hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?#@?#H?#bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2&8?#@?#H?#bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4b_se_expand/Conv2D/Casthu  ?B
G
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*?28?"@?"H?"bCast_3hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?"@?"H?"bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3a_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?"@?"H?"bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6d_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?"@?"H?"bXModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4c_project_conv/Conv2D/Casthu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?"@?"H?"bQModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4c_se_reduce/BiasAddhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?"@?"H?"bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3b_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?"@?"H?"bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5b_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?"@?"H?"bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block1a_dwconv/depthwise/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?"@?"H?"bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3a_dwconv/depthwise/Casthu  ?B
?
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28?"@?"H?"b?sparse_categorical_crossentropy/weighted_loss/num_elements/Casthu  ?B
?
 Cast_GPU_DT_INT32_DT_HALF_kernel*?28?"@?"H?"bTgradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/global_average_pooling2d/Casthu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?"@?"H?"bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6c_se_reduce/mulhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?"@?"H?"bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4b_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2K8?"@?"H?"bXModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4a_project_conv/Conv2D/Casthu  ?B
H
!Cast_GPU_DT_INT64_DT_FLOAT_kernel*?28?"@?"H?"bCast_2hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?"@?"H?"bAssignAddVariableOp_4hu  ?B
?
 Cast_GPU_DT_HALF_DT_FLOAT_kernel*?28?!@?!H?!bigradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_se_expand/BiasAdd/Cast/Casthu  ?B
?
?void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *?28?!@?!H?!bAll_11hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?!@?!H?!bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2b_se_expand/Conv2D/Casthu  ?B
b
"AddV2_GPU_DT_INT64_DT_INT64_kernel*?28?!@?!H?!bcond_1/then/_10/cond_1/Adam/addhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?!@?!H?!bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3a_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2&8?!@?!H?!bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5a_se_expand/Conv2D/Casthu  ?B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*?28?!@?!H?!b
IsFinite_8hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block1a_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2)8? @? H? bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block7a_dwconv/depthwise/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2/8? @? H? bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5a_dwconv/depthwise/Casthu  ?B
|
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? b;Model0_FineTuned_10Layers_SixEpochsTotal/dense/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4c_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5b_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6c_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2	8? @? H? bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4a_dwconv/depthwise/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2
8? @? H? bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3b_se_reduce/Conv2D/Casthu  ?B
G
!Equal_GPU_DT_FLOAT_DT_BOOL_kernel*?28? @? H? bEqualhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block1a_se_expand/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2b_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4a_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5c_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bVModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6d_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bXModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block1a_project_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3a_se_reduce/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2
8? @? H? bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4a_se_expand/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bXModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3a_project_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2&8? @? H? bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block5a_se_reduce/Conv2D/Casthu  ?B
H
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28? @? H? bCast_4hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28? @? H? bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block1a_se_reduce/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28? @? H? bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_se_reduce/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28? @? H? bMModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block4a_se_reduce/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28? @? H? b4cond_1/then/_10/cond_1/Adam/Adam/AssignAddVariableOphu  ?B
~
 Cast_GPU_DT_HALF_DT_FLOAT_kernel*?28?@?H?b=Model0_FineTuned_10Layers_SixEpochsTotal/softmax_float32/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?@?H?bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_se_expand/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?@?H?bUModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_dwconv/depthwise/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?@?H?b?gradient_tape/Model0_FineTuned_10Layers_SixEpochsTotal/softmax_float32/Cast/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Casthu  ?B
?
!Cast_GPU_DT_INT64_DT_FLOAT_kernel*?28?@?H?b`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Casthu  ?B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*?28?@?H?b
IsFinite_6hu  ?B