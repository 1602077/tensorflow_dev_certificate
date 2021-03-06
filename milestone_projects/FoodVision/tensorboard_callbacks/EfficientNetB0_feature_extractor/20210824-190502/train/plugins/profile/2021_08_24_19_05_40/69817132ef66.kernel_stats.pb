
?
?void convolve_common_engine_float_NHWC<__half, __half, 128, 6, 7, 3, 3, 5, true, false, false, false, false>(int, int, int, __half const*, __half const*, int, __half*, conv_kernel_common_params, unsigned long long, unsigned long, float, float, int, __half const*, __half const*, bool)C?2* 28???@??H??bEFeatureExtraction_ThreeEpochs/efficientnetb0/block6a_dwconv/depthwiseh?u  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 
8??@@??@H??@bOFeatureExtraction_ThreeEpochs/efficientnetb0/block2a_expand_bn/FusedBatchNormV3hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?n8??:@??:H??:bBFeatureExtraction_ThreeEpochs/efficientnetb0/block2b_se_excite/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?I8??5@??5H??5bJFeatureExtraction_ThreeEpochs/efficientnetb0/block2a_expand_activation/mulhu  ?B
?
?void convolve_common_engine_float_NHWC<__half, __half, 128, 5, 5, 3, 3, 3, true, false, false, false, false>(int, int, int, __half const*, __half const*, int, __half*, conv_kernel_common_params, unsigned long long, unsigned long, float, float, int, __half const*, __half const*, bool)T?*2?b8??4@??4H??4Xb=FeatureExtraction_ThreeEpochs/efficientnetb0/stem_conv/Conv2Dhu  zB
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?b8׆4@׆4H׆4bBFeatureExtraction_ThreeEpochs/efficientnetb0/block1a_se_excite/mulhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?28??+@??+H??+bDFeatureExtraction_ThreeEpochs/efficientnetb0/block1a_se_squeeze/Meanhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)(*?2(8??*@??*H??*bCFeatureExtraction_ThreeEpochs/efficientnetb0/block2a_dwconv_pad/Padhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??(@??(H??(bNFeatureExtraction_ThreeEpochs/efficientnetb0/block2a_expand_activation/Sigmoidhu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_128x128_nn_align8>(cutlass_75_tensorop_f16_s1688gemm_f16_128x128_nn_align8::Params)? ??*?2?8??'@??'H??'XbGFeatureExtraction_ThreeEpochs/efficientnetb0/block2a_expand_conv/Conv2Dhu  ?A
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?I8??'@??'H??'bBFeatureExtraction_ThreeEpochs/efficientnetb0/block2a_se_excite/mulhu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 5, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)P*?2?-8??&@??&H??&bEFeatureExtraction_ThreeEpochs/efficientnetb0/block3b_dwconv/depthwisehu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 5, 2, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)d*?2?8??!@??!H??!bEFeatureExtraction_ThreeEpochs/efficientnetb0/block3a_dwconv/depthwisehu  HB
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 3, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)0*?2?n8??!@??!H??!bEFeatureExtraction_ThreeEpochs/efficientnetb0/block2b_dwconv/depthwisehu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 5, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)P*?2? 8??@??H??bEFeatureExtraction_ThreeEpochs/efficientnetb0/block5b_dwconv/depthwisehu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 5, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)P*?2? 8ۺ@ۺHۺbEFeatureExtraction_ThreeEpochs/efficientnetb0/block5c_dwconv/depthwisehu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 3, 2, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct):*?2?I8??@??H??bEFeatureExtraction_ThreeEpochs/efficientnetb0/block2a_dwconv/depthwisehu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 3, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)0*?2?b8ۯ@ۯHۯbEFeatureExtraction_ThreeEpochs/efficientnetb0/block1a_dwconv/depthwisehu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?-8??@??H??bBFeatureExtraction_ThreeEpochs/efficientnetb0/block3b_se_excite/mulhu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8::Params)` ??*?2?18ܡ@ܡHܡXbHFeatureExtraction_ThreeEpochs/efficientnetb0/block1a_project_conv/Conv2Dhu  HB
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?	*?2 8ܣ@ܣHܣbOFeatureExtraction_ThreeEpochs/efficientnetb0/block3a_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?	*?2 8??@??H??bHFeatureExtraction_ThreeEpochs/efficientnetb0/block2b_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?	*?2 8ܔ@ܔHܔbOFeatureExtraction_ThreeEpochs/efficientnetb0/block2b_expand_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)(*?2(8??@??H??b>FeatureExtraction_ThreeEpochs/efficientnetb0/stem_conv_pad/Padhu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 5, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)P*?2?8??@??H??bEFeatureExtraction_ThreeEpochs/efficientnetb0/block5a_dwconv/depthwisehu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8ܶ@ܶHܶbJFeatureExtraction_ThreeEpochs/efficientnetb0/block2b_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bCFeatureExtraction_ThreeEpochs/efficientnetb0/block2b_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bJFeatureExtraction_ThreeEpochs/efficientnetb0/block3a_expand_activation/mulhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bHFeatureExtraction_ThreeEpochs/efficientnetb0/block1a_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bEFeatureExtraction_ThreeEpochs/efficientnetb0/stem_bn/FusedBatchNormV3hu  ?B
?
Div_GPU_DT_HALF_DT_HALF_kernel*?2?$8??@??H??bBFeatureExtraction_ThreeEpochs/efficientnetb0/normalization/truedivhu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 5, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)P*?2?8??@??H??bEFeatureExtraction_ThreeEpochs/efficientnetb0/block6b_dwconv/depthwisehu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 5, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)P*?2?8??@??H??bEFeatureExtraction_ThreeEpochs/efficientnetb0/block6c_dwconv/depthwisehu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 5, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)P*?2?8??@??H??bEFeatureExtraction_ThreeEpochs/efficientnetb0/block6d_dwconv/depthwisehu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8::Params)` ??*?2?18??@??H??XbGFeatureExtraction_ThreeEpochs/efficientnetb0/block2b_expand_conv/Conv2Dhu  HB
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8::Params)` ??*?2?18??@??H??XbGFeatureExtraction_ThreeEpochs/efficientnetb0/block3a_expand_conv/Conv2Dhu  HB
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8݋@݋H݋b@FeatureExtraction_ThreeEpochs/efficientnetb0/stem_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8܅@܅H܅bCFeatureExtraction_ThreeEpochs/efficientnetb0/block1a_activation/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)(*?2(8??@??H??bCFeatureExtraction_ThreeEpochs/efficientnetb0/block3a_dwconv_pad/Padhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2? 8݈@݈H݈bBFeatureExtraction_ThreeEpochs/efficientnetb0/block5b_se_excite/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2? 8??@??H??bBFeatureExtraction_ThreeEpochs/efficientnetb0/block5c_se_excite/mulhu  ?B
?
Sub_GPU_DT_HALF_DT_HALF_kernel*?2?$8??@??H??b>FeatureExtraction_ThreeEpochs/efficientnetb0/normalization/subhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bNFeatureExtraction_ThreeEpochs/efficientnetb0/block2b_expand_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bGFeatureExtraction_ThreeEpochs/efficientnetb0/block2b_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bNFeatureExtraction_ThreeEpochs/efficientnetb0/block3a_expand_activation/Sigmoidhu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x128_nn_align8>(cutlass_75_tensorop_f16_s1688gemm_f16_64x128_nn_align8::Params)? ??*?2?8??@??H??XbHFeatureExtraction_ThreeEpochs/efficientnetb0/block2b_project_conv/Conv2Dhu  ?A
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 
8??@??H??bHFeatureExtraction_ThreeEpochs/efficientnetb0/block2a_bn/FusedBatchNormV3hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bBFeatureExtraction_ThreeEpochs/efficientnetb0/block3a_se_excite/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bDFeatureExtraction_ThreeEpochs/efficientnetb0/stem_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bGFeatureExtraction_ThreeEpochs/efficientnetb0/block1a_activation/Sigmoidhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bCFeatureExtraction_ThreeEpochs/efficientnetb0/block2a_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bBFeatureExtraction_ThreeEpochs/efficientnetb0/block4c_se_excite/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bBFeatureExtraction_ThreeEpochs/efficientnetb0/block4b_se_excite/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bBFeatureExtraction_ThreeEpochs/efficientnetb0/block5a_se_excite/mulhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2$8??@??H??bDFeatureExtraction_ThreeEpochs/efficientnetb0/block2b_se_squeeze/Meanhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?28??@??H??bDFeatureExtraction_ThreeEpochs/efficientnetb0/block2a_se_squeeze/Meanhu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8::Params)` ??*?2?8??
@??
H??
XbHFeatureExtraction_ThreeEpochs/efficientnetb0/block2a_project_conv/Conv2Dhu  HB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??
@??
H??
bGFeatureExtraction_ThreeEpochs/efficientnetb0/block2a_activation/Sigmoidhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??
@??
H??
bPFeatureExtraction_ThreeEpochs/efficientnetb0/block1a_project_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??	@??	H??	bOFeatureExtraction_ThreeEpochs/efficientnetb0/block4a_expand_bn/FusedBatchNormV3hu  ?B
?
 Cast_GPU_DT_HALF_DT_FLOAT_kernel*?2??8??	@??	H??	b?FeatureExtraction_ThreeEpochs/efficientnetb0/normalization/Casthu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??	@??	H??	bOFeatureExtraction_ThreeEpochs/efficientnetb0/block3b_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??	@??	H??	bHFeatureExtraction_ThreeEpochs/efficientnetb0/block3b_bn/FusedBatchNormV3hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2??8??	@??	H??	bAFeatureExtraction_ThreeEpochs/efficientnetb0/normalization/Cast_1hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2??8??	@??	H??	b;FeatureExtraction_ThreeEpochs/efficientnetb0/rescaling/Casthu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bJFeatureExtraction_ThreeEpochs/efficientnetb0/block4a_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bCFeatureExtraction_ThreeEpochs/efficientnetb0/block3b_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bJFeatureExtraction_ThreeEpochs/efficientnetb0/block3b_expand_activation/mulhu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 3, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)0*?2?8??@??H??bEFeatureExtraction_ThreeEpochs/efficientnetb0/block4c_dwconv/depthwisehu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 3, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)0*?2?8??@??H??bEFeatureExtraction_ThreeEpochs/efficientnetb0/block4b_dwconv/depthwisehu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?**?2 	8??@??H??bOFeatureExtraction_ThreeEpochs/efficientnetb0/block5c_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?**?2 	8??@??H??bOFeatureExtraction_ThreeEpochs/efficientnetb0/block5b_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?**?2 	8??@??H??bHFeatureExtraction_ThreeEpochs/efficientnetb0/block5c_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?**?2 	8??@??H??bOFeatureExtraction_ThreeEpochs/efficientnetb0/block6a_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?**?2 	8??@??H??bHFeatureExtraction_ThreeEpochs/efficientnetb0/block5b_bn/FusedBatchNormV3hu  ?B
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?218??@??H??PXbGFeatureExtraction_ThreeEpochs/efficientnetb0/block5b_expand_conv/Conv2Dhu  ?A
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?218??@??H??PXbGFeatureExtraction_ThreeEpochs/efficientnetb0/block6a_expand_conv/Conv2Dhu  ?A
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?218??@??H??PXbGFeatureExtraction_ThreeEpochs/efficientnetb0/block5c_expand_conv/Conv2Dhu  ?A
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)(*?2(8??@??H??bCFeatureExtraction_ThreeEpochs/efficientnetb0/block4a_dwconv_pad/Padhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bBFeatureExtraction_ThreeEpochs/efficientnetb0/block6d_se_excite/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bBFeatureExtraction_ThreeEpochs/efficientnetb0/block6c_se_excite/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bBFeatureExtraction_ThreeEpochs/efficientnetb0/block6b_se_excite/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bGFeatureExtraction_ThreeEpochs/efficientnetb0/block3b_activation/Sigmoidhu  ?B
?
7turing_fp16_s1688gemm_fp16_64x128_sliced1x2_ldg8_f2f_nn???*?2?8??@??H??PXbHFeatureExtraction_ThreeEpochs/efficientnetb0/block3b_project_conv/Conv2Dhu  ?A
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bNFeatureExtraction_ThreeEpochs/efficientnetb0/block3b_expand_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bNFeatureExtraction_ThreeEpochs/efficientnetb0/block4a_expand_activation/Sigmoidhu  ?B
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?2?8??@??H??PXbGFeatureExtraction_ThreeEpochs/efficientnetb0/block3b_expand_conv/Conv2Dhu  ?A
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bBFeatureExtraction_ThreeEpochs/efficientnetb0/block7a_se_excite/mulhu  ?B
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?2?8??@??H??PXbGFeatureExtraction_ThreeEpochs/efficientnetb0/block4a_expand_conv/Conv2Dhu  ?A
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)(*?2(8??@??H??bCFeatureExtraction_ThreeEpochs/efficientnetb0/block6a_dwconv_pad/Padhu  ?B
?
:turing_fp16_s1688gemm_fp16_128x256_ldg8_f2f_stages_32x1_nn??? ??*?2
8??@??H??PXb<FeatureExtraction_ThreeEpochs/efficientnetb0/top_conv/Conv2Dh
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 3, 2, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct):*?2?8??@??H??bEFeatureExtraction_ThreeEpochs/efficientnetb0/block4a_dwconv/depthwisehu  ?B
?
.turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_nn???*?218??@??H??PXbHFeatureExtraction_ThreeEpochs/efficientnetb0/block5c_project_conv/Conv2Dhu  ?A
?
.turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_nn???*?218߬@߬H߬PXbHFeatureExtraction_ThreeEpochs/efficientnetb0/block5b_project_conv/Conv2Dhu  ?A
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bJFeatureExtraction_ThreeEpochs/efficientnetb0/block5c_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bJFeatureExtraction_ThreeEpochs/efficientnetb0/block6a_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8ߜ@ߜHߜbCFeatureExtraction_ThreeEpochs/efficientnetb0/block5c_activation/mulhu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 3, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)0*?2?8ߛ@ߛHߛbEFeatureExtraction_ThreeEpochs/efficientnetb0/block7a_dwconv/depthwisehu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8ߙ@ߙHߙbCFeatureExtraction_ThreeEpochs/efficientnetb0/block5b_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bJFeatureExtraction_ThreeEpochs/efficientnetb0/block5b_expand_activation/mulhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?	*?2 8??@??H??bHFeatureExtraction_ThreeEpochs/efficientnetb0/block3a_bn/FusedBatchNormV3hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bBFeatureExtraction_ThreeEpochs/efficientnetb0/block4a_se_excite/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bCFeatureExtraction_ThreeEpochs/efficientnetb0/block3a_activation/mulhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bOFeatureExtraction_ThreeEpochs/efficientnetb0/block5a_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bOFeatureExtraction_ThreeEpochs/efficientnetb0/block4b_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bOFeatureExtraction_ThreeEpochs/efficientnetb0/block4c_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bHFeatureExtraction_ThreeEpochs/efficientnetb0/block4b_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bHFeatureExtraction_ThreeEpochs/efficientnetb0/block5a_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bHFeatureExtraction_ThreeEpochs/efficientnetb0/block4c_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bGFeatureExtraction_ThreeEpochs/efficientnetb0/block5b_activation/Sigmoidhu  ?B
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?28??@??H??PXbHFeatureExtraction_ThreeEpochs/efficientnetb0/block7a_project_conv/Conv2Dhu  ?A
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bGFeatureExtraction_ThreeEpochs/efficientnetb0/block5c_activation/Sigmoidhu  ?B
}
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?	8??@??H??b:FeatureExtraction_ThreeEpochs/efficientnetb0/rescaling/mulhu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8::Params)` ??*?2?8??@??H??XbHFeatureExtraction_ThreeEpochs/efficientnetb0/block3a_project_conv/Conv2Dhu  HB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bNFeatureExtraction_ThreeEpochs/efficientnetb0/block5b_expand_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bNFeatureExtraction_ThreeEpochs/efficientnetb0/block6a_expand_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bNFeatureExtraction_ThreeEpochs/efficientnetb0/block5c_expand_activation/Sigmoidhu  ?B
?
.turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_nn???*?218??@??H??PXbHFeatureExtraction_ThreeEpochs/efficientnetb0/block5a_project_conv/Conv2Dhu  ?A
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bJFeatureExtraction_ThreeEpochs/efficientnetb0/block5a_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bJFeatureExtraction_ThreeEpochs/efficientnetb0/block4c_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bJFeatureExtraction_ThreeEpochs/efficientnetb0/block4b_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bCFeatureExtraction_ThreeEpochs/efficientnetb0/block5a_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bCFeatureExtraction_ThreeEpochs/efficientnetb0/block4b_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bCFeatureExtraction_ThreeEpochs/efficientnetb0/block4c_activation/mulhu  ?B
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?218??@??H??PXbHFeatureExtraction_ThreeEpochs/efficientnetb0/block4b_project_conv/Conv2Dhu  ?A
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?218??@??H??PXbHFeatureExtraction_ThreeEpochs/efficientnetb0/block4c_project_conv/Conv2Dhu  ?A
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2<8??@??H??bDFeatureExtraction_ThreeEpochs/efficientnetb0/block3b_se_squeeze/Meanhu  ?B
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?218??@??H??PXbGFeatureExtraction_ThreeEpochs/efficientnetb0/block4b_expand_conv/Conv2Dhu  ?A
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?218??@??H??PXbGFeatureExtraction_ThreeEpochs/efficientnetb0/block4c_expand_conv/Conv2Dhu  ?A
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?218??@??H??PXbGFeatureExtraction_ThreeEpochs/efficientnetb0/block5a_expand_conv/Conv2Dhu  ?A
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?P*?2 8??@??H??bDFeatureExtraction_ThreeEpochs/efficientnetb0/top_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bGFeatureExtraction_ThreeEpochs/efficientnetb0/block3a_activation/Sigmoidhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bBFeatureExtraction_ThreeEpochs/efficientnetb0/block6a_se_excite/mulhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 
8??@??H??bPFeatureExtraction_ThreeEpochs/efficientnetb0/block2a_project_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 
8߃@߃H߃bPFeatureExtraction_ThreeEpochs/efficientnetb0/block2b_project_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bGFeatureExtraction_ThreeEpochs/efficientnetb0/block4c_activation/Sigmoidhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2$8??@??H??bDFeatureExtraction_ThreeEpochs/efficientnetb0/block3a_se_squeeze/Meanhu  ?B
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?2	8??@??H??PXbGFeatureExtraction_ThreeEpochs/efficientnetb0/block6d_expand_conv/Conv2Dhu  ?A
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?2	8??@??H??PXbGFeatureExtraction_ThreeEpochs/efficientnetb0/block6b_expand_conv/Conv2Dhu  ?A
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?2	8??@??H??PXbGFeatureExtraction_ThreeEpochs/efficientnetb0/block6c_expand_conv/Conv2Dhu  ?A
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?2	8??@??H??PXbGFeatureExtraction_ThreeEpochs/efficientnetb0/block7a_expand_conv/Conv2Dhu  ?A
?
7turing_fp16_s1688gemm_fp16_64x128_sliced1x2_ldg8_f2f_nn???*?28??@??H??PXbHFeatureExtraction_ThreeEpochs/efficientnetb0/block6b_project_conv/Conv2Dhu  ?A
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bNFeatureExtraction_ThreeEpochs/efficientnetb0/block4c_expand_activation/Sigmoidhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?H*?2 8??@??H??bOFeatureExtraction_ThreeEpochs/efficientnetb0/block6b_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?H*?2 8??@??H??bOFeatureExtraction_ThreeEpochs/efficientnetb0/block6d_expand_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bNFeatureExtraction_ThreeEpochs/efficientnetb0/block5a_expand_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bNFeatureExtraction_ThreeEpochs/efficientnetb0/block4b_expand_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bGFeatureExtraction_ThreeEpochs/efficientnetb0/block4b_activation/Sigmoidhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?H*?2 8??@??H??bOFeatureExtraction_ThreeEpochs/efficientnetb0/block7a_expand_bn/FusedBatchNormV3hu  ?B
?
7turing_fp16_s1688gemm_fp16_64x128_sliced1x2_ldg8_f2f_nn???*?28??@??H??PXbHFeatureExtraction_ThreeEpochs/efficientnetb0/block6d_project_conv/Conv2Dhu  ?A
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bGFeatureExtraction_ThreeEpochs/efficientnetb0/block5a_activation/Sigmoidhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?H*?2 8??@??H??bOFeatureExtraction_ThreeEpochs/efficientnetb0/block6c_expand_bn/FusedBatchNormV3hu  ?B
?
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??b<FeatureExtraction_ThreeEpochs/efficientnetb0/block2b_add/addhu  ?B
?
7turing_fp16_s1688gemm_fp16_64x128_sliced1x2_ldg8_f2f_nn???*?28??@??H??PXbHFeatureExtraction_ThreeEpochs/efficientnetb0/block6c_project_conv/Conv2Dhu  ?A
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?H*?2 8??@??H??bHFeatureExtraction_ThreeEpochs/efficientnetb0/block7a_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?H*?2 8??@??H??bHFeatureExtraction_ThreeEpochs/efficientnetb0/block6b_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?H*?2 8??@??H??bHFeatureExtraction_ThreeEpochs/efficientnetb0/block6c_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?H*?2 8??@??H??bHFeatureExtraction_ThreeEpochs/efficientnetb0/block6d_bn/FusedBatchNormV3hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??b?FeatureExtraction_ThreeEpochs/efficientnetb0/top_activation/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorGeneratorOp<tensorflow::generator::SparseXentLossGenerator<float, long>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorGeneratorOp<tensorflow::generator::SparseXentLossGenerator<float, long>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)>*?28ߎ@ߎHߎbgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2?8??@??H??bDFeatureExtraction_ThreeEpochs/efficientnetb0/block5b_se_squeeze/Meanhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2?8??@??H??bDFeatureExtraction_ThreeEpochs/efficientnetb0/block5c_se_squeeze/Meanhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bJFeatureExtraction_ThreeEpochs/efficientnetb0/block7a_expand_activation/mulhu  ?B
?
.turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_nn???*?218??@??H??PXbHFeatureExtraction_ThreeEpochs/efficientnetb0/block4a_project_conv/Conv2Dhu  ?A
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bCFeatureExtraction_ThreeEpochs/efficientnetb0/block6c_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bJFeatureExtraction_ThreeEpochs/efficientnetb0/block6b_expand_activation/mulhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bHFeatureExtraction_ThreeEpochs/efficientnetb0/block4a_bn/FusedBatchNormV3hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bJFeatureExtraction_ThreeEpochs/efficientnetb0/block6c_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bCFeatureExtraction_ThreeEpochs/efficientnetb0/block7a_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bJFeatureExtraction_ThreeEpochs/efficientnetb0/block6d_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bCFeatureExtraction_ThreeEpochs/efficientnetb0/block6b_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bCFeatureExtraction_ThreeEpochs/efficientnetb0/block6d_activation/mulhu  ?B
?
7turing_fp16_s1688gemm_fp16_64x128_sliced1x2_ldg8_f2f_nn???*?28??@??H??PXbHFeatureExtraction_ThreeEpochs/efficientnetb0/block6a_project_conv/Conv2Dhu  ?A
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bCFeatureExtraction_ThreeEpochs/efficientnetb0/top_activation/Sigmoidhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bCFeatureExtraction_ThreeEpochs/efficientnetb0/block4a_activation/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8ߜ@ߜHߜbNFeatureExtraction_ThreeEpochs/efficientnetb0/block6d_expand_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bNFeatureExtraction_ThreeEpochs/efficientnetb0/block6c_expand_activation/Sigmoidhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2x8??@??H??bDFeatureExtraction_ThreeEpochs/efficientnetb0/block5a_se_squeeze/Meanhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2x8??@??H??bDFeatureExtraction_ThreeEpochs/efficientnetb0/block4b_se_squeeze/Meanhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bGFeatureExtraction_ThreeEpochs/efficientnetb0/block6d_activation/Sigmoidhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?**?2 8??@??H??bHFeatureExtraction_ThreeEpochs/efficientnetb0/block6a_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bNFeatureExtraction_ThreeEpochs/efficientnetb0/block7a_expand_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bGFeatureExtraction_ThreeEpochs/efficientnetb0/block6b_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bGFeatureExtraction_ThreeEpochs/efficientnetb0/block7a_activation/Sigmoidhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2x8??@??H??bDFeatureExtraction_ThreeEpochs/efficientnetb0/block4c_se_squeeze/Meanhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bNFeatureExtraction_ThreeEpochs/efficientnetb0/block6b_expand_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bGFeatureExtraction_ThreeEpochs/efficientnetb0/block6c_activation/Sigmoidhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bPFeatureExtraction_ThreeEpochs/efficientnetb0/block3b_project_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bPFeatureExtraction_ThreeEpochs/efficientnetb0/block3a_project_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bGFeatureExtraction_ThreeEpochs/efficientnetb0/block4a_activation/Sigmoidhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bPFeatureExtraction_ThreeEpochs/efficientnetb0/block5a_project_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bPFeatureExtraction_ThreeEpochs/efficientnetb0/block5b_project_bn/FusedBatchNormV3hu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2::Params)? ??*?28??@??H??XbEFeatureExtraction_ThreeEpochs/efficientnetb0/block4b_se_expand/Conv2Dhu  ?A
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bPFeatureExtraction_ThreeEpochs/efficientnetb0/block5c_project_bn/FusedBatchNormV3hu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2::Params)? ??*?28??@??H??XbEFeatureExtraction_ThreeEpochs/efficientnetb0/block6a_se_expand/Conv2Dhu  ?A
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2::Params)? ??*?28߼@߼H߼XbEFeatureExtraction_ThreeEpochs/efficientnetb0/block4c_se_expand/Conv2Dhu  ?A
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2::Params)? ??*?28??@??H??XbEFeatureExtraction_ThreeEpochs/efficientnetb0/block5a_se_expand/Conv2Dhu  ?A
?
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??b<FeatureExtraction_ThreeEpochs/efficientnetb0/block3b_add/addhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2?8??@??H??b;FeatureExtraction_ThreeEpochs/global_average_pooling2d/Meanhu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2::Params)? ??*?28??@??H??XbEFeatureExtraction_ThreeEpochs/efficientnetb0/block5b_se_expand/Conv2Dhu  ?A
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2::Params)? ??*?28??@??H??XbEFeatureExtraction_ThreeEpochs/efficientnetb0/block5c_se_expand/Conv2Dhu  ?A
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)6*?28??@??H??bgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bGFeatureExtraction_ThreeEpochs/efficientnetb0/block6a_activation/Sigmoidhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bPFeatureExtraction_ThreeEpochs/efficientnetb0/block4a_project_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bPFeatureExtraction_ThreeEpochs/efficientnetb0/block7a_project_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8ߟ@ߟHߟbPFeatureExtraction_ThreeEpochs/efficientnetb0/block6c_project_bn/FusedBatchNormV3hu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2<8??@??H??bDFeatureExtraction_ThreeEpochs/efficientnetb0/block4a_se_squeeze/Meanhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bPFeatureExtraction_ThreeEpochs/efficientnetb0/block6b_project_bn/FusedBatchNormV3hu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?2(8??@??H??b9cond_1/then/_10/cond_1/Adam/Adam/update/ResourceApplyAdamhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bCFeatureExtraction_ThreeEpochs/efficientnetb0/block6a_activation/mulhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8ߔ@ߔHߔbPFeatureExtraction_ThreeEpochs/efficientnetb0/block6d_project_bn/FusedBatchNormV3hu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2?8??@??H??bDFeatureExtraction_ThreeEpochs/efficientnetb0/block7a_se_squeeze/Meanhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2?8??@??H??bDFeatureExtraction_ThreeEpochs/efficientnetb0/block6b_se_squeeze/Meanhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bPFeatureExtraction_ThreeEpochs/efficientnetb0/block6a_project_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long) *?28??@??H??bArgMaxhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2?8??@??H??bDFeatureExtraction_ThreeEpochs/efficientnetb0/block6c_se_squeeze/Meanhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2?8??@??H??bDFeatureExtraction_ThreeEpochs/efficientnetb0/block6d_se_squeeze/Meanhu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align1>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align1::Params)q ??*?2
8??@??H??Xb*FeatureExtraction_ThreeEpochs/dense/MatMulhu  HB
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bPFeatureExtraction_ThreeEpochs/efficientnetb0/block4c_project_bn/FusedBatchNormV3hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8??@??H??bAFeatureExtraction_ThreeEpochs/efficientnetb0/top_conv/Conv2D/Casthu  ?B
?
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??b<FeatureExtraction_ThreeEpochs/efficientnetb0/block5b_add/addhu  ?B
?
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??b<FeatureExtraction_ThreeEpochs/efficientnetb0/block5c_add/addhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8?{@?{H?{bPFeatureExtraction_ThreeEpochs/efficientnetb0/block4b_project_bn/FusedBatchNormV3hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?t@?tH?tbMFeatureExtraction_ThreeEpochs/efficientnetb0/block7a_project_conv/Conv2D/Casthu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2::Params)` ??*?28?q@?qH?qXbEFeatureExtraction_ThreeEpochs/efficientnetb0/block5b_se_reduce/Conv2Dhu  HB
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2::Params)` ??*?28?q@?qH?qXbEFeatureExtraction_ThreeEpochs/efficientnetb0/block5c_se_reduce/Conv2Dhu  HB
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2::Params)` ??*?28?o@?oH?oXbEFeatureExtraction_ThreeEpochs/efficientnetb0/block6a_se_reduce/Conv2Dhu  HB
}
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2{8?o@?oH?ob<FeatureExtraction_ThreeEpochs/efficientnetb0/block4b_add/addhu  ?B
}
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2{8?o@?oH?ob<FeatureExtraction_ThreeEpochs/efficientnetb0/block4c_add/addhu  ?B
?
dvoid tensorflow::BiasGradNHWC_SharedAtomics<Eigen::half>(int, Eigen::half const*, Eigen::half*, int) ?*?28?i@?iH?ibEgradient_tape/FeatureExtraction_ThreeEpochs/dense/BiasAdd/BiasAddGradhu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2::Params)` ??*?28?a@?aH?aXbEFeatureExtraction_ThreeEpochs/efficientnetb0/block4c_se_reduce/Conv2Dhu  HB
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2::Params)` ??*?28?`@?`H?`XbEFeatureExtraction_ThreeEpochs/efficientnetb0/block4b_se_reduce/Conv2Dhu  HB
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8::Params)^ ??*?28?`@?`H?`XbEFeatureExtraction_ThreeEpochs/efficientnetb0/block6d_se_reduce/Conv2Dhu  HA
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8::Params)^ ??*?28?`@?`H?`XbEFeatureExtraction_ThreeEpochs/efficientnetb0/block7a_se_reduce/Conv2Dhu  HA
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?`@?`H?`bMFeatureExtraction_ThreeEpochs/efficientnetb0/block6d_project_conv/Conv2D/Casthu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2::Params)` ??*?28?`@?`H?`XbEFeatureExtraction_ThreeEpochs/efficientnetb0/block5a_se_reduce/Conv2Dhu  HB
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8::Params)^ ??*?28?`@?`H?`XbEFeatureExtraction_ThreeEpochs/efficientnetb0/block6c_se_reduce/Conv2Dhu  HA
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?]@?]H?]XbEFeatureExtraction_ThreeEpochs/efficientnetb0/block4a_se_reduce/Conv2Dhu  HA
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nt_align1>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nt_align1::Params)v ??*?2(8?]@?]H?]b:gradient_tape/FeatureExtraction_ThreeEpochs/dense/MatMul_1hu  HB
?
*volta_fp16_s884gemm_fp16_64x64_ldg8_f2f_nnj??*?28?\@?\H?\PXbEFeatureExtraction_ThreeEpochs/efficientnetb0/block7a_se_expand/Conv2Dhu  B
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8::Params)^ ??*?28?[@?[H?[XbEFeatureExtraction_ThreeEpochs/efficientnetb0/block6b_se_reduce/Conv2Dhu  HA
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?[@?[H?[XbEFeatureExtraction_ThreeEpochs/efficientnetb0/block2b_se_reduce/Conv2Dhu  HA
}
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2J8?[@?[H?[b<FeatureExtraction_ThreeEpochs/efficientnetb0/block6c_add/addhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *?2 8?[@?[H?[bAllhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?Z@?ZH?ZbLFeatureExtraction_ThreeEpochs/efficientnetb0/block6c_expand_conv/Conv2D/Casthu  ?B
?
*volta_fp16_s884gemm_fp16_64x64_ldg8_f2f_nnj??*?28?Z@?ZH?ZPXbEFeatureExtraction_ThreeEpochs/efficientnetb0/block6c_se_expand/Conv2Dhu  B
}
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2J8?Y@?YH?Yb<FeatureExtraction_ThreeEpochs/efficientnetb0/block6d_add/addhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?X@?XH?XbLFeatureExtraction_ThreeEpochs/efficientnetb0/block6b_expand_conv/Conv2D/Casthu  ?B
}
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2J8?V@?VH?Vb<FeatureExtraction_ThreeEpochs/efficientnetb0/block6b_add/addhu  ?B
?
*volta_fp16_s884gemm_fp16_64x64_ldg8_f2f_nnj??*?28?T@?TH?TPXbEFeatureExtraction_ThreeEpochs/efficientnetb0/block6b_se_expand/Conv2Dhu  B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?S@?SH?SbMFeatureExtraction_ThreeEpochs/efficientnetb0/block6c_project_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?R@?RH?RbMFeatureExtraction_ThreeEpochs/efficientnetb0/block6b_project_conv/Conv2D/Casthu  ?B
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?R@?RH?RXbEFeatureExtraction_ThreeEpochs/efficientnetb0/block3b_se_reduce/Conv2Dhu  HA
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2?8?P@?PH?PbDFeatureExtraction_ThreeEpochs/efficientnetb0/block6a_se_squeeze/Meanhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?P@?PH?PbLFeatureExtraction_ThreeEpochs/efficientnetb0/block6d_expand_conv/Conv2D/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?P@?PH?PbFFeatureExtraction_ThreeEpochs/efficientnetb0/block5a_se_expand/Sigmoidhu  ?B
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?P@?PH?PXbEFeatureExtraction_ThreeEpochs/efficientnetb0/block3a_se_reduce/Conv2Dhu  HA
?
*volta_fp16_s884gemm_fp16_64x64_ldg8_f2f_nnj??*?28?P@?PH?PPXbEFeatureExtraction_ThreeEpochs/efficientnetb0/block6d_se_expand/Conv2Dhu  B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?N@?NH?NbLFeatureExtraction_ThreeEpochs/efficientnetb0/block7a_expand_conv/Conv2D/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2$8?M@?MH?MbFFeatureExtraction_ThreeEpochs/efficientnetb0/block6c_se_expand/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2$8?M@?MH?MbFFeatureExtraction_ThreeEpochs/efficientnetb0/block7a_se_expand/Sigmoidhu  ?B
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?L@?LH?LXbEFeatureExtraction_ThreeEpochs/efficientnetb0/block2a_se_expand/Conv2Dhu  HA
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?K@?KH?KXbEFeatureExtraction_ThreeEpochs/efficientnetb0/block2a_se_reduce/Conv2Dhu  HA
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?I@?IH?IbFFeatureExtraction_ThreeEpochs/efficientnetb0/block4a_se_expand/Sigmoidhu  ?B
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?H@?HH?HXbEFeatureExtraction_ThreeEpochs/efficientnetb0/block3a_se_expand/Conv2Dhu  HA
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?F@?FH?FbFFeatureExtraction_ThreeEpochs/efficientnetb0/block3b_se_expand/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?E@?EH?EbFFeatureExtraction_ThreeEpochs/efficientnetb0/block5c_se_expand/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2$8?D@?DH?DbFFeatureExtraction_ThreeEpochs/efficientnetb0/block6b_se_expand/Sigmoidhu  ?B
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?D@?DH?DXbEFeatureExtraction_ThreeEpochs/efficientnetb0/block2b_se_expand/Conv2Dhu  HA
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?D@?DH?DbEFeatureExtraction_ThreeEpochs/efficientnetb0/block6c_dwconv/depthwisehu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?D@?DH?DbEFeatureExtraction_ThreeEpochs/efficientnetb0/block6d_dwconv/depthwisehu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?C@?CH?CbFFeatureExtraction_ThreeEpochs/efficientnetb0/block5b_se_expand/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?B@?BH?BbFFeatureExtraction_ThreeEpochs/efficientnetb0/block6a_se_expand/Sigmoidhu  ?B
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8::Params)^ ??*?28?B@?BH?BXbEFeatureExtraction_ThreeEpochs/efficientnetb0/block1a_se_reduce/Conv2Dhu  HA
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?B@?BH?BbEFeatureExtraction_ThreeEpochs/efficientnetb0/block6a_dwconv/depthwisehu  ?B
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?A@?AH?AXbEFeatureExtraction_ThreeEpochs/efficientnetb0/block4a_se_expand/Conv2Dhu  HA
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?@@?@H?@bEFeatureExtraction_ThreeEpochs/efficientnetb0/block6b_dwconv/depthwisehu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?@@?@H?@bFFeatureExtraction_ThreeEpochs/efficientnetb0/block4b_se_expand/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?@@?@H?@bFFeatureExtraction_ThreeEpochs/efficientnetb0/block4c_se_expand/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2$8?@@?@H?@bFFeatureExtraction_ThreeEpochs/efficientnetb0/block6d_se_expand/Sigmoidhu  ?B
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?@@?@H?@XbEFeatureExtraction_ThreeEpochs/efficientnetb0/block3b_se_expand/Conv2Dhu  HA
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8::Params)^ ??*?28?@@?@H?@XbEFeatureExtraction_ThreeEpochs/efficientnetb0/block1a_se_expand/Conv2Dhu  HA
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8??@??H??bMFeatureExtraction_ThreeEpochs/efficientnetb0/block6a_project_conv/Conv2D/Casthu  ?B
q
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?=@?=H?=b/FeatureExtraction_ThreeEpochs/dense/MatMul/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?<@?<H?<bLFeatureExtraction_ThreeEpochs/efficientnetb0/block5b_expand_conv/Conv2D/Casthu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?;@?;H?;bEFeatureExtraction_ThreeEpochs/efficientnetb0/block5b_dwconv/depthwisehu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?;@?;H?;bEFeatureExtraction_ThreeEpochs/efficientnetb0/block5a_dwconv/depthwisehu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?;@?;H?;bEFeatureExtraction_ThreeEpochs/efficientnetb0/block5c_dwconv/depthwisehu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?;@?;H?;bLFeatureExtraction_ThreeEpochs/efficientnetb0/block5c_expand_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?:@?:H?:bMFeatureExtraction_ThreeEpochs/efficientnetb0/block5b_project_conv/Conv2D/Casthu  ?B
?
?void tensorflow::functor::RowReduceKernel<cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long>, float*, cub::Sum>(cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long>, float*, int, int, cub::Sum, std::iterator_traits<cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long> >::value_type)*?28?:@?:H?:b5FeatureExtraction_ThreeEpochs/softmax_float32/Softmaxhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?9@?9H?9bLFeatureExtraction_ThreeEpochs/efficientnetb0/block6a_expand_conv/Conv2D/Casthu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?9@?9H?9bEFeatureExtraction_ThreeEpochs/efficientnetb0/block4c_dwconv/depthwisehu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?8@?8H?8bEFeatureExtraction_ThreeEpochs/efficientnetb0/block4b_dwconv/depthwisehu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?8@?8H?8bEFeatureExtraction_ThreeEpochs/efficientnetb0/block2b_dwconv/depthwisehu  ?B
?
?void tensorflow::(anonymous namespace)::GenerateNormalizedProb<float, float, 4>(float const*, float const*, float const*, float*, int, int, bool)%*?28?6@?6H?6b5FeatureExtraction_ThreeEpochs/softmax_float32/Softmaxhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?4@?4H?4bFFeatureExtraction_ThreeEpochs/efficientnetb0/block2b_se_expand/Sigmoidhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?4@?4H?4bEFeatureExtraction_ThreeEpochs/efficientnetb0/block3a_dwconv/depthwisehu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?3@?3H?3bEFeatureExtraction_ThreeEpochs/efficientnetb0/block7a_dwconv/depthwisehu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?3@?3H?3bMFeatureExtraction_ThreeEpochs/efficientnetb0/block5c_project_conv/Conv2D/Casthu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?2@?2H?2bEFeatureExtraction_ThreeEpochs/efficientnetb0/block4a_dwconv/depthwisehu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?2@?2H?2bFFeatureExtraction_ThreeEpochs/efficientnetb0/block3a_se_expand/Sigmoidhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?1@?1H?1bFFeatureExtraction_ThreeEpochs/efficientnetb0/block6a_se_expand/BiasAddhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?1@?1H?1bMFeatureExtraction_ThreeEpochs/efficientnetb0/block4b_project_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?1@?1H?1bJFeatureExtraction_ThreeEpochs/efficientnetb0/block6b_se_reduce/Conv2D/Casthu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?1@?1H?1bEFeatureExtraction_ThreeEpochs/efficientnetb0/block3b_dwconv/depthwisehu  ?B
?

?	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorReshapingOp<Eigen::IndexList<int, Eigen::type2index<1l> > const, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> > const> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorReshapingOp<Eigen::IndexList<int, Eigen::type2index<1l> > const, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> > const> const> const> const, Eigen::GpuDevice>, int)*?28?0@?0H?0bgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2J8?0@?0H?0bJFeatureExtraction_ThreeEpochs/efficientnetb0/block5b_se_reduce/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?0@?0H?0bJFeatureExtraction_ThreeEpochs/efficientnetb0/block6b_se_expand/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?0@?0H?0bJFeatureExtraction_ThreeEpochs/efficientnetb0/block6c_se_expand/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?0@?0H?0bJFeatureExtraction_ThreeEpochs/efficientnetb0/block7a_se_reduce/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_HALF_DT_FLOAT_kernel*?2?8?0@?0H?0bBgradient_tape/FeatureExtraction_ThreeEpochs/dense/MatMul/Cast/Casthu  ?B
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2 8?0@?0H?0bmul_2hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?0@?0H?0bFFeatureExtraction_ThreeEpochs/efficientnetb0/block6c_se_reduce/Sigmoidhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?2$8?0@?0H?0bFFeatureExtraction_ThreeEpochs/efficientnetb0/block6d_se_expand/BiasAddhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?0@?0H?0b;cond_1/then/_10/cond_1/Adam/Adam/update_1/ResourceApplyAdamhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2&8?0@?0H?0bJFeatureExtraction_ThreeEpochs/efficientnetb0/block4c_se_reduce/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?0@?0H?0bMFeatureExtraction_ThreeEpochs/efficientnetb0/block5a_project_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?0@?0H?0bJFeatureExtraction_ThreeEpochs/efficientnetb0/block6d_se_expand/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?0@?0H?0bJFeatureExtraction_ThreeEpochs/efficientnetb0/block6d_se_reduce/Conv2D/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?0@?0H?0bFFeatureExtraction_ThreeEpochs/efficientnetb0/block4b_se_reduce/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?0@?0H?0bFFeatureExtraction_ThreeEpochs/efficientnetb0/block6a_se_reduce/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?0@?0H?0bFFeatureExtraction_ThreeEpochs/efficientnetb0/block2a_se_expand/Sigmoidhu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?0@?0H?0XbEFeatureExtraction_ThreeEpochs/efficientnetb0/block5a_se_reduce/Conv2Dhu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?0@?0H?0XbEFeatureExtraction_ThreeEpochs/efficientnetb0/block6a_se_reduce/Conv2Dhu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?0@?0H?0Xb*FeatureExtraction_ThreeEpochs/dense/MatMulhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?0@?0H?0bFFeatureExtraction_ThreeEpochs/efficientnetb0/block4b_se_expand/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?0@?0H?0bFFeatureExtraction_ThreeEpochs/efficientnetb0/block4c_se_expand/BiasAddhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?0@?0H?0Xb=FeatureExtraction_ThreeEpochs/efficientnetb0/stem_conv/Conv2Dhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?/@?/H?/bFFeatureExtraction_ThreeEpochs/efficientnetb0/block7a_se_reduce/Sigmoidhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?/@?/H?/bJFeatureExtraction_ThreeEpochs/efficientnetb0/block6c_se_reduce/Conv2D/Casthu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?/@?/H?/bEFeatureExtraction_ThreeEpochs/efficientnetb0/block2a_dwconv/depthwisehu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?.@?.H?.bFFeatureExtraction_ThreeEpochs/efficientnetb0/block6d_se_reduce/BiasAddhu  ?B
?
?void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Max>(float const*, float*, int, int, cub::Max, std::iterator_traits<float const*>::value_type)**?28?.@?.H?.bgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2J8?.@?.H?.bJFeatureExtraction_ThreeEpochs/efficientnetb0/block5c_se_reduce/Conv2D/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?.@?.H?.bFFeatureExtraction_ThreeEpochs/efficientnetb0/block6d_se_reduce/Sigmoidhu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?.@?.H?.XbEFeatureExtraction_ThreeEpochs/efficientnetb0/block5b_se_reduce/Conv2Dhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2J8?.@?.H?.bJFeatureExtraction_ThreeEpochs/efficientnetb0/block6a_se_reduce/Conv2D/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?.@?.H?.bFFeatureExtraction_ThreeEpochs/efficientnetb0/block3a_se_reduce/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorGeneratorOp<tensorflow::generator::SparseXentGradGenerator<float, long>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorGeneratorOp<tensorflow::generator::SparseXentGradGenerator<float, long>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28?.@?.H?.bgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?-@?-H?-bLFeatureExtraction_ThreeEpochs/efficientnetb0/block2b_expand_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2q8?-@?-H?-bJFeatureExtraction_ThreeEpochs/efficientnetb0/block6b_dwconv/depthwise/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?-@?-H?-bLFeatureExtraction_ThreeEpochs/efficientnetb0/block5a_expand_conv/Conv2D/Casthu  ?B

Sqrt_GPU_DT_HALF_DT_HALF_kernel*?28?-@?-H?-b?FeatureExtraction_ThreeEpochs/efficientnetb0/normalization/Sqrthu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?-@?-H?-bFFeatureExtraction_ThreeEpochs/efficientnetb0/block4b_se_reduce/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?2$8?-@?-H?-bFFeatureExtraction_ThreeEpochs/efficientnetb0/block6b_se_expand/BiasAddhu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?-@?-H?-XbEFeatureExtraction_ThreeEpochs/efficientnetb0/block4c_se_reduce/Conv2Dhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?-@?-H?-bFFeatureExtraction_ThreeEpochs/efficientnetb0/block5b_se_expand/BiasAddhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?,@?,H?,bFFeatureExtraction_ThreeEpochs/efficientnetb0/block6b_se_reduce/Sigmoidhu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?,@?,H?,XbEFeatureExtraction_ThreeEpochs/efficientnetb0/block4b_se_reduce/Conv2Dhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?2$8?,@?,H?,bFFeatureExtraction_ThreeEpochs/efficientnetb0/block7a_se_expand/BiasAddhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?,@?,H?,bBFeatureExtraction_ThreeEpochs/efficientnetb0/block6b_se_reduce/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?,@?,H?,bFFeatureExtraction_ThreeEpochs/efficientnetb0/block4c_se_reduce/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?,@?,H?,bAssignAddVariableOp_2hu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?,@?,H?,bFFeatureExtraction_ThreeEpochs/efficientnetb0/block5c_se_reduce/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?,@?,H?,bFFeatureExtraction_ThreeEpochs/efficientnetb0/block6b_se_reduce/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?,@?,H?,bFFeatureExtraction_ThreeEpochs/efficientnetb0/block5c_se_expand/BiasAddhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?+@?+H?+bFFeatureExtraction_ThreeEpochs/efficientnetb0/block3b_se_reduce/Sigmoidhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?+@?+H?+bLFeatureExtraction_ThreeEpochs/efficientnetb0/block4c_expand_conv/Conv2D/Casthu  ?B
b
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?+@?+H?+b!cond_1/then/_10/cond_1/Adam/Pow_1hu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?+@?+H?+bFFeatureExtraction_ThreeEpochs/efficientnetb0/block5b_se_reduce/BiasAddhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2K8?+@?+H?+bMFeatureExtraction_ThreeEpochs/efficientnetb0/block4a_project_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2q8?+@?+H?+bJFeatureExtraction_ThreeEpochs/efficientnetb0/block6c_dwconv/depthwise/Casthu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?+@?+H?+bBFeatureExtraction_ThreeEpochs/efficientnetb0/block5c_se_reduce/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?+@?+H?+bdiv_no_nan_1hu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?+@?+H?+bFFeatureExtraction_ThreeEpochs/efficientnetb0/block1a_se_reduce/BiasAddhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?*@?*H?*bKFeatureExtraction_ThreeEpochs/efficientnetb0/block6b_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?*@?*H?*bLFeatureExtraction_ThreeEpochs/efficientnetb0/block3a_expand_conv/Conv2D/Casthu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?*@?*H?*b+FeatureExtraction_ThreeEpochs/dense/BiasAddhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?*@?*H?*bBFeatureExtraction_ThreeEpochs/efficientnetb0/block5b_se_reduce/mulhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?*@?*H?*bFFeatureExtraction_ThreeEpochs/efficientnetb0/block5a_se_reduce/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?*@?*H?*bFFeatureExtraction_ThreeEpochs/efficientnetb0/block3a_se_expand/BiasAddhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?*@?*H?*bJFeatureExtraction_ThreeEpochs/efficientnetb0/block2a_se_expand/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?*@?*H?*bLFeatureExtraction_ThreeEpochs/efficientnetb0/block4b_expand_conv/Conv2D/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?*@?*H?*bLgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nanhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?)@?)H?)bFFeatureExtraction_ThreeEpochs/efficientnetb0/block1a_se_reduce/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?)@?)H?)bFFeatureExtraction_ThreeEpochs/efficientnetb0/block4a_se_reduce/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?)@?)H?)b3sparse_categorical_crossentropy/weighted_loss/valuehu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?)@?)H?)XbEFeatureExtraction_ThreeEpochs/efficientnetb0/block6c_se_reduce/Conv2Dhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?)@?)H?)bFFeatureExtraction_ThreeEpochs/efficientnetb0/block2b_se_reduce/BiasAddhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?)@?)H?)bKFeatureExtraction_ThreeEpochs/efficientnetb0/block3b_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2
8?)@?)H?)bJFeatureExtraction_ThreeEpochs/efficientnetb0/block4a_se_reduce/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2J8?)@?)H?)bJFeatureExtraction_ThreeEpochs/efficientnetb0/block6a_se_expand/Conv2D/Casthu  ?B
X
"AddV2_GPU_DT_INT64_DT_INT64_kernel*?28?)@?)H?)bcond/then/_0/cond/addhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?)@?)H?)bBFeatureExtraction_ThreeEpochs/efficientnetb0/block2b_se_reduce/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?)@?)H?)b=cond/then/_0/cond/cond/else/_75/cond/cond/AssignAddVariableOphu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?)@?)H?)XbEFeatureExtraction_ThreeEpochs/efficientnetb0/block6d_se_reduce/Conv2Dhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2q8?(@?(H?(bJFeatureExtraction_ThreeEpochs/efficientnetb0/block6d_dwconv/depthwise/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?(@?(H?(bJFeatureExtraction_ThreeEpochs/efficientnetb0/block2b_se_expand/Conv2D/Casthu  ?B
b
"AddV2_GPU_DT_INT64_DT_INT64_kernel*?28?(@?(H?(bcond_1/then/_10/cond_1/Adam/addhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2&8?(@?(H?(bJFeatureExtraction_ThreeEpochs/efficientnetb0/block4b_se_reduce/Conv2D/Casthu  ?B
?
"Maximum_GPU_DT_HALF_DT_HALF_kernel*?28?(@?(H?(bBFeatureExtraction_ThreeEpochs/efficientnetb0/normalization/Maximumhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?(@?(H?(bBFeatureExtraction_ThreeEpochs/efficientnetb0/block7a_se_reduce/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?(@?(H?(bFFeatureExtraction_ThreeEpochs/efficientnetb0/block1a_se_expand/Sigmoidhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?(@?(H?(bFFeatureExtraction_ThreeEpochs/efficientnetb0/block6a_se_reduce/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?(@?(H?(bFFeatureExtraction_ThreeEpochs/efficientnetb0/block2b_se_expand/BiasAddhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?28?(@?(H?(b1sparse_categorical_crossentropy/weighted_loss/Sumhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2B8?(@?(H?(bJFeatureExtraction_ThreeEpochs/efficientnetb0/block5c_dwconv/depthwise/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?(@?(H?(bKFeatureExtraction_ThreeEpochs/efficientnetb0/block2b_se_reduce/BiasAdd/Casthu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?(@?(H?(bUgradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mulhu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?(@?(H?(XbEFeatureExtraction_ThreeEpochs/efficientnetb0/block7a_se_reduce/Conv2Dhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?(@?(H?(bFFeatureExtraction_ThreeEpochs/efficientnetb0/block7a_se_reduce/BiasAddhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *?28?(@?(H?(bAll_1hu  ?B
`
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?'@?'H?'bcond_1/then/_10/cond_1/Adam/Powhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?'@?'H?'bFFeatureExtraction_ThreeEpochs/efficientnetb0/block5a_se_expand/BiasAddhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?'@?'H?'bJFeatureExtraction_ThreeEpochs/efficientnetb0/block7a_se_expand/Conv2D/Casthu  ?B
M
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*?28?'@?'H?'bIsFinitehu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?'@?'H?'bFFeatureExtraction_ThreeEpochs/efficientnetb0/block1a_se_expand/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?'@?'H?'bFFeatureExtraction_ThreeEpochs/efficientnetb0/block6c_se_reduce/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?2$8?'@?'H?'bFFeatureExtraction_ThreeEpochs/efficientnetb0/block6c_se_expand/BiasAddhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28?&@?&H?&bgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?&@?&H?&bFFeatureExtraction_ThreeEpochs/efficientnetb0/block3b_se_reduce/BiasAddhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?&@?&H?&bFFeatureExtraction_ThreeEpochs/efficientnetb0/block5a_se_reduce/Sigmoidhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?&@?&H?&bFFeatureExtraction_ThreeEpochs/efficientnetb0/block3a_se_reduce/BiasAddhu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?&@?&H?&XbEFeatureExtraction_ThreeEpochs/efficientnetb0/block5c_se_reduce/Conv2Dhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?&@?&H?&bEFeatureExtraction_ThreeEpochs/efficientnetb0/block1a_dwconv/depthwisehu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?%@?%H?%bFFeatureExtraction_ThreeEpochs/efficientnetb0/block2a_se_reduce/BiasAddhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?%@?%H?%bMFeatureExtraction_ThreeEpochs/efficientnetb0/block4c_project_conv/Conv2D/Casthu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?%@?%H?%bFFeatureExtraction_ThreeEpochs/efficientnetb0/block4c_se_reduce/BiasAddhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2&8?%@?%H?%bLFeatureExtraction_ThreeEpochs/efficientnetb0/block3b_expand_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2&8?%@?%H?%bJFeatureExtraction_ThreeEpochs/efficientnetb0/block4c_se_expand/Conv2D/Casthu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?%@?%H?%XbEFeatureExtraction_ThreeEpochs/efficientnetb0/block6b_se_reduce/Conv2Dhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2B8?$@?$H?$bJFeatureExtraction_ThreeEpochs/efficientnetb0/block6a_dwconv/depthwise/Casthu  ?B
O
'Reciprocal_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?$@?$H?$btruedivhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?$@?$H?$bFFeatureExtraction_ThreeEpochs/efficientnetb0/block2a_se_expand/BiasAddhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?$@?$H?$bLFeatureExtraction_ThreeEpochs/efficientnetb0/block2a_expand_conv/Conv2D/Casthu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?$@?$H?$bBFeatureExtraction_ThreeEpochs/efficientnetb0/block4c_se_reduce/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?$@?$H?$bBFeatureExtraction_ThreeEpochs/efficientnetb0/block6d_se_reduce/mulhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *?28?$@?$H?$bAll_2hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2&8?$@?$H?$bJFeatureExtraction_ThreeEpochs/efficientnetb0/block5a_se_reduce/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?$@?$H?$bKFeatureExtraction_ThreeEpochs/efficientnetb0/block5b_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2B8?$@?$H?$bJFeatureExtraction_ThreeEpochs/efficientnetb0/block5b_dwconv/depthwise/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?#@?#H?#bAssignAddVariableOp_1hu  ?B
?
?void tensorflow::functor::CleanupSegments<bool*, bool*, tensorflow::functor::And>(bool*, bool*, int, int, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type)* 28?#@?#H?#bAllhu  HB
s
 Cast_GPU_DT_HALF_DT_FLOAT_kernel*?28?#@?#H?#b2FeatureExtraction_ThreeEpochs/softmax_float32/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2&8?#@?#H?#bLFeatureExtraction_ThreeEpochs/efficientnetb0/block4a_expand_conv/Conv2D/Casthu  ?B
g
(GreaterEqual_GPU_DT_INT64_DT_BOOL_kernel*?28?#@?#H?#bcond/then/_0/cond/GreaterEqualhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?#@?#H?#bAssignAddVariableOp_3hu  ?B
?
?void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Max>(float const*, float*, int, int, cub::Max, std::iterator_traits<float const*>::value_type)**?28?#@?#H?#b5FeatureExtraction_ThreeEpochs/softmax_float32/Softmaxhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2
8?#@?#H?#bJFeatureExtraction_ThreeEpochs/efficientnetb0/block3b_se_reduce/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2J8?#@?#H?#bJFeatureExtraction_ThreeEpochs/efficientnetb0/block5c_se_expand/Conv2D/Casthu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?#@?#H?#bBFeatureExtraction_ThreeEpochs/efficientnetb0/block2a_se_reduce/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?#@?#H?#bBFeatureExtraction_ThreeEpochs/efficientnetb0/block3b_se_reduce/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?#@?#H?#bBFeatureExtraction_ThreeEpochs/efficientnetb0/block4a_se_reduce/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?#@?#H?#bAssignAddVariableOp_4hu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?#@?#H?#bFFeatureExtraction_ThreeEpochs/efficientnetb0/block4a_se_reduce/BiasAddhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?28?#@?#H?#bSum_2hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2	8?"@?"H?"bMFeatureExtraction_ThreeEpochs/efficientnetb0/block2a_project_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2	8?"@?"H?"bJFeatureExtraction_ThreeEpochs/efficientnetb0/block4a_dwconv/depthwise/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?"@?"H?"bMFeatureExtraction_ThreeEpochs/efficientnetb0/block2b_project_conv/Conv2D/Casthu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?"@?"H?"bBFeatureExtraction_ThreeEpochs/efficientnetb0/block1a_se_reduce/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?"@?"H?"bBFeatureExtraction_ThreeEpochs/efficientnetb0/block3a_se_reduce/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28?"@?"H?"bBgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1hu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?"@?"H?"bFFeatureExtraction_ThreeEpochs/efficientnetb0/block4a_se_expand/BiasAddhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?"@?"H?"bKFeatureExtraction_ThreeEpochs/efficientnetb0/block1a_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?"@?"H?"bMFeatureExtraction_ThreeEpochs/efficientnetb0/block1a_project_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?"@?"H?"bJFeatureExtraction_ThreeEpochs/efficientnetb0/block2a_se_reduce/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?"@?"H?"bJFeatureExtraction_ThreeEpochs/efficientnetb0/block2b_se_reduce/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?"@?"H?"bKFeatureExtraction_ThreeEpochs/efficientnetb0/block6c_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2&8?"@?"H?"bMFeatureExtraction_ThreeEpochs/efficientnetb0/block3b_project_conv/Conv2D/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?"@?"H?"b4cond_1/then/_10/cond_1/Adam/Adam/AssignAddVariableOphu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?"@?"H?"bFFeatureExtraction_ThreeEpochs/efficientnetb0/block3b_se_expand/BiasAddhu  ?B
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?"@?"H?"bmul_3hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?"@?"H?"bJFeatureExtraction_ThreeEpochs/efficientnetb0/block1a_se_reduce/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?"@?"H?"bKFeatureExtraction_ThreeEpochs/efficientnetb0/block5b_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?"@?"H?"bJFeatureExtraction_ThreeEpochs/efficientnetb0/block1a_dwconv/depthwise/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?"@?"H?"bJFeatureExtraction_ThreeEpochs/efficientnetb0/block3a_se_reduce/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?"@?"H?"bKFeatureExtraction_ThreeEpochs/efficientnetb0/block6d_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2
8?"@?"H?"bJFeatureExtraction_ThreeEpochs/efficientnetb0/block3b_se_expand/Conv2D/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?"@?"H?"bFFeatureExtraction_ThreeEpochs/efficientnetb0/block2b_se_reduce/Sigmoidhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?"@?"H?"bJFeatureExtraction_ThreeEpochs/efficientnetb0/block1a_se_expand/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?"@?"H?"bKFeatureExtraction_ThreeEpochs/efficientnetb0/block2a_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?"@?"H?"bKFeatureExtraction_ThreeEpochs/efficientnetb0/block3a_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?"@?"H?"bKFeatureExtraction_ThreeEpochs/efficientnetb0/block4c_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?"@?"H?"bKFeatureExtraction_ThreeEpochs/efficientnetb0/block6d_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?"@?"H?"bJFeatureExtraction_ThreeEpochs/efficientnetb0/block3b_dwconv/depthwise/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?"@?"H?"bFFeatureExtraction_ThreeEpochs/efficientnetb0/block5b_se_reduce/Sigmoidhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?!@?!H?!bKFeatureExtraction_ThreeEpochs/efficientnetb0/block6a_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?!@?!H?!bKFeatureExtraction_ThreeEpochs/efficientnetb0/block6c_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?!@?!H?!bKFeatureExtraction_ThreeEpochs/efficientnetb0/block7a_se_reduce/BiasAdd/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?!@?!H?!bAssignAddVariableOphu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?!@?!H?!b?gradient_tape/FeatureExtraction_ThreeEpochs/softmax_float32/Cast/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?!@?!H?!bJFeatureExtraction_ThreeEpochs/efficientnetb0/block4c_dwconv/depthwise/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2&8?!@?!H?!bJFeatureExtraction_ThreeEpochs/efficientnetb0/block4b_se_expand/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2&8?!@?!H?!bJFeatureExtraction_ThreeEpochs/efficientnetb0/block5a_se_expand/Conv2D/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?!@?!H?!b
div_no_nanhu  ?B
?
!Cast_GPU_DT_FLOAT_DT_INT64_kernel*?28?!@?!H?!bbsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?!@?!H?!bBFeatureExtraction_ThreeEpochs/efficientnetb0/block4b_se_reduce/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28? @? H? bFFeatureExtraction_ThreeEpochs/efficientnetb0/block5c_se_reduce/Sigmoidhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bKFeatureExtraction_ThreeEpochs/efficientnetb0/block2b_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bKFeatureExtraction_ThreeEpochs/efficientnetb0/block5a_se_reduce/BiasAdd/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28? @? H? bFFeatureExtraction_ThreeEpochs/efficientnetb0/block2a_se_reduce/Sigmoidhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bKFeatureExtraction_ThreeEpochs/efficientnetb0/block5c_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bJFeatureExtraction_ThreeEpochs/efficientnetb0/block3a_dwconv/depthwise/Casthu  ?B
P
%LogicalAnd_GPU_DT_BOOL_DT_BOOL_kernel*?28? @? H? b
LogicalAndhu  ?B
D
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28? @? H? bMulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28? @? H? bBFeatureExtraction_ThreeEpochs/efficientnetb0/block6a_se_reduce/mulhu  ?B
G
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*?28? @? H? bCast_3hu  ?B
q
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? b0FeatureExtraction_ThreeEpochs/dense/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bKFeatureExtraction_ThreeEpochs/efficientnetb0/block1a_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bKFeatureExtraction_ThreeEpochs/efficientnetb0/block2a_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bKFeatureExtraction_ThreeEpochs/efficientnetb0/block3a_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bKFeatureExtraction_ThreeEpochs/efficientnetb0/block3b_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bKFeatureExtraction_ThreeEpochs/efficientnetb0/block4a_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bKFeatureExtraction_ThreeEpochs/efficientnetb0/block4a_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bKFeatureExtraction_ThreeEpochs/efficientnetb0/block4b_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bKFeatureExtraction_ThreeEpochs/efficientnetb0/block5c_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bKFeatureExtraction_ThreeEpochs/efficientnetb0/block6b_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bKFeatureExtraction_ThreeEpochs/efficientnetb0/block4b_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bKFeatureExtraction_ThreeEpochs/efficientnetb0/block4c_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bKFeatureExtraction_ThreeEpochs/efficientnetb0/block5a_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bKFeatureExtraction_ThreeEpochs/efficientnetb0/block6a_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bJFeatureExtraction_ThreeEpochs/efficientnetb0/block2a_dwconv/depthwise/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bJFeatureExtraction_ThreeEpochs/efficientnetb0/block3a_se_expand/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bBFeatureExtraction_ThreeEpochs/efficientnetb0/stem_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bKFeatureExtraction_ThreeEpochs/efficientnetb0/block7a_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bJFeatureExtraction_ThreeEpochs/efficientnetb0/block2b_dwconv/depthwise/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2
8? @? H? bJFeatureExtraction_ThreeEpochs/efficientnetb0/block4a_se_expand/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bJFeatureExtraction_ThreeEpochs/efficientnetb0/block4b_dwconv/depthwise/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? bMFeatureExtraction_ThreeEpochs/efficientnetb0/block3a_project_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2)8? @? H? bJFeatureExtraction_ThreeEpochs/efficientnetb0/block7a_dwconv/depthwise/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2/8? @? H? bJFeatureExtraction_ThreeEpochs/efficientnetb0/block5a_dwconv/depthwise/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2J8? @? H? bJFeatureExtraction_ThreeEpochs/efficientnetb0/block5b_se_expand/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_HALF_DT_FLOAT_kernel*?28? @? H? bCgradient_tape/FeatureExtraction_ThreeEpochs/dense/BiasAdd/Cast/Casthu  ?B
?
 Cast_GPU_DT_HALF_DT_FLOAT_kernel*?28? @? H? bnFeatureExtraction_ThreeEpochs/softmax_float32/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Casthu  ?B
F
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28? @? H? bCasthu  ?B
H
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28? @? H? bCast_4hu  ?B
?
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28? @? H? b?sparse_categorical_crossentropy/weighted_loss/num_elements/Casthu  ?B
H
!Cast_GPU_DT_INT64_DT_FLOAT_kernel*?28? @? H? bCast_2hu  ?B
d
!Cast_GPU_DT_INT64_DT_FLOAT_kernel*?28? @? H? b"cond_1/then/_10/cond_1/Adam/Cast_1hu  ?B
?
!Cast_GPU_DT_INT64_DT_FLOAT_kernel*?28? @? H? b`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Casthu  ?B
G
!Equal_GPU_DT_FLOAT_DT_BOOL_kernel*?28? @? H? bEqualhu  ?B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*?28? @? H? b
IsFinite_1hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28? @? H? bBFeatureExtraction_ThreeEpochs/efficientnetb0/block5a_se_reduce/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28? @? H? bBFeatureExtraction_ThreeEpochs/efficientnetb0/block6c_se_reduce/mulhu  ?B