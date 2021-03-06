
?
?void convolve_common_engine_float_NHWC<__half, __half, 128, 6, 7, 3, 3, 5, true, false, false, false, false>(int, int, int, __half const*, __half const*, int, __half*, conv_kernel_common_params, unsigned long long, unsigned long, float, float, int, __half const*, __half const*, bool)C?2* 28???@??H??bXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_dwconv/depthwiseh?u  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 
8??B@??BH??BbbModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_expand_bn/FusedBatchNormV3hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?n8ί?@ί?Hί?bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2b_se_excite/mulhu  ?B
?
?void convolve_common_engine_float_NHWC<__half, __half, 128, 5, 5, 3, 3, 3, true, false, false, false, false>(int, int, int, __half const*, __half const*, int, __half*, conv_kernel_common_params, unsigned long long, unsigned long, float, float, int, __half const*, __half const*, bool)T?*2?b8??:@??:H??:XbPModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/stem_conv/Conv2Dhu  zB
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?b8??8@??8H??8bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block1a_se_excite/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?I8??5@??5H??5b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_expand_activation/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorGeneratorOp<tensorflow::generator::ProjectiveGenerator<Eigen::GpuDevice, Eigen::half, (tensorflow::generator::Mode)0>, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorGeneratorOp<tensorflow::generator::ProjectiveGenerator<Eigen::GpuDevice, Eigen::half, (tensorflow::generator::Mode)0>, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)(*?2(8??.@??.H??.bxModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_rotation/transform/ImageProjectiveTransformV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorGeneratorOp<tensorflow::generator::ProjectiveGenerator<Eigen::GpuDevice, Eigen::half, (tensorflow::generator::Mode)0>, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorGeneratorOp<tensorflow::generator::ProjectiveGenerator<Eigen::GpuDevice, Eigen::half, (tensorflow::generator::Mode)0>, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)(*?2(8??-@??-H??-btModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_zoom/transform/ImageProjectiveTransformV3hu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?28??,@??,H??,bWModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block1a_se_squeeze/Meanhu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_128x128_nn_align8>(cutlass_75_tensorop_f16_s1688gemm_f16_128x128_nn_align8::Params)? ??*?2?8??+@??+H??+XbZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_expand_conv/Conv2Dhu  ?A
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 5, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)P*?2?-8??*@??*H??*bXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3b_dwconv/depthwisehu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?I8??*@??*H??*bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_se_excite/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??)@??)H??)baModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_expand_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)(*?2(8??)@??)H??)bVModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_dwconv_pad/Padhu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 3, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)0*?2?n8??$@??$H??$bXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2b_dwconv/depthwisehu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 5, 2, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)d*?2?8??#@??#H??#bXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3a_dwconv/depthwisehu  HB
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 5, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)P*?2? 8؞@؞H؞bXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5b_dwconv/depthwisehu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 5, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)P*?2? 8??@??H??bXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5c_dwconv/depthwisehu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 3, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)0*?2?b8??@??H??bXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block1a_dwconv/depthwisehu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 3, 2, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct):*?2?I8??@??H??bXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_dwconv/depthwisehu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?-8??@??H??bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3b_se_excite/mulhu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8::Params)` ??*?2?18??@??H??Xb[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block1a_project_conv/Conv2Dhu  HB
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?	*?2 8??@??H??b[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2b_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?	*?2 8??@??H??bbModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2b_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?	*?2 8??@??H??bbModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3a_expand_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)(*?2(8??@??H??bQModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/stem_conv_pad/Padhu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 5, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)P*?2?8ڮ@ڮHڮbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5a_dwconv/depthwisehu  ?B
?
?void tensorflow::(anonymous namespace)::ResizeBilinearKernel<Eigen::half>(int, Eigen::half const*, float, float, int, int, int, int, int, int, float*)*?2(8??@??H??bbModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/resizing/resize/ResizeBilinearhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/stem_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??b[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block1a_bn/FusedBatchNormV3hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bVModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2b_activation/mulhu  ?B
?
Div_GPU_DT_HALF_DT_HALF_kernel*?2?$8??@??H??bWModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/normalization_1/truedivhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2b_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3a_expand_activation/mulhu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8::Params)` ??*?2?18??@??H??XbZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3a_expand_conv/Conv2Dhu  HB
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 5, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)P*?2?8??@??H??bXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6d_dwconv/depthwisehu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8::Params)` ??*?2?18??@??H??XbZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2b_expand_conv/Conv2Dhu  HB
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 5, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)P*?2?8??@??H??bXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6c_dwconv/depthwisehu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 5, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)P*?2?8ڍ@ڍHڍbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6b_dwconv/depthwisehu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorReverseOp<Eigen::array<bool, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorReverseOp<Eigen::array<bool, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?2(8??@??H??bzModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_flip/stateless_random_flip_left_right/ReverseV2hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2? 8ۼ@ۼHۼbUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5b_se_excite/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2? 8??@??H??bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5c_se_excite/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bVModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block1a_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bSModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/stem_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?$8??@??H??bvModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_flip/stateless_random_flip_left_right/mul_1hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)(*?2(8??@??H??bVModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3a_dwconv_pad/Padhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?$8??@??H??btModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_flip/stateless_random_flip_left_right/mulhu  ?B
?
Sub_GPU_DT_HALF_DT_HALF_kernel*?2?$8??@??H??bSModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/normalization_1/subhu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x128_nn_align8>(cutlass_75_tensorop_f16_s1688gemm_f16_64x128_nn_align8::Params)? ??*?2?8??@??H??Xb[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2b_project_conv/Conv2Dhu  ?A
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 
8ۨ@ۨHۨb[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??baModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2b_expand_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2b_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??baModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3a_expand_activation/Sigmoidhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3a_se_excite/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8ܻ@ܻHܻbWModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/stem_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block1a_activation/Sigmoidhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bVModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4b_se_excite/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5a_se_excite/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4c_se_excite/mulhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2$8??@??H??bWModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2b_se_squeeze/Meanhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?28??@??H??bWModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_se_squeeze/Meanhu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8::Params)` ??*?2?8??@??H??Xb[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_project_conv/Conv2Dhu  HB
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??
@??
H??
bZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_activation/Sigmoidhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??
@??
H??
bcModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block1a_project_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??
@??
H??
b[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3b_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??
@??
H??
bbModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3b_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??
@??
H??
bbModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4a_expand_bn/FusedBatchNormV3hu  ?B
?
 Cast_GPU_DT_HALF_DT_FLOAT_kernel*?2??8??
@??
H??
bDModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/Casthu  ?B
?
 Cast_GPU_DT_HALF_DT_FLOAT_kernel*?2??8ކ
@ކ
Hކ
bTModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/normalization_1/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2??8??	@??	H??	bVModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/normalization_1/Cast_1hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2??8??	@??	H??	bPModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/rescaling_1/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2??8??	@??	H??	bTModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_flip/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2??8??	@??	H??	bQModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/resizing/Casthu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bVModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3b_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4a_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3b_expand_activation/mulhu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 3, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)0*?2?8??@??H??bXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4c_dwconv/depthwisehu  ?B
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 3, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)0*?2?8??@??H??bXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4b_dwconv/depthwisehu  ?B
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?218??@??H??PXbZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_expand_conv/Conv2Dhu  ?A
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?218??@??H??PXbZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5b_expand_conv/Conv2Dhu  ?A
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_se_excite/mulhu  ?B
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?218??@??H??PXbZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5c_expand_conv/Conv2Dhu  ?A
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?**?2 	8??@??H??bbModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?**?2 	8??@??H??b[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5c_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?**?2 	8??@??H??bbModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5c_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?**?2 	8??@??H??bbModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5b_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?**?2 	8??@??H??b[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5b_bn/FusedBatchNormV3hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6b_se_excite/mulhu  ?B
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?2?8??@??H??PXbZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3b_expand_conv/Conv2Dhu  ?A
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8޿@޿H޿bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6d_se_excite/mulhu  ?B
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?2?8??@??H??PXbZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4a_expand_conv/Conv2Dhu  ?A
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6c_se_excite/mulhu  ?B
?
:turing_fp16_s1688gemm_fp16_128x256_ldg8_f2f_stages_32x1_nn??? ??*?2
8??@??H??PXbOModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/top_conv/Conv2Dh
?
7turing_fp16_s1688gemm_fp16_64x128_sliced1x2_ldg8_f2f_nn???*?2?8??@??H??PXb[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3b_project_conv/Conv2Dhu  ?A
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3b_activation/Sigmoidhu  ?B
?
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2?	8??@??H??btModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_flip/stateless_random_flip_left_right/addhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??baModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4a_expand_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)(*?2(8??@??H??bVModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4a_dwconv_pad/Padhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??baModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3b_expand_activation/Sigmoidhu  ?B
?
.turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_nn???*?218??@??H??PXb[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5c_project_conv/Conv2Dhu  ?A
?
.turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_nn???*?218??@??H??PXb[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5b_project_conv/Conv2Dhu  ?A
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nt???*?2	8??@??H??PXb~gradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_project_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 3, 2, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct):*?2?8??@??H??bXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4a_dwconv/depthwisehu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorPaddingOp<Eigen::array<Eigen::IndexPair<int>, 4ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)(*?2(8??@??H??bVModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_dwconv_pad/Padhu  ?B
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nt???*?2
8??@??H??PXbrgradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/top_conv/Conv2D/Conv2DBackpropFilterhu  ?A
?
?void conv2d_c1_k1_nhwc_kernel<__half, __half, __half, float, float, 3, 1, true, false>(float, cudnnTensorStruct, __half const*, cudnnFilterStruct, __half const*, cudnnConvolutionStruct, float, cudnnTensorStruct, __half*, cudnn::reduced_divisor, cudnn::reduced_divisor, cudnn::reduced_divisor, int, __half const*, float const*, cudnnActivationStruct)0*?2?8??@??H??bXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_dwconv/depthwisehu  ?B
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_tn???*?28??@??H??PXbqgradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/top_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8ޢ@ޢHޢb]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5c_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bVModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5c_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bVModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5b_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8ޟ@ޟHޟb]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5b_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8ߛ@ߛHߛbUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4a_se_excite/mulhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?	*?2 8??@??H??b[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3a_bn/FusedBatchNormV3hu  ?B
?
Div_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bagradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/global_average_pooling2d_1/truedivhu  ?B
?
.turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_tn???*?2	8??@??H??PXb}gradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_project_conv/Conv2D/Conv2DBackpropInputhu  ?A
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bVModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3a_activation/mulhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8޹@޹H޹bbModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4b_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??b[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5a_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??b[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4c_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bbModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4c_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??b[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4b_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bbModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5a_expand_bn/FusedBatchNormV3hu  ?B
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?28߮@߮H߮PXb[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_project_conv/Conv2Dhu  ?A
?
.turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_nn???*?218??@??H??PXb[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5a_project_conv/Conv2Dhu  ?A
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align8::Params)` ??*?2?8??@??H??Xb[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3a_project_conv/Conv2Dhu  HB
?
?void Eigen::internal::OuterReductionKernel<16, Eigen::TensorReductionEvaluatorBase<Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<0l>> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<long, Eigen::type2index<1l> > const, Eigen::TensorReshapingOp<Eigen::IndexList<Eigen::type2index<1l>, long> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, Eigen::internal::SumReducer<float>, long>(Eigen::internal::SumReducer<float>, Eigen::TensorReductionEvaluatorBase<Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<0l>> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<long, Eigen::type2index<1l> > const, Eigen::TensorReshapingOp<Eigen::IndexList<Eigen::type2index<1l>, long> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long, long, Eigen::TensorReductionEvaluatorBase<Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<0l>> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<long, Eigen::type2index<1l> > const, Eigen::TensorReshapingOp<Eigen::IndexList<Eigen::type2index<1l>, long> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>::CoeffReturnType*)<*?2?8??@??H??bigradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/top_bn/FusedBatchNormGradV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??baModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5c_expand_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5c_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??baModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5b_expand_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5b_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??baModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_expand_activation/Sigmoidhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?	8??@??H??bOModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/rescaling_1/mulhu  ?B
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?218??@??H??PXb[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4c_project_conv/Conv2Dhu  ?A
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<long, Eigen::type2index<1l> > const, Eigen::TensorReshapingOp<Eigen::IndexList<Eigen::type2index<1l>, long> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<long, Eigen::type2index<1l> > const, Eigen::TensorReshapingOp<Eigen::IndexList<Eigen::type2index<1l>, long> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?2(8??@??H??bigradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/top_bn/FusedBatchNormGradV3hu  ?B
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?218??@??H??PXb[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4b_project_conv/Conv2Dhu  ?A
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?218??@??H??PXbZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4c_expand_conv/Conv2Dhu  ?A
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?218??@??H??PXbZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5a_expand_conv/Conv2Dhu  ?A
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?218??@??H??PXbZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4b_expand_conv/Conv2Dhu  ?A
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2<8??@??H??bWModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3b_se_squeeze/Meanhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5a_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4c_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bVModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5a_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bVModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4c_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bVModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4b_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4b_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_se_excite/mulhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?P*?2 8??@??H??bWModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/top_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 
8ߟ@ߟHߟbcModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2b_project_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 
8ޙ@ޙHޙbcModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_project_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3a_activation/Sigmoidhu  ?B
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?2	8??@??H??PXbZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6d_expand_conv/Conv2Dhu  ?A
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?2	8??@??H??PXbZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6c_expand_conv/Conv2Dhu  ?A
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?2	8??@??H??PXbZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_expand_conv/Conv2Dhu  ?A
?
:turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn???*?2	8ߎ@ߎHߎPXbZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6b_expand_conv/Conv2Dhu  ?A
?
7turing_fp16_s1688gemm_fp16_64x128_sliced1x2_ldg8_f2f_nn???*?28??@??H??PXb[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6b_project_conv/Conv2Dhu  ?A
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??baModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5a_expand_activation/Sigmoidhu  ?B
?
7turing_fp16_s1688gemm_fp16_64x128_sliced1x2_ldg8_f2f_nn???*?28??@??H??PXb[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6c_project_conv/Conv2Dhu  ?A
?
7turing_fp16_s1688gemm_fp16_64x128_sliced1x2_ldg8_f2f_nn???*?28??@??H??PXb[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6d_project_conv/Conv2Dhu  ?A
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2$8??@??H??bWModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3a_se_squeeze/Meanhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?H*?2 8??@??H??bbModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6c_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?H*?2 8??@??H??bbModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6d_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?H*?2 8??@??H??bbModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_expand_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?H*?2 8??@??H??bbModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6b_expand_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??baModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4c_expand_activation/Sigmoidhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?H*?2 8??@??H??b[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4b_activation/Sigmoidhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?H*?2 8??@??H??b[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6c_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4c_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??baModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4b_expand_activation/Sigmoidhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?H*?2 8??@??H??b[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6d_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?H*?2 8??@??H??b[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6b_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5a_activation/Sigmoidhu  ?B
?
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bOModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2b_add/addhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?2(8??@??H??b;cond_1/then/_10/cond_1/Adam/Adam/update_7/ResourceApplyAdamhu  ?B
U
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bAdam/gradients/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8ߛ@ߛHߛbRModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/top_activation/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorGeneratorOp<tensorflow::generator::SparseXentLossGenerator<float, long>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorGeneratorOp<tensorflow::generator::SparseXentLossGenerator<float, long>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)>*?28ߗ@ߗHߗbgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshu  ?B
?
.turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_nn???*?218??@??H??PXb[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4a_project_conv/Conv2Dhu  ?A
W
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bAdam/gradients/mul_1hu  ?B
W
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bAdam/gradients/mul_2hu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2?8??@??H??bWModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5b_se_squeeze/Meanhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?2(8??@??H??b;cond_1/then/_10/cond_1/Adam/Adam/update_4/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2?8??@??H??bWModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5c_se_squeeze/Meanhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bigradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_se_excite/mul/Mul_1hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6b_expand_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bVModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_activation/mulhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??b[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4a_bn/FusedBatchNormV3hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bVModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6d_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bVModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6b_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bVModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6c_activation/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6c_expand_activation/mulhu  ?B
?
7turing_fp16_s1688gemm_fp16_64x128_sliced1x2_ldg8_f2f_nn???*?28??@??H??PXb[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_project_conv/Conv2Dhu  ?A
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6d_expand_activation/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bAdam/gradients/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bVModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/top_activation/Sigmoidhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bVModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4a_activation/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)**?2(8??@??H??begradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/global_average_pooling2d_1/BroadcastTohu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??baModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_expand_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_activation/Sigmoidhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?**?2 8ߣ@ߣHߣb[Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??baModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6c_expand_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??baModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6b_expand_activation/Sigmoidhu  ?B
U
Sub_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bAdam/gradients/subhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8ߟ@ߟHߟbZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6b_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6d_activation/Sigmoidhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2x8??@??H??bWModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4c_se_squeeze/Meanhu  ?B
W
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2?8ߙ@ߙHߙbAdam/gradients/addhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??baModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6d_expand_activation/Sigmoidhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2x8??@??H??bWModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5a_se_squeeze/Meanhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6c_activation/Sigmoidhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2x8ߓ@ߓHߓbWModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4b_se_squeeze/Meanhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bcModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3b_project_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bcModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3a_project_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4a_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*?28??@? H?+bjModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_rotation/rotation_matrix/concathu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*?28??@? H?(bbModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_zoom/zoom_matrix/concathu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<Eigen::half*, Eigen::half*, tensorflow::functor::Sum<Eigen::half> >(Eigen::half*, Eigen::half*, int, int, int, tensorflow::functor::Sum<Eigen::half>)%*?2?8??@??H??bigradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_se_excite/mul/Sum_1hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bcModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5c_project_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bcModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5b_project_bn/FusedBatchNormV3hu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2::Params)? ??*?28??@??H??XbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4c_se_expand/Conv2Dhu  ?A
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2::Params)? ??*?28??@??H??XbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5a_se_expand/Conv2Dhu  ?A
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bcModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5a_project_bn/FusedBatchNormV3hu  ?B
?
?void Eigen::internal::OuterReductionKernel<16, Eigen::TensorReductionEvaluatorBase<Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<0l>> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<long, Eigen::type2index<1l> > const, Eigen::TensorReshapingOp<Eigen::IndexList<Eigen::type2index<1l>, long> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, Eigen::internal::SumReducer<float>, long>(Eigen::internal::SumReducer<float>, Eigen::TensorReductionEvaluatorBase<Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<0l>> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<long, Eigen::type2index<1l> > const, Eigen::TensorReshapingOp<Eigen::IndexList<Eigen::type2index<1l>, long> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long, long, Eigen::TensorReductionEvaluatorBase<Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<0l>> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<long, Eigen::type2index<1l> > const, Eigen::TensorReshapingOp<Eigen::IndexList<Eigen::type2index<1l>, long> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>::CoeffReturnType*)<*?2{8??@??H??bugradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_project_bn/FusedBatchNormGradV3hu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2::Params)? ??*?28??@??H??XbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4b_se_expand/Conv2Dhu  ?A
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2::Params)? ??*?28??@??H??XbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5c_se_expand/Conv2Dhu  ?A
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2::Params)? ??*?28??@??H??XbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_se_expand/Conv2Dhu  ?A
?
?
void Eigen::internal::OuterReductionKernel<16, Eigen::TensorReductionEvaluatorBase<Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<0l>> const, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, Eigen::internal::SumReducer<float>, long>(Eigen::internal::SumReducer<float>, Eigen::TensorReductionEvaluatorBase<Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<0l>> const, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long, long, Eigen::TensorReductionEvaluatorBase<Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<0l>> const, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>::CoeffReturnType*)&*?2?8??@??H??bigradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/top_bn/FusedBatchNormGradV3hu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_256x128_nn_align2::Params)? ??*?28??@??H??XbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5b_se_expand/Conv2Dhu  ?A
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2(8??@??H??bZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_activation/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)6*?28??@??H??bgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?2(8??@??H??b<cond_1/then/_10/cond_1/Adam/Adam/update_10/ResourceApplyAdamhu  ?B
?
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bOModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3b_add/addhu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2?8??@??H??bPModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/global_average_pooling2d_1/Meanhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8߮@߮H߮bcModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4a_project_bn/FusedBatchNormV3hu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2<8??@??H??bWModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4a_se_squeeze/Meanhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bcModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6b_project_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bcModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6d_project_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bcModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6c_project_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bcModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_project_bn/FusedBatchNormV3hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?2?8ߟ@ߟHߟbVModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_activation/mulhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bcModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_project_bn/FusedBatchNormV3hu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2?8??@??H??bWModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_se_squeeze/Meanhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long) *?28??@??H??bArgMaxhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<long, Eigen::type2index<1l> > const, Eigen::TensorReshapingOp<Eigen::IndexList<Eigen::type2index<1l>, long> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half, 4, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorConversionOp<Eigen::half, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<long, Eigen::type2index<1l> > const, Eigen::TensorReshapingOp<Eigen::IndexList<Eigen::type2index<1l>, long> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const> const, Eigen::GpuDevice>, long)*?2(8??@??H??bugradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_project_bn/FusedBatchNormGradV3hu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2?8??@??H??bWModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6d_se_squeeze/Meanhu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align1>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align1::Params)q ??*?2
8??@??H??Xb?Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/dense_1/MatMulhu  HB
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2?8??@??H??bWModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6c_se_squeeze/Meanhu  ?B
?
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bOModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5c_add/addhu  ?B
?
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2?8??@??H??bOModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5b_add/addhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8??@??H??bTModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/top_conv/Conv2D/Casthu  ?B
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2?8??@??H??bWModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6b_se_squeeze/Meanhu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bcModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4c_project_bn/FusedBatchNormV3hu  ?B
?
?void cudnn::bn_fw_inf_1C11_kernel_NHWC<__half, float, true, true>(float, float, cudnnTensorStruct, __half const*, cudnnTensorStruct, __half*, cudnnTensorStruct, float const*, float const*, float const*, float const*, float, int, int) ?*?2 8??@??H??bcModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4b_project_bn/FusedBatchNormV3hu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_tn_align1>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_tn_align1::Params)r ??*?28??@??H??XbMgradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/dense_1/MatMulhu  HB
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28??@??H??b;cond_1/then/_10/cond_1/Adam/Adam/update_5/ResourceApplyAdamhu  ?B
?
 Cast_GPU_DT_HALF_DT_FLOAT_kernel*?2?8??@??H??bsgradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_project_conv/Conv2D/Cast/Casthu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2::Params)` ??*?28??@??H??XbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5b_se_reduce/Conv2Dhu  HB
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?@?H?b`Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_project_conv/Conv2D/Casthu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2::Params)` ??*?28?@?H?XbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_se_reduce/Conv2Dhu  HB
?
 Cast_GPU_DT_HALF_DT_FLOAT_kernel*?2?8?~@?~H?~bggradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/top_conv/Conv2D/Cast/Casthu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2::Params)` ??*?28?|@?|H?|XbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5c_se_reduce/Conv2Dhu  HB
?
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2{8?s@?sH?sbOModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4b_add/addhu  ?B
?
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2{8?q@?qH?qbOModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4c_add/addhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *?2 8?q@?qH?qbAll_10hu  ?B
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2d8?o@?oH?obmul_9hu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2::Params)` ??*?28?o@?oH?oXbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5a_se_reduce/Conv2Dhu  HB
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8::Params)^ ??*?28?m@?mH?mXbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_se_reduce/Conv2Dhu  HA
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?2(8?m@?mH?mb9cond_1/then/_10/cond_1/Adam/Adam/update/ResourceApplyAdamhu  ?B
?
dvoid tensorflow::BiasGradNHWC_SharedAtomics<Eigen::half>(int, Eigen::half const*, Eigen::half*, int) ?*?28?m@?mH?mbZgradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/dense_1/BiasAdd/BiasAddGradhu  ?B
?
?void cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And)+ *?2-8?m@?mH?mbAll_4hu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2::Params)` ??*?28?k@?kH?kXbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4c_se_reduce/Conv2Dhu  HB
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?2(8?g@?gH?gb;cond_1/then/_10/cond_1/Adam/Adam/update_2/ResourceApplyAdamhu  ?B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*?2?8?e@?eH?eb
IsFinite_7hu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nt_align1>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nt_align1::Params)v ??*?2(8?c@?cH?cbOgradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/dense_1/MatMul_1hu  HB
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8::Params)^ ??*?28?a@?aH?aXbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6c_se_reduce/Conv2Dhu  HA
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2Z8?a@?aH?abmul_6hu  ?B
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8::Params)^ ??*?28?`@?`H?`XbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6d_se_reduce/Conv2Dhu  HA
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nn_align2::Params)` ??*?28?`@?`H?`XbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4b_se_reduce/Conv2Dhu  HB
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?`@?`H?`XbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3b_se_reduce/Conv2Dhu  HA
?
*volta_fp16_s884gemm_fp16_64x64_ldg8_f2f_nnj??*?28?`@?`H?`PXbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_se_expand/Conv2Dhu  B
?
?
void Eigen::internal::OuterReductionKernel<16, Eigen::TensorReductionEvaluatorBase<Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<0l>> const, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, Eigen::internal::SumReducer<float>, long>(Eigen::internal::SumReducer<float>, Eigen::TensorReductionEvaluatorBase<Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<0l>> const, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long, long, Eigen::TensorReductionEvaluatorBase<Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<0l>> const, Eigen::TensorConversionOp<float, Eigen::TensorReshapingOp<Eigen::DSizes<long, 2> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 4, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>::CoeffReturnType*)&*?2{8?`@?`H?`bugradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_project_bn/FusedBatchNormGradV3hu  ?B
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8::Params)^ ??*?28?`@?`H?`XbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6b_se_reduce/Conv2Dhu  HA
?
?void tensorflow::functor::ColumnReduceSimpleKernel<cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, cub::Sum>(cub::TransformInputIterator<float, tensorflow::functor::HalfToFloat, Eigen::half*, long>, tensorflow::TransformOutputIterator<Eigen::half, float, tensorflow::functor::DividesBy<float, Eigen::half>, long>, int, int, int, cub::Sum)$*?2?8?`@?`H?`bWModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_se_squeeze/Meanhu  ?B
?
*volta_fp16_s884gemm_fp16_64x64_ldg8_f2f_nnj??*?28?`@?`H?`PXbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6c_se_expand/Conv2Dhu  B
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?_@?_H?_XbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4a_se_reduce/Conv2Dhu  HA
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?_@?_H?_b`Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6c_project_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?_@?_H?_b`Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6d_project_conv/Conv2D/Casthu  ?B
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_tn_align8>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_tn_align8::Params)^ ??*?28?_@?_H?_Xbzgradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_se_expand/Conv2D/Conv2DBackpropInputhu  HA
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?^@?^H?^b`Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6b_project_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?^@?^H?^b_Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_expand_conv/Conv2D/Casthu  ?B
?
?void cutlass::Kernel<cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nt_align8>(cutlass_75_tensorop_f16_s1688gemm_f16_64x64_nt_align8::Params)` ??*?28?]@?]H?]Xb{gradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_se_reduce/Conv2D/Conv2DBackpropFilterhu  HB
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?\@?\H?\XbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3a_se_reduce/Conv2Dhu  HA
?
?void tensorflow::(anonymous namespace)::DynamicStitchKernel<int>(int, int, tensorflow::GpuDeviceArrayStruct<int, 8>, tensorflow::GpuDeviceArrayStruct<int const*, 8>, int*)*?28?\@?\H?\bggradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/global_average_pooling2d_1/DynamicStitchhu  ?B
?
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2J8?\@?\H?\bOModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6c_add/addhu  ?B
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?\@?\H?\XbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2b_se_reduce/Conv2Dhu  HA
?
*volta_fp16_s884gemm_fp16_64x64_ldg8_f2f_nnj??*?28?\@?\H?\PXbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6d_se_expand/Conv2Dhu  B
?
*volta_fp16_s884gemm_fp16_64x64_ldg8_f2f_nnj??*?28?Z@?ZH?ZPXbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6b_se_expand/Conv2Dhu  B
?
*volta_fp16_s884gemm_fp16_64x64_ldg8_f2f_ntj??*?28?X@?XH?XPXb{gradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_se_expand/Conv2D/Conv2DBackpropFilterhu  HB
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?X@?XH?Xb_Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6c_expand_conv/Conv2D/Casthu  ?B
?
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2J8?U@?UH?UbOModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6b_add/addhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?U@?UH?Ub_Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6d_expand_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?T@?TH?Tb_Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6b_expand_conv/Conv2D/Casthu  ?B
?
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?2J8?Q@?QH?QbOModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6d_add/addhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?P@?PH?PbYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4c_se_expand/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2$8?P@?PH?PbYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6b_se_expand/Sigmoidhu  ?B
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?P@?PH?PXbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_se_expand/Conv2Dhu  HA
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?O@?OH?ObYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5a_se_expand/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?O@?OH?ObYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5b_se_expand/Sigmoidhu  ?B
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?O@?OH?OXbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2b_se_expand/Conv2Dhu  HA
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?N@?NH?NbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6d_dwconv/depthwisehu  ?B
?
dvoid tensorflow::BiasGradNHWC_SharedAtomics<Eigen::half>(int, Eigen::half const*, Eigen::half*, int) ?*?28?N@?NH?Nbsgradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_se_reduce/BiasAdd/BiasAddGradhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?N@?NH?NbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6c_dwconv/depthwisehu  ?B
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?M@?MH?MXbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_se_reduce/Conv2Dhu  HA
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?M@?MH?MbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_dwconv/depthwisehu  ?B
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?M@?MH?MXbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3a_se_expand/Conv2Dhu  HA
?
Kvoid Eigen::internal::ReductionInitKernel<float, long>(float, long, float*)*?28?L@? H?,bigradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/top_bn/FusedBatchNormGradV3hu  ?B
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?L@?LH?LXbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4a_se_expand/Conv2Dhu  HA
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align2::Params)P ??*?28?K@?KH?KXbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3b_se_expand/Conv2Dhu  HA
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2$8?K@?KH?KbYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6c_se_expand/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2$8?J@?JH?JbYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_se_expand/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?J@?JH?JbYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4b_se_expand/Sigmoidhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?I@?IH?IbDModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/dense_1/MatMul/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?H@?HH?Hb`Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_project_conv/Conv2D/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?H@?HH?HbYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3b_se_expand/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?G@?GH?GbYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5c_se_expand/Sigmoidhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?G@?GH?GbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6b_dwconv/depthwisehu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?2$8?F@?FH?FbYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6d_se_expand/Sigmoidhu  ?B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*?2?8?E@?EH?Eb
IsFinite_4hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?E@?EH?EbYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4a_se_expand/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?C@?CH?CbYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_se_expand/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*?28?C@?H?#bVModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_zoom/concathu  ?B
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8::Params)^ ??*?28?B@?BH?BXbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block1a_se_reduce/Conv2Dhu  HA
?
?void cutlass::Kernel<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8>(cutlass_75_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_nn_align8::Params)^ ??*?28?@@?@H?@XbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block1a_se_expand/Conv2Dhu  HA
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?@@?@H?@bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2b_se_expand/Sigmoidhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?@@?@H?@b;cond_1/then/_10/cond_1/Adam/Adam/update_3/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?@@?@H?@b;cond_1/then/_10/cond_1/Adam/Adam/update_9/ResourceApplyAdamhu  ?B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*?28??@??H??bIsFinite_10hu  ?B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?2 8??@??H??bmul_12hu  ?B
?
?void cub::DeviceReduceKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And>(bool*, bool*, int, cub::GridEvenShare<int>, tensorflow::functor::And)+ *?228??@??H??bAll_7hu  ?B
?
?void tensorflow::(anonymous namespace)::GenerateNormalizedProb<float, float, 4>(float const*, float const*, float const*, float*, int, int, bool)%*?28??@??H??bHModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/softmax_float32/Softmaxhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28??@??H??bXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3a_dwconv/depthwisehu  ?B
?
Kvoid Eigen::internal::ReductionInitKernel<float, long>(float, long, float*)*?28??@?H? bugradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_project_bn/FusedBatchNormGradV3hu  ?B
?
?void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *?28?=@?=H?=bAll_8hu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?=@?=H?=b;cond_1/then/_10/cond_1/Adam/Adam/update_8/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?<@?<H?<bXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4c_dwconv/depthwisehu  ?B
?
?void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *?2 8?<@?<H?<bAllhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *?2 8?<@?<H?<bAll_2hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?;@?;H?;b_Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5b_expand_conv/Conv2D/Casthu  ?B
?
?void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)'*?28?;@?;H?;byModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_zoom/stateful_uniform/StatelessRandomUniformV2hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?:@?:H?:bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6d_se_reduce/Sigmoidhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?:@?:H?:bXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3b_dwconv/depthwisehu  ?B
?
?void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)'*?28?:@?:H?:b?Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2hu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?:@?:H?:bXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4b_dwconv/depthwisehu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?9@?9H?9bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_se_expand/Sigmoidhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?9@?9H?9b<cond_1/then/_10/cond_1/Adam/Adam/update_11/ResourceApplyAdamhu  ?B
?
 Cast_GPU_DT_HALF_DT_FLOAT_kernel*?2?8?9@?9H?9bWgradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/dense_1/MatMul/Cast/Casthu  ?B
?
?void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)2 *?28?8@?8H?8bAll_7hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?8@?8H?8bAdam/gradients/Sigmoid_1hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?7@?7H?7b_Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_expand_conv/Conv2D/Casthu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?7@?7H?7bXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5c_dwconv/depthwisehu  ?B
?
?void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformFullIntDistribution<tensorflow::random::PhiloxRandom, long> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformFullIntDistribution<tensorflow::random::PhiloxRandom, long>::ResultElementType*, long, tensorflow::random::UniformFullIntDistribution<tensorflow::random::PhiloxRandom, long>)'*?28?7@?7H?7biModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_flip/stateful_uniform_full_inthu  ?B
?
?void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *?28?6@?6H?6bAll_9hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?6@?6H?6bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3a_se_expand/Sigmoidhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?5@?5H?5bXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2b_dwconv/depthwisehu  ?B
?
?void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)'*?28?5@?5H?5b}Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_rotation/stateful_uniform/StatelessRandomUniformV2hu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?5@?5H?5bXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5b_dwconv/depthwisehu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?5@?5H?5bXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4a_dwconv/depthwisehu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?5@?5H?5bXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5a_dwconv/depthwisehu  ?B
?

?	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorReshapingOp<Eigen::IndexList<int, Eigen::type2index<1l> > const, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> > const> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorReshapingOp<Eigen::IndexList<int, Eigen::type2index<1l> > const, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> > const> const> const> const, Eigen::GpuDevice>, int)*?28?3@?3H?3bgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?3@?3H?3Xb?Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/dense_1/MatMulhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?3@?3H?3b;cond_1/then/_10/cond_1/Adam/Adam/update_6/ResourceApplyAdamhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?3@?3H?3b`Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5c_project_conv/Conv2D/Casthu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?3@?3H?3bXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_dwconv/depthwisehu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?3@?3H?3b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6c_se_reduce/Conv2D/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sigmoid_gradient_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sigmoid_gradient_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?2$8?2@?2H?2bsgradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_se_expand/Sigmoid/SigmoidGradhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?2@?2H?2bXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_dwconv/depthwisehu  ?B
?
?void tensorflow::functor::RowReduceKernel<cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long>, float*, cub::Sum>(cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long>, float*, int, int, cub::Sum, std::iterator_traits<cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long> >::value_type)*?28?2@?2H?2bHModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/softmax_float32/Softmaxhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?2@?2H?2b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_se_reduce/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?1@?1H?1b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6c_se_expand/Conv2D/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?1@?1H?1bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block1a_se_reduce/Sigmoidhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?1@?1H?1b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6d_se_expand/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?0@?0H?0b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_se_expand/Conv2D/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?0@?0H?0bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4c_se_reduce/Sigmoidhu  ?B
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?0@?0H?0bmul_4hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?0@?0H?0bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3b_se_reduce/Sigmoidhu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?0@?0H?0XbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6b_se_reduce/Conv2Dhu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?0@?0H?0XbPModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/stem_conv/Conv2Dhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?0@?0H?0bdiv_no_nan_1hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2q8?0@?0H?0b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6c_dwconv/depthwise/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?0@?0H?0b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6b_se_reduce/Conv2D/Casthu  ?B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?0@?0H?0bmul_13hu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?0@?0H?0bUgradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mulhu  ?B
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?0@?0H?0bmul_2hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?0@?0H?0bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_se_reduce/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28?0@?0H?0bgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorGeneratorOp<tensorflow::generator::SparseXentGradGenerator<float, long>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorGeneratorOp<tensorflow::generator::SparseXentGradGenerator<float, long>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28?0@?0H?0bgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?0@?0H?0XbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4b_se_reduce/Conv2Dhu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?0@?0H?0XbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5b_se_reduce/Conv2Dhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?0@?0H?0bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block1a_se_expand/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?0@?0H?0bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4a_se_reduce/BiasAddhu  ?B
?
?void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*?28?0@?0H?0b;cond_1/then/_10/cond_1/Adam/Adam/update_1/ResourceApplyAdamhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *?28?0@?0H?0bAll_5hu  ?B
?
?void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *?28?0@?0H?0bAll_6hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?/@?/H?/b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6b_se_expand/Conv2D/Casthu  ?B
?

?	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_rsqrt_op<float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_rsqrt_op<float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const, Eigen::GpuDevice>, long)*?28?/@?/H?/bigradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/top_bn/FusedBatchNormGradV3hu  ?B
?
?void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *?28?/@?/H?/bAll_3hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?/@?/H?/b`Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4c_project_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?/@?/H?/b`Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5a_project_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?/@?/H?/b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6d_se_reduce/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?/@?/H?/b`Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5b_project_conv/Conv2D/Casthu  ?B
`
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?/@?/H?/bcond_1/then/_10/cond_1/Adam/Powhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?/@?/H?/bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_se_reduce/Sigmoidhu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?/@?/H?/XbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4c_se_reduce/Conv2Dhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?/@?/H?/bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5b_se_expand/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?2$8?/@?/H?/bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6c_se_expand/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?2$8?/@?/H?/bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_se_expand/BiasAddhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?/@?/H?/b_Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5c_expand_conv/Conv2D/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?/@?/H?/bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6c_se_reduce/Sigmoidhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?/@?/H?/b
div_no_nanhu  ?B
?
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?/@?/H?/baModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_zoom/zoom_matrix/sub_2hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?/@?/H?/bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_se_reduce/Sigmoidhu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?/@?/H?/XbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5a_se_reduce/Conv2Dhu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?/@?/H?/XbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5c_se_reduce/Conv2Dhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?/@?/H?/bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6b_se_reduce/Sigmoidhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?/@?/H?/bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4c_se_reduce/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?2$8?/@?/H?/bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6b_se_expand/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?2$8?/@?/H?/bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6d_se_expand/BiasAddhu  ?B
F
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28?.@?.H?.bCasthu  ?B
?
!Cast_GPU_DT_INT64_DT_FLOAT_kernel*?28?.@?.H?.b`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Casthu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?.@?.H?.bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4b_se_expand/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?.@?.H?.bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5c_se_reduce/BiasAddhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?28?.@?.H?.bSum_2hu  ?B
?
?void tensorflow::functor::ShuffleInTensor3Simple<Eigen::half, 1, 2, 0, false>(int, Eigen::half const*, tensorflow::functor::Dimension<3>, Eigen::half*)*?28?.@?.H?.bXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block1a_dwconv/depthwisehu  ?B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*?28?.@?.H?.b
IsFinite_9hu  ?B
?
?void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Max>(float const*, float*, int, int, cub::Max, std::iterator_traits<float const*>::value_type)**?28?.@?.H?.bHModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/softmax_float32/Softmaxhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2B8?.@?.H?.b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_dwconv/depthwise/Casthu  ?B
P
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*?28?.@?.H?.bIsFinite_11hu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?.@?.H?.XbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_se_reduce/Conv2Dhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?-@?-H?-bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2b_se_reduce/Sigmoidhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*?28?-@?-H?-b1sparse_categorical_crossentropy/weighted_loss/Sumhu  ?B
?
?void tensorflow::functor::CleanupSegments<bool*, bool*, tensorflow::functor::And>(bool*, bool*, int, int, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type)* 28?-@?-H?-bAll_10hu  HB
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?-@?-H?-b_Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5a_expand_conv/Conv2D/Casthu  ?B
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?-@?-H?-bmul_8hu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?-@?-H?-b@Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/dense_1/BiasAddhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?-@?-H?-b`Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4b_project_conv/Conv2D/Casthu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?-@?-H?-XbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6d_se_reduce/Conv2Dhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?-@?-H?-bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_se_expand/BiasAddhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?-@?-H?-b_Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4b_expand_conv/Conv2D/Casthu  ?B
S
Sub_GPU_DT_HALF_DT_HALF_kernel*?28?-@?-H?-bAdam/gradients/sub_1hu  ?B
?
?void cub::DeviceReduceSingleTileKernel<cub::DeviceReducePolicy<bool, bool, int, tensorflow::functor::And>::Policy600, bool*, bool*, int, tensorflow::functor::And, bool>(bool*, bool*, int, tensorflow::functor::And, bool)2 *?28?-@?-H?-bAll_4hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?,@?,H?,bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5b_se_reduce/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?,@?,H?,b>cond/then/_0/cond/cond/else/_195/cond/cond/AssignAddVariableOphu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?,@?,H?,bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block1a_se_reduce/BiasAddhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?,@?,H?,bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5a_se_reduce/Sigmoidhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2&8?,@?,H?,b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4b_se_reduce/Conv2D/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?,@?,H?,bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5c_se_reduce/Sigmoidhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2&8?,@?,H?,b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4c_se_expand/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2/8?,@?,H?,b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5a_dwconv/depthwise/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2q8?,@?,H?,b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6b_dwconv/depthwise/Casthu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?,@?,H?,bdModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_zoom/stateful_uniform/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?,@?,H?,bAssignAddVariableOphu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2	8?,@?,H?,b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4a_dwconv/depthwise/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2J8?,@?,H?,b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5b_se_expand/Conv2D/Casthu  ?B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?+@?+H?+bmul_10hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?+@?+H?+b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2?8?+@?+H?+b_Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4c_expand_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?+@?+H?+b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block1a_se_expand/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?+@?+H?+b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4c_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?+@?+H?+b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5c_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2J8?+@?+H?+b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5c_se_expand/Conv2D/Casthu  ?B
S
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?+@?+H?+bAdam/gradients/mul_3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?+@?+H?+bigradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/top_bn/FusedBatchNormGradV3hu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?+@?+H?+XbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_se_reduce/Conv2Dhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?+@?+H?+bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4a_se_expand/BiasAddhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?+@?+H?+bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4b_se_reduce/Sigmoidhu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?+@?+H?+XbXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6c_se_reduce/Conv2Dhu  ?B
?
 Cos_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?+@?+H?+biModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_rotation/rotation_matrix/Cos_2hu  ?B
?
Sub_GPU_DT_HALF_DT_HALF_kernel*?28?+@?+H?+btModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_flip/stateless_random_flip_left_right/subhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?+@?+H?+bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3a_se_reduce/Sigmoidhu  ?B
?

?	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_rsqrt_op<float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_rsqrt_op<float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const, Eigen::GpuDevice>, long)*?28?+@?+H?+bugradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_project_bn/FusedBatchNormGradV3hu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?+@?+H?+bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6b_se_reduce/BiasAddhu  ?B
?
?void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Max>(float const*, float*, int, int, cub::Max, std::iterator_traits<float const*>::value_type)**?28?*@?*H?*bgsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitshu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2J8?*@?*H?*b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_se_expand/Conv2D/Casthu  ?B
S
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?*@?*H?*bAdam/gradients/mul_4hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 1> const, Eigen::DSizes<int, 1> const, Eigen::TensorMap<Eigen::Tensor<double const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 1> const, Eigen::DSizes<int, 1> const, Eigen::TensorMap<Eigen::Tensor<double const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28?*@?*H?*bpModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_zoom/stateful_uniform/strided_slice_1hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?*@?*H?*b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_se_expand/BiasAdd/Casthu  ?B
D
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?*@?*H?*bMulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?*@?*H?*bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block1a_se_expand/Sigmoidhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?*@?*H?*bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3a_se_expand/BiasAddhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?*@?*H?*b_Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_expand_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?*@?*H?*b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4a_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?*@?*H?*b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6b_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?*@?*H?*b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4b_dwconv/depthwise/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2&8?*@?*H?*b_Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4a_expand_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2&8?*@?*H?*b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5a_se_expand/Conv2D/Casthu  ?B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*?28?*@?*H?*b
IsFinite_1hu  ?B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*?28?*@?*H?*b
IsFinite_6hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?*@?*H?*bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4a_se_reduce/Sigmoidhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?*@?*H?*bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3b_se_expand/BiasAddhu  ?B
?
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28?*@?*H?*b?sparse_categorical_crossentropy/weighted_loss/num_elements/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?*@?*H?*b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block1a_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?*@?*H?*b`Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3a_project_conv/Conv2D/Casthu  ?B
?
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?*@?*H?*b_Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_zoom/zoom_matrix/subhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?)@?)H?)b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4b_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2
8?)@?)H?)b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4a_se_expand/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2J8?)@?)H?)b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5c_se_reduce/Conv2D/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28?)@?)H?)bBgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1hu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?)@?)H?)bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5b_se_reduce/BiasAddhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2B8?)@?)H?)b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5b_dwconv/depthwise/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 1> const, Eigen::DSizes<int, 1> const, Eigen::TensorMap<Eigen::Tensor<double const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 1> const, Eigen::DSizes<int, 1> const, Eigen::TensorMap<Eigen::Tensor<double const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28?)@?)H?)bwModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_flip/stateful_uniform_full_int/strided_slicehu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?)@?)H?)b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2b_dwconv/depthwise/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?)@?)H?)b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2b_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2&8?)@?)H?)b`Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3b_project_conv/Conv2D/Casthu  ?B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*?28?)@?)H?)b
IsFinite_3hu  ?B
?
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28?(@?(H?(bTModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_zoom/Casthu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?(@?(H?(bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_se_reduce/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 1> const, Eigen::DSizes<int, 1> const, Eigen::TensorMap<Eigen::Tensor<double const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 1> const, Eigen::DSizes<int, 1> const, Eigen::TensorMap<Eigen::Tensor<double const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28?(@?(H?(btModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_rotation/stateful_uniform/strided_slice_1hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?(@?(H?(b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5c_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2
8?(@?(H?(b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3b_se_expand/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?(@?(H?(b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3b_dwconv/depthwise/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2J8?(@?(H?(b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5b_se_reduce/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_HALF_DT_FLOAT_kernel*?28?(@?(H?(bqgradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_se_reduce/BiasAdd/Cast/Casthu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?(@?(H?(biModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_rotation/rotation_matrix/mul_2hu  ?B
?
?void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *?28?(@?(H?(bAll_12hu  ?B
?
?void tensorflow::functor::BlockReduceKernel<int*, int*, 256, tensorflow::functor::Prod<int> >(int*, int*, int, tensorflow::functor::Prod<int>, std::iterator_traits<int*>::value_type)0*?28?(@?(H?(beModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_zoom/stateful_uniform/Prodhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?(@?(H?(b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_se_reduce/BiasAdd/Casthu  ?B
P
%LogicalAnd_GPU_DT_BOOL_DT_BOOL_kernel*?28?(@?(H?(b
LogicalAndhu  ?B
?
 Cast_GPU_DT_HALF_DT_FLOAT_kernel*?28?(@?(H?(b?Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/softmax_float32/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?(@?(H?(b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_se_expand/BiasAdd/Casthu  ?B
?
?void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *?28?(@?(H?(bAll_1hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?(@?(H?(b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4a_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2&8?(@?(H?(b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4c_se_reduce/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_HALF_DT_FLOAT_kernel*?2?8?(@?(H?(bpgradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_se_expand/Conv2D/Cast/Casthu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?(@?(H?(bmModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_rotation/rotation_matrix/truediv_1hu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?(@?(H?(bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_se_reduce/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?(@?(H?(bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5c_se_expand/BiasAddhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 1> const, Eigen::DSizes<int, 1> const, Eigen::TensorMap<Eigen::Tensor<double const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 1> const, Eigen::DSizes<int, 1> const, Eigen::TensorMap<Eigen::Tensor<double const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28?'@?'H?'bnModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_zoom/stateful_uniform/strided_slicehu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?'@?'H?'bAssignAddVariableOp_3hu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?'@?'H?'bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_se_reduce/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?'@?'H?'bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3a_se_reduce/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?'@?'H?'bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4b_se_reduce/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?'@?'H?'bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6c_se_reduce/BiasAddhu  ?B
?
?void tensorflow::functor::ColumnReduceKernel<Eigen::half const*, Eigen::half*, cub::Sum>(Eigen::half const*, Eigen::half*, int, int, cub::Sum, std::iterator_traits<Eigen::half const*>::value_type)?*  2$8?'@?'H?'bsgradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_se_expand/BiasAdd/BiasAddGradhu  ?B
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?'@?'H?'bmul_7hu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?'@?'H?'bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_se_expand/BiasAddhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?&@?&H?&bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6b_se_reduce/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?&@?&H?&bLgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nanhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 2> const, Eigen::DSizes<int, 2> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 2> const, Eigen::DSizes<int, 2> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28?&@?&H?&bkModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_zoom/zoom_matrix/strided_slice_3hu  ?B
?
?void tensorflow::functor::CleanupSegments<bool*, bool*, tensorflow::functor::And>(bool*, bool*, int, int, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type)* 28?&@?&H?&bAll_2hu  HB
?
 Cast_GPU_DT_HALF_DT_FLOAT_kernel*?2?8?&@?&H?&bpgradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_se_reduce/Conv2D/Cast/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?&@?&H?&b3sparse_categorical_crossentropy/weighted_loss/valuehu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2)8?&@?&H?&b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_dwconv/depthwise/Casthu  ?B
?
"Maximum_GPU_DT_HALF_DT_HALF_kernel*?28?&@?&H?&bWModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/normalization_1/Maximumhu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?&@?&H?&bhModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_rotation/stateful_uniform/mulhu  ?B
b
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?&@?&H?&b!cond_1/then/_10/cond_1/Adam/Pow_1hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_round_half_to_even_op<float, false, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_round_half_to_even_op<float, false, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28?&@?&H?&bvModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_flip/stateless_random_flip_left_right/Roundhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2&8?&@?&H?&b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5a_se_reduce/Conv2D/Casthu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?&@?&H?&bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5a_se_reduce/BiasAddhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2B8?%@?%H?%b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5c_dwconv/depthwise/Casthu  ?B
U
 AddV2_GPU_DT_HALF_DT_HALF_kernel*?28?%@?%H?%bAdam/gradients/add_1hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2&8?%@?%H?%b_Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3b_expand_conv/Conv2D/Casthu  ?B
S
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?%@?%H?%bAdam/gradients/mul_5hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_const_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_const_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28?%@?%H?%baModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_zoom/zoom_matrix/zeroshu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?%@?%H?%bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5a_se_expand/BiasAddhu  ?B
b
"AddV2_GPU_DT_INT64_DT_INT64_kernel*?28?%@?%H?%bcond_1/then/_10/cond_1/Adam/addhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2
8?%@?%H?%b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3b_se_reduce/Conv2D/Casthu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?%@?%H?%bkModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_rotation/rotation_matrix/truedivhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?%@?%H?%bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6d_se_reduce/mulhu  ?B
?
 Sin_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?%@?%H?%biModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_rotation/rotation_matrix/Sin_2hu  ?B
?
Sqrt_GPU_DT_HALF_DT_HALF_kernel*?28?%@?%H?%bTModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/normalization_1/Sqrthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?%@?%H?%bugradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_project_bn/FusedBatchNormGradV3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?%@?%H?%bAssignAddVariableOp_1hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?%@?%H?%b4cond_1/then/_10/cond_1/Adam/Adam/AssignAddVariableOphu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?%@?%H?%bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2b_se_reduce/BiasAddhu  ?B
?
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?%@?%H?%b`Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_zoom/stateful_uniformhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?%@?%H?%bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4b_se_reduce/mulhu  ?B
?
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?%@?%H?%bgModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_rotation/rotation_matrix/addhu  ?B
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*?28?%@?%H?%b
IsFinite_8hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?%@?%H?%bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4a_se_reduce/mulhu  ?B
?
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?%@?%H?%bgModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_rotation/rotation_matrix/subhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?%@?%H?%bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4c_se_expand/BiasAddhu  ?B
?
?void tensorflow::functor::CleanupSegments<bool*, bool*, tensorflow::functor::And>(bool*, bool*, int, int, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type)* 28?%@?%H?%bAllhu  HB
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*?268?$@?$H?$b
IsFinite_2hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?$@?$H?$bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3a_se_reduce/mulhu  ?B
?
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?$@?$H?$biModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_rotation/rotation_matrix/sub_3hu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?$@?$H?$bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2b_se_expand/BiasAddhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?$@?$H?$b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_se_reduce/Conv2D/Casthu  ?B
G
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?$@?$H?$bmul_11hu  ?B
?
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?$@?$H?$biModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_rotation/rotation_matrix/sub_4hu  ?B
?
?void splitKreduce_kernel<__half, __half, float, __half>(cublasSplitKParams<float>, __half const*, __half const*, __half*, float const*, float const*, __half const*)**?28?$@?$H?$Xbzgradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_se_expand/Conv2D/Conv2DBackpropInputhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?$@?$H?$b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6d_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2J8?$@?$H?$b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_se_reduce/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2K8?$@?$H?$b`Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4a_project_conv/Conv2D/Casthu  ?B
?
!Cast_GPU_DT_FLOAT_DT_INT64_kernel*?28?$@?$H?$bbsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1hu  ?B
O
'Reciprocal_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?$@?$H?$btruedivhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?$@?$H?$b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block1a_se_reduce/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?$@?$H?$b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6b_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2
8?$@?$H?$b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4a_se_reduce/Conv2D/Casthu  ?B
g
(GreaterEqual_GPU_DT_INT64_DT_BOOL_kernel*?28?$@?$H?$bcond/then/_0/cond/GreaterEqualhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?$@?$H?$bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5c_se_reduce/mulhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?#@?#H?#b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3b_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?#@?#H?#b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3a_se_reduce/Conv2D/Casthu  ?B
M
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*?268?#@?#H?#bIsFinitehu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?#@?#H?#bgModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_rotation/rotation_matrix/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?#@?#H?#bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5a_se_reduce/mulhu  ?B
?
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?#@?#H?#bgModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_rotation/rotation_matrix/Neghu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<Eigen::half>, Eigen::TensorMap<Eigen::Tensor<Eigen::half const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)!*?28?#@?#H?#bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5b_se_reduce/Sigmoidhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?#@?#H?#b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5a_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?#@?#H?#bEModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/dense_1/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?#@?#H?#b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block1a_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?#@?#H?#b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?#@?#H?#bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/stem_conv/Conv2D/Casthu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?#@?#H?#biModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_rotation/rotation_matrix/mul_1hu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?#@?#H?#biModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_rotation/rotation_matrix/mul_3hu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?#@?#H?#b_Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_zoom/zoom_matrix/mulhu  ?B
?
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?#@?#H?#baModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_zoom/zoom_matrix/sub_3hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 2> const, Eigen::DSizes<int, 2> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 2> const, Eigen::DSizes<int, 2> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28?#@?#H?#bkModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_zoom/zoom_matrix/strided_slice_4hu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?#@?#H?#bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_se_reduce/BiasAddhu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?#@?#H?#bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6d_se_reduce/BiasAddhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?#@?#H?#b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3a_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?#@?#H?#b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6c_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?#@?#H?#b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6d_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?#@?#H?#b`Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block1a_project_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?#@?#H?#b?gradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/softmax_float32/Cast/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<unsigned long, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<unsigned long, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?#@?#H?#bgModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_zoom/stateful_uniform/Cast_1hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?#@?#H?#b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3b_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?#@?#H?#b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5b_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2&8?#@?#H?#b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4b_se_expand/Conv2D/Casthu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?#@?#H?#bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block1a_se_reduce/mulhu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 1> const, Eigen::DSizes<int, 1> const, Eigen::TensorMap<Eigen::Tensor<double const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 1> const, Eigen::DSizes<int, 1> const, Eigen::TensorMap<Eigen::Tensor<double const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28?#@?#H?#brModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_rotation/stateful_uniform/strided_slicehu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28?#@?#H?#bAssignAddVariableOp_4hu  ?B
H
!Cast_GPU_DT_INT64_DT_FLOAT_kernel*?28?"@?"H?"bCast_2hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?"@?"H?"bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2b_se_reduce/mulhu  ?B
?
?void tensorflow::functor::BlockReduceKernel<bool*, bool*, 256, tensorflow::functor::And>(bool*, bool*, int, tensorflow::functor::And, std::iterator_traits<bool*>::value_type) *?28?"@?"H?"bAll_11hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?"@?"H?"b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4c_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?"@?"H?"b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_dwconv/depthwise/Casthu  ?B
?
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28?"@?"H?"bVModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_zoom/Cast_1hu  ?B
?
 Cast_GPU_DT_INT32_DT_HALF_kernel*?28?"@?"H?"b^gradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/global_average_pooling2d_1/Casthu  ?B
G
!Equal_GPU_DT_FLOAT_DT_BOOL_kernel*?28?"@?"H?"bEqualhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?"@?"H?"b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5a_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?"@?"H?"b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3a_se_expand/Conv2D/Casthu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?"@?"H?"beModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_zoom/zoom_matrix/truediv_1hu  ?B
?
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?"@?"H?"baModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_zoom/zoom_matrix/sub_1hu  ?B
?
Btensorflow::functor::SkipKernel(long const*, unsigned long, long*)*28?"@?"H?"boModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_zoom/stateful_uniform/RngReadAndSkiphu  ??
O
$IsFinite_GPU_DT_FLOAT_DT_BOOL_kernel*?28?!@?!H?!b
IsFinite_5hu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?!@?!H?!baModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_zoom/zoom_matrix/mul_1hu  ?B
?
Btensorflow::functor::SkipKernel(long const*, unsigned long, long*)*28?!@?!H?!bsModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_rotation/stateful_uniform/RngReadAndSkiphu  ??
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2	8?!@?!H?!b`Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_project_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?!@?!H?!b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4c_dwconv/depthwise/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?2q8?!@?!H?!b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6d_dwconv/depthwise/Casthu  ?B
H
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28?!@?!H?!bCast_4hu  ?B
?
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?!@?!H?!biModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_rotation/rotation_matrix/sub_8hu  ?B
?
lvoid tensorflow::BiasNHWCKernel<Eigen::half>(int, Eigen::half const*, Eigen::half const*, Eigen::half*, int)*?28?!@?!H?!bYModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3b_se_reduce/BiasAddhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?!@?!H?!bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4c_se_reduce/mulhu  ?B
?
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28?!@?!H?!bXModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_rotation/Casthu  ?B
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28?!@?!H?!bmul_5hu  ?B
?
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28? @? H? bcModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_zoom/zoom_matrix/truedivhu  ?B
F
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*?28? @? H? bmul_3hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28? @? H? bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_se_reduce/mulhu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2b_se_reduce/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? b`Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2b_project_conv/Conv2D/Casthu  ?B
?
Btensorflow::functor::SkipKernel(long const*, unsigned long, long*)*28? @? H? bxModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_flip/stateful_uniform_full_int/RngReadAndSkiphu  ??
X
"AddV2_GPU_DT_INT64_DT_INT64_kernel*?28? @? H? bcond/then/_0/cond/addhu  ?B
?
 Cast_GPU_DT_HALF_DT_FLOAT_kernel*?28? @? H? bXgradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/dense_1/BiasAdd/Cast/Casthu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 1> const, Eigen::DSizes<int, 1> const, Eigen::TensorMap<Eigen::Tensor<double const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<double, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSlicingOp<Eigen::DSizes<int, 1> const, Eigen::DSizes<int, 1> const, Eigen::TensorMap<Eigen::Tensor<double const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28? @? H? byModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_flip/stateful_uniform_full_int/strided_slice_1hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<unsigned long, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<unsigned long, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28? @? H? bkModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_rotation/stateful_uniform/Cast_1hu  ?B
?
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*?28? @? H? bdModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_rotation/stateful_uniformhu  ?B
G
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*?28? @? H? bCast_3hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? buModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_flip/stateless_random_flip_left_right/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2b_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block5b_se_reduce/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block1a_dwconv/depthwise/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block4b_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2b_se_expand/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28? @? H? b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3a_dwconv/depthwise/Casthu  ?B
?
 Cast_GPU_DT_HALF_DT_FLOAT_kernel*?28? @? H? bqgradient_tape/Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_se_expand/BiasAdd/Cast/Casthu  ?B
?
 Cast_GPU_DT_HALF_DT_FLOAT_kernel*?28? @? H? bEModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/softmax_float32/Casthu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28? @? H? bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6c_se_reduce/mulhu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28? @? H? bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block7a_se_reduce/mulhu  ?B
?
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*?28? @? H? biModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_rotation/rotation_matrix/sub_2hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_const_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_const_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28? @? H? biModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_rotation/rotation_matrix/zeroshu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_const_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_const_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*?28? @? H? bcModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_zoom/zoom_matrix/zeros_2hu  ?B
?
?void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*?28? @? H? bAssignAddVariableOp_2hu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?@?H?b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6c_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?@?H?b_Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2b_expand_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?@?H?b_Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3a_expand_conv/Conv2D/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?@?H?b^Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3a_se_expand/BiasAdd/Casthu  ?B
?
 Cast_GPU_DT_FLOAT_DT_HALF_kernel*?28?@?H?b]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_se_expand/Conv2D/Casthu  ?B
?
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*?28?@?H?bZModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_rotation/Cast_1hu  ?B
d
!Cast_GPU_DT_INT64_DT_FLOAT_kernel*?28?@?H?b"cond_1/then/_10/cond_1/Adam/Cast_1hu  ?B
?
Mul_GPU_DT_HALF_DT_HALF_kernel*?28?@?H?bUModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block3b_se_reduce/mulhu  ?B