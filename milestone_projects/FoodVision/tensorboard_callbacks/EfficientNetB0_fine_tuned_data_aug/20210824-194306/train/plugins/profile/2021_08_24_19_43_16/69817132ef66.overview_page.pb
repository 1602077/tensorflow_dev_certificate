?*	??i?W@??i?W@!??i?W@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'??i?W@ ???@1?D?k??R@I?0&???*@r0*W-?'?@?n???
A2?
RIterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2$TT?J??W@!z??I\wE@)TT?J??W@1z??I\wE@:Preprocessing2p
9Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2 C˺,?S@!~????A@)C˺,?S@1~????A@:Preprocessing2X
!Iterator::Root::Prefetch::BatchV2[}uU?[@!??]϶qH@)????:=@1??v?h*@:Preprocessing2z
CIterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch ???9L)@!??????@)???9L)@1??????@:Preprocessing2?
aIterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2$?(^ems
@!????b???)?(^ems
@1????b???:Preprocessing2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[7]::FlatMap[0]::TFRecord?]?????!8??n&??)?]?????18??n&??:Advanced file read2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[6]::FlatMap[0]::TFRecordz?S?4???!??????)z?S?4???1??????:Advanced file read2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4$?? v???!??BL?E??)?? v???1??BL?E??:Preprocessing2?
tIterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality$???A ??!RM?b???)q??Z??1 ?׫?J??:Preprocessing2a
*Iterator::Root::Prefetch::BatchV2::Shuffle ?W?\?S@!?r????A@)}??????1?n??? ??:Preprocessing2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[5]::FlatMap[0]::TFRecord6??`?
??!?]?rr̞?)6??`?
??1?]?rr̞?:Advanced file read2O
Iterator::Root::Prefetchٯ;?y???!ɛ?jc??)ٯ;?y???1ɛ?jc??:Preprocessing2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[6]::FlatMapԻx?n???!????9d??)?eO?s??1Q?N????:Preprocessing2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[8]::FlatMap[0]::TFRecord???g???!?l?????)???g???1?l?????:Advanced file read2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[7]::FlatMap+?ެA??!o??Af??)???[??1kCC.???:Preprocessing2E
Iterator::Root??>eĵ?!Y[G????)xcAaP???1?-?!???:Preprocessing2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[5]::FlatMap??%?L1??!?.³???)???!9???1X?5#?9??:Preprocessing2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[8]::FlatMap???}?A??!?T?С???)?F>?x?q?1??? &0`?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?14.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?cit?,3@Q??"?4T@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	 ???@ ???@! ???@      ??!       "	?D?k??R@?D?k??R@!?D?k??R@*      ??!       2      ??!       :	?0&???*@?0&???*@!?0&???*@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?cit?,3@y??"?4T@?
"?
XModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block6a_dwconv/depthwiseDepthwiseConv2dNative????w??!????w??"-
IteratorGetNext/_4_Recv=6y???!N??S????"?
bModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_expand_bn/FusedBatchNormV3FusedBatchNormV3n??͍?!??|?gz??"n
UModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2b_se_excite/mulMulg????|??!*?S?L^??"n
PModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/stem_conv/Conv2DConv2D??????!?<<Zr2??0"n
UModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block1a_se_excite/mulMuldnR3ʁ??!T?֫????"v
]Model3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block2a_expand_activation/mulMul??+y4??!s(3u$???"?
xModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_rotation/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3???2??!???h??"?
tModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/augmentation_layer/random_zoom/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3?Jos
n??!??/3??"q
WModel3_FineTuned_10Layers_SixEpochsTotal_DataAug/efficientnetb0/block1a_se_squeeze/MeanMeannȯ?V:??!6xZ?W??IR???U?M@Q?YWk?D@Y?#??7@a^9?~DS@qv?G @y?ټp?p|?"?	
both?Your program is POTENTIALLY input-bound because 4.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?14.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 