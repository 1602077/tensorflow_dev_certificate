?'	?R	OrV@?R	OrV@!?R	OrV@	z?r??z?r??!z?r??"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?R	OrV@?Q??{??1ک??``O@ID? ?8@Y??"?-???r0*NbX9&?@@5^??)A2p
9Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2 ??cꮡR@!1{\tF@)??cꮡR@11{\tF@:Preprocessing2?
RIterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2 Q??r?Q@!z??Ж?D@)Q??r?Q@1z??Ж?D@:Preprocessing2X
!Iterator::Root::Prefetch::BatchV2?c*?!W@!??Q??K@)????1@1??? h%@:Preprocessing2?
aIterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2 ?'??98@!???g @)?'??98@1???g @:Preprocessing2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[4]::FlatMap[0]::TFRecord{??????!
???zA??){??????1
???zA??:Advanced file read2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4 ץF?g???!??qE5??)ץF?g???1??qE5??:Preprocessing2?
tIterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality ?ui????!u_q?ߔ??)?E??????1??p????:Preprocessing2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[5]::FlatMap[0]::TFRecord
I?Ǵ6???!Dt?F??)I?Ǵ6???1Dt?F??:Advanced file read2a
*Iterator::Root::Prefetch::BatchV2::Shuffle ?/?1"?R@!??????F@)??>????1?ԜPٞ??:Preprocessing2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[3]::FlatMap[0]::TFRecordj???'??!T6U䲬??)j???'??1T6U䲬??:Advanced file read2z
CIterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch ?e?????!?i[?T??)?e?????1?i[?T??:Preprocessing2O
Iterator::Root::PrefetchsHj?dr??!?B???ߟ?)sHj?dr??1?B???ߟ?:Preprocessing2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[3]::FlatMap?	?????!Eñ?-1??)ѓ2????1?3r????:Preprocessing2E
Iterator::Root??3?????!񿜡-l??)???d???1!=??????:Preprocessing2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[4]::FlatMap???kzP??!t?U?u??)?q?߅???1???N??:Preprocessing2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[5]::FlatMap
??i?:??!?3? 3???)?4`??i??1?????Ή?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?27.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9z?r??I?1?'??=@QU?Z?yQ@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?Q??{???Q??{??!?Q??{??      ??!       "	ک??``O@ک??``O@!ک??``O@*      ??!       2      ??!       :	D? ?8@D? ?8@!D? ?8@B      ??!       J	??"?-?????"?-???!??"?-???R      ??!       Z	??"?-?????"?-???!??"?-???b      ??!       JGPUYz?r??b q?1?'??=@yU?Z?yQ@?"{
PModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block6a_dwconv/depthwiseDepthwiseConv2dNative??`hG ??!??`hG ??"-
IteratorGetNext/_1_Send???K???!`_ aZ???"?
ZModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_expand_bn/FusedBatchNormV3FusedBatchNormV3?NC?s??!L?B????"f
MModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2b_se_excite/mulMulSCP?ܐ?!????e???"f
HModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/stem_conv/Conv2DConv2D?K㩇K??!???¥??0"f
MModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block1a_se_excite/mulMulK??~????!?/Л7???"n
UModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_expand_activation/mulMul?$??Y???!Ƹ]k{??"i
OModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block1a_se_squeeze/MeanMean?:??=S??!??Y?=??"p
RModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block2a_expand_conv/Conv2DConv2D1??M???!??a|????0"{
PModel0_FineTuned_10Layers_SixEpochsTotal/efficientnetb0/block3b_dwconv/depthwiseDepthwiseConv2dNativeN?0???!??
Ӫ??IeUK>??P@Q5Ui?-d@@Y???B?7@a?;@oS@q??]ٍ@ynJ?׈??"?	
device?Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?27.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 