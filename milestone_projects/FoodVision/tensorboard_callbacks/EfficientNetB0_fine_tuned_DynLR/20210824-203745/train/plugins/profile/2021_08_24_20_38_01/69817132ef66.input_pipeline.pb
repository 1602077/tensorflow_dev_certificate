	ףp=
-e@ףp=
-e@!ףp=
-e@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'ףp=
-e@ <?Bus??1?XQ?i?^@I?T??F@r0*?|?5^n?@?|????A2?
RIterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2(Ou??p?X@!q4֌?@@)Ou??p?X@1q4֌?@@:Preprocessing2p
9Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2 6????W@!=???Og@@)6????W@1=???Og@@:Preprocessing2z
CIterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch ??7??&K@!?G0E??2@)??7??&K@1?G0E??2@:Preprocessing2X
!Iterator::Root::Prefetch::BatchV2?]?n`@!ý????F@)????B@1Ro?(@:Preprocessing2?
aIterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2(??ʆ5?@!?hJ?/??)??ʆ5?@1?hJ?/??:Preprocessing2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[3]::FlatMap[0]::TFRecord?!??T2??!??BU????)?!??T2??1??BU????:Advanced file read2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[4]::FlatMap[0]::TFRecordoH?'??!ة'?y??)oH?'??1ة'?y??:Advanced file read2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[5]::FlatMap[0]::TFRecord
?o?^}<??!?@A\???)?o?^}<??1?@A\???:Advanced file read2?
tIterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality(?Y,E????!ξNc?a??)e?????1?fw??t??:Preprocessing2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4(??L?*???!&?N??)??L?*???1&?N??:Preprocessing2a
*Iterator::Root::Prefetch::BatchV2::Shuffle ?Ŧ?B?W@!B6??jq@@)КiQ??1O?&?5??:Preprocessing2O
Iterator::Root::PrefetchV)=?K???!v?a??1??)V)=?K???1v?a??1??:Preprocessing2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[3]::FlatMapt(CUL%??!4z???e??)?j?v/??1A[?}??:Preprocessing2E
Iterator::Root?E}?;l??!:???w???)??z?V???1?d??N??:Preprocessing2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[4]::FlatMap(-\Va???!???x???)SZK ???1???eΈ?:Preprocessing2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[5]::FlatMap
.u?׃I??!???????)?) ?3h??1s??0鞆?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?27.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?%vOO?;@Q?v",,R@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	 <?Bus?? <?Bus??! <?Bus??      ??!       "	?XQ?i?^@?XQ?i?^@!?XQ?i?^@*      ??!       2      ??!       :	?T??F@?T??F@!?T??F@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?%vOO?;@y?v",,R@