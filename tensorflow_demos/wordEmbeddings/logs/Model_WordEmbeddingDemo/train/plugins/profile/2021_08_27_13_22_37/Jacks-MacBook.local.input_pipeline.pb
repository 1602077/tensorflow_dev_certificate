	?E????G@?E????G@!?E????G@	-Ei]KQ??-Ei]KQ??!-Ei]KQ??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?E????G@???(\?"@A{?G?C@YJ+???rEagerKernelExecute 0*	     ?B@2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?? ?rh??!?Ϻ??F@)?? ?rh??1?Ϻ??F@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism????????!E>?S?P@)????Mb??1?Y7?"?5@:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCache;?O??n??!1E>?S8@){?G?zt?1o0E>?+@:Preprocessing2x
AIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl????Mbp?!?Y7?"?%@)????Mbp?1?Y7?"?%@:Preprocessing2F
Iterator::Modely?&1???!?n0E>?R@)?~j?t?h?1v?)?Y7 @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 19.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9.Ei]KQ??I??(?k?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???(\?"@???(\?"@!???(\?"@      ?!       "      ?!       *      ?!       2	{?G?C@{?G?C@!{?G?C@:      ?!       B      ?!       J	J+???J+???!J+???R      ?!       Z	J+???J+???!J+???b      ?!       JCPU_ONLYY.Ei]KQ??b q??(?k?X@