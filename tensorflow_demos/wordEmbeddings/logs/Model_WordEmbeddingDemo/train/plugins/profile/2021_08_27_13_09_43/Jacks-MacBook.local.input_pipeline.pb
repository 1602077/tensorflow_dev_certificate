	j?t??E@j?t??E@!j?t??E@	?? ??u???? ??u??!?? ??u??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:j?t??E@7?A`??@Ad;?O??B@Y???S㥫?rEagerKernelExecute 0*	      A@2F
Iterator::Model;?O??n??!yxxxxxJ@);?O??n??1yxxxxxJ@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch????Mb??!??????7@)????Mb??1??????7@:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCache????Mb??!??????7@)????Mbp?1??????'@:Preprocessing2x
AIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl????Mbp?!??????'@)????Mbp?1??????'@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 15.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?? ??u??I?w?"?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	7?A`??@7?A`??@!7?A`??@      ?!       "      ?!       *      ?!       2	d;?O??B@d;?O??B@!d;?O??B@:      ?!       B      ?!       J	???S㥫????S㥫?!???S㥫?R      ?!       Z	???S㥫????S㥫?!???S㥫?b      ?!       JCPU_ONLYY?? ??u??b q?w?"?X@