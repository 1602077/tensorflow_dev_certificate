	!?rh?E@!?rh?E@!!?rh?E@	3?i??S??3?i??S??!3?i??S??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:!?rh?E@??(\??@AˡE???A@Y?v??/??rEagerKernelExecute 0*	     ?J@2F
Iterator::Model9??v????!M0??>?H@)9??v????1M0??>?H@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???Q???!???sHM<@)???Q???1???sHM<@:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCache?~j?t???!??V?9?6@);?O??n??1"5?x+?0@:Preprocessing2x
AIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl?~j?t?h?!??V?9?@)?~j?t?h?1??V?9?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 15.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no93?i??S??I KV?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??(\??@??(\??@!??(\??@      ?!       "      ?!       *      ?!       2	ˡE???A@ˡE???A@!ˡE???A@:      ?!       B      ?!       J	?v??/???v??/??!?v??/??R      ?!       Z	?v??/???v??/??!?v??/??b      ?!       JCPU_ONLYY3?i??S??b q KV?X@