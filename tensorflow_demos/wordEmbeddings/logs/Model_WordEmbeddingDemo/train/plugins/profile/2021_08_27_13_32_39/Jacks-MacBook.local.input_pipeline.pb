	?&1?E@?&1?E@!?&1?E@	??Kn??????Kn????!??Kn????"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?&1?E@??"??~@A?????A@YD?l?????rEagerKernelExecute 0*	     ?@@2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?~j?t???!/?袋.B@)?~j?t???1/?袋.B@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??~j?t??!?&?l??L@)y?&1?|?16?d?M65@:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCache?I+???!??????@@)?~j?t?x?1/?袋.2@:Preprocessing2x
AIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl{?G?zt?!N6?d?M.@){?G?zt?1N6?d?M.@:Preprocessing2F
Iterator::Model?I+???!??????P@)?~j?t?h?1/?袋."@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 15.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??Kn????I?H???X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??"??~@??"??~@!??"??~@      ?!       "      ?!       *      ?!       2	?????A@?????A@!?????A@:      ?!       B      ?!       J	D?l?????D?l?????!D?l?????R      ?!       Z	D?l?????D?l?????!D?l?????b      ?!       JCPU_ONLYY??Kn????b q?H???X@