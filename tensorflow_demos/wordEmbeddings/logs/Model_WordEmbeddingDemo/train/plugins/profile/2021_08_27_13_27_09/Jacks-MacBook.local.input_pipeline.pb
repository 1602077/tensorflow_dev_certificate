	9??v?OE@9??v?OE@!9??v?OE@	??q??:????q??:??!??q??:??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:9??v?OE@???Mb?@AH?z??A@Y???Mb??rEagerKernelExecute 0*	      F@2F
Iterator::Model?I+???!      I@)?I+???1      I@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?I+???!      9@)?I+???1      9@:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCache?I+???!      9@)y?&1?|?1?E]t?/@:Preprocessing2x
AIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl????Mbp?!/?袋."@)????Mbp?1/?袋."@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 15.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??q??:??I???Y??X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???Mb?@???Mb?@!???Mb?@      ?!       "      ?!       *      ?!       2	H?z??A@H?z??A@!H?z??A@:      ?!       B      ?!       J	???Mb?????Mb??!???Mb??R      ?!       Z	???Mb?????Mb??!???Mb??b      ?!       JCPU_ONLYY??q??:??b q???Y??X@