	?ɧǶ?R@?ɧǶ?R@!?ɧǶ?R@	JK~Ա?#@JK~Ա?#@!JK~Ա?#@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?ɧǶ?R@_???:?@1e??]??L@Iʊ?? ?@YA?+??@r0*?&1,@X9??A2?
RIterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2?oa?x?R@!*Q???B@)?oa?x?R@1*Q???B@:Preprocessing2p
9Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2 xD???&N@!N3?q?^>@)xD???&N@1N3?q?^>@:Preprocessing2X
!Iterator::Root::Prefetch::BatchV2?e?s~QT@!?#wD@)?_YiR?4@1G?a??$@:Preprocessing2z
CIterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch ??x)3@!?.??L#@)??x)3@1?.??L#@:Preprocessing2?
aIterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2@KW???$@!???+J@)@KW???$@1???+J@:Preprocessing2E
Iterator::RootD??%@!?: ?7@);m??a@1?~???@:Preprocessing2O
Iterator::Root::PrefetchM?x$^?@!?j??@)M?x$^?@1?j??@:Preprocessing2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[5]::FlatMap??S9????!???????)??ME*???1F??????:Preprocessing2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[5]::FlatMap[0]::TFRecord^?Y-????!?T?4{???)^?Y-????1?T?4{???:Advanced file read2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[6]::FlatMap[0]::TFRecord??ߖ??!KAH?u???)??ߖ??1KAH?u???:Advanced file read2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4?O??n??!?-&?R;??)?O??n??1?-&?R;??:Preprocessing2a
*Iterator::Root::Prefetch::BatchV2::Shuffle ????KN@!m2????>@)͓k
dv??1H??s????:Preprocessing2?
tIterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality????_???!?rDҎ
??)?j{????1Pn?f????:Preprocessing2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[6]::FlatMapy?	?5???!{&'이??)?.o?j??1??ܻ???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.moderate"?8.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2s4.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9JK~Ա?#@I@??:?)@Q/f???JS@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	_???:?@_???:?@!_???:?@      ??!       "	e??]??L@e??]??L@!e??]??L@*      ??!       2      ??!       :	ʊ?? ?@ʊ?? ?@!ʊ?? ?@B      ??!       J	A?+??@A?+??@!A?+??@R      ??!       Z	A?+??@A?+??@!A?+??@b      ??!       JGPUYJK~Ա?#@b q@??:?)@y/f???JS@