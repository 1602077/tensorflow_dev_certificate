	?e?-?d@?e?-?d@!?e?-?d@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?e?-?d@e?,?i??1?,D??_@I
1?Tm3C@r0*^?I-?@???x?	A2?
RIterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2(}??O9XV@!??&?DE@)}??O9XV@1??&?DE@:Preprocessing2p
9Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2 ??9???M@!S??i<@)??9???M@1S??i<@:Preprocessing2X
!Iterator::Root::Prefetch::BatchV2xF[?D?U@!3?½?D@)<??ؖ?;@1?&?^?z*@:Preprocessing2z
CIterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch ? ?b?:@!?^{?	?)@)? ?b?:@1?^{?	?)@:Preprocessing2?
aIterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2(]??ʾ?@!??-?? @)]??ʾ?@1??-?? @:Preprocessing2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[15]::FlatMap[0]::TFRecordGr?????!B?0џ??)Gr?????1B?0џ??:Advanced file read2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4(???????!(əN???)???????1(əN???:Preprocessing2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[0]::FlatMap[0]::TFRecord-^,????!??%?????)-^,????1??%?????:Advanced file read2?
tIterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality(k(?????!Y?B? ??)?ʡE????1!???
??:Preprocessing2a
*Iterator::Root::Prefetch::BatchV2::Shuffle R<???M@!O??,?<@)???????1#?::?[??:Preprocessing2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[14]::FlatMap[0]::TFRecord9??????!/??D	??)9??????1/??D	??:Advanced file read2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[1]::FlatMap[0]::TFRecord?????Ը?!?Xg???)?????Ը?1?Xg???:Advanced file read2O
Iterator::Root::Prefetch?L?T???! t????)?L?T???1 t????:Preprocessing2E
Iterator::Root_???:T??!??[?J??)6?u?!??1?xCf????:Preprocessing2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[14]::FlatMap?Xl?????!`?j????)֪]???1?L??G/??:Preprocessing2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[0]::FlatMap??~? ??!0#?F???)Y???j??1???ғ??:Preprocessing2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[15]::FlatMap???V`???!?2?Ֆ??)k?C4????1?9??=p??:Preprocessing2?
?Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Prefetch::ParallelMapV2::ParallelMapV2::AssertCardinality::ParallelInterleaveV4[1]::FlatMap+Kt?Y???!??l=??)3?68?z?1X???S?i?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?23.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIh:?$?"8@Qfq?6^?R@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	e?,?i??e?,?i??!e?,?i??      ??!       "	?,D??_@?,D??_@!?,D??_@*      ??!       2      ??!       :	
1?Tm3C@
1?Tm3C@!
1?Tm3C@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qh:?$?"8@yfq?6^?R@