# v2 (HostJIT) backend — AoT serialize/deserialize/load not supported.
# Included at the end of _bindings_impl.pyx; provides stub functions so that
# the class methods exist but raise a clear error if called.

_AOT_NOT_SUPPORTED_MSG = (
    "AoT serialize/deserialize is not supported with the HostJIT (v2) backend."
)

def _aot_reduce_serialize(self):              raise NotImplementedError(_AOT_NOT_SUPPORTED_MSG)
def _aot_reduce_deserialize(blob):            raise NotImplementedError(_AOT_NOT_SUPPORTED_MSG)
def _aot_scan_serialize(self):                raise NotImplementedError(_AOT_NOT_SUPPORTED_MSG)
def _aot_scan_deserialize(blob):              raise NotImplementedError(_AOT_NOT_SUPPORTED_MSG)
def _aot_segmented_reduce_serialize(self):    raise NotImplementedError(_AOT_NOT_SUPPORTED_MSG)
def _aot_segmented_reduce_deserialize(blob):  raise NotImplementedError(_AOT_NOT_SUPPORTED_MSG)
def _aot_merge_sort_serialize(self):          raise NotImplementedError(_AOT_NOT_SUPPORTED_MSG)
def _aot_merge_sort_deserialize(blob):        raise NotImplementedError(_AOT_NOT_SUPPORTED_MSG)
def _aot_unique_by_key_serialize(self):       raise NotImplementedError(_AOT_NOT_SUPPORTED_MSG)
def _aot_unique_by_key_deserialize(blob):     raise NotImplementedError(_AOT_NOT_SUPPORTED_MSG)
def _aot_radix_sort_serialize(self):          raise NotImplementedError(_AOT_NOT_SUPPORTED_MSG)
def _aot_radix_sort_deserialize(blob):        raise NotImplementedError(_AOT_NOT_SUPPORTED_MSG)
def _aot_unary_transform_serialize(self):     raise NotImplementedError(_AOT_NOT_SUPPORTED_MSG)
def _aot_unary_transform_deserialize(blob):   raise NotImplementedError(_AOT_NOT_SUPPORTED_MSG)
def _aot_binary_transform_serialize(self):    raise NotImplementedError(_AOT_NOT_SUPPORTED_MSG)
def _aot_binary_transform_deserialize(blob):  raise NotImplementedError(_AOT_NOT_SUPPORTED_MSG)
def _aot_histogram_serialize(self):           raise NotImplementedError(_AOT_NOT_SUPPORTED_MSG)
def _aot_histogram_deserialize(blob):         raise NotImplementedError(_AOT_NOT_SUPPORTED_MSG)
def _aot_binary_search_serialize(self):       raise NotImplementedError(_AOT_NOT_SUPPORTED_MSG)
def _aot_binary_search_deserialize(blob):     raise NotImplementedError(_AOT_NOT_SUPPORTED_MSG)
def _aot_three_way_partition_serialize(self): raise NotImplementedError(_AOT_NOT_SUPPORTED_MSG)
def _aot_three_way_partition_deserialize(blob): raise NotImplementedError(_AOT_NOT_SUPPORTED_MSG)
def _aot_segmented_sort_serialize(self):      raise NotImplementedError(_AOT_NOT_SUPPORTED_MSG)
def _aot_segmented_sort_deserialize(blob):    raise NotImplementedError(_AOT_NOT_SUPPORTED_MSG)
