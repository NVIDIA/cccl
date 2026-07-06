# v2 (HostJIT) backend: serialize/deserialize/load not supported.
# Included at the end of _bindings_impl.pyx; provides stub functions so that
# the class methods exist but raise a clear error if called.

_SERIALIZATION_NOT_SUPPORTED_MSG = (
    "serialize/deserialize is not supported with the HostJIT (v2) backend."
)

def _serialization_reduce_serialize(self):              raise NotImplementedError(_SERIALIZATION_NOT_SUPPORTED_MSG)
def _serialization_reduce_deserialize(blob):            raise NotImplementedError(_SERIALIZATION_NOT_SUPPORTED_MSG)
def _serialization_scan_serialize(self):                raise NotImplementedError(_SERIALIZATION_NOT_SUPPORTED_MSG)
def _serialization_scan_deserialize(blob):              raise NotImplementedError(_SERIALIZATION_NOT_SUPPORTED_MSG)
def _serialization_segmented_reduce_serialize(self):    raise NotImplementedError(_SERIALIZATION_NOT_SUPPORTED_MSG)
def _serialization_segmented_reduce_deserialize(blob):  raise NotImplementedError(_SERIALIZATION_NOT_SUPPORTED_MSG)
def _serialization_merge_sort_serialize(self):          raise NotImplementedError(_SERIALIZATION_NOT_SUPPORTED_MSG)
def _serialization_merge_sort_deserialize(blob):        raise NotImplementedError(_SERIALIZATION_NOT_SUPPORTED_MSG)
def _serialization_unique_by_key_serialize(self):       raise NotImplementedError(_SERIALIZATION_NOT_SUPPORTED_MSG)
def _serialization_unique_by_key_deserialize(blob):     raise NotImplementedError(_SERIALIZATION_NOT_SUPPORTED_MSG)
def _serialization_radix_sort_serialize(self):          raise NotImplementedError(_SERIALIZATION_NOT_SUPPORTED_MSG)
def _serialization_radix_sort_deserialize(blob):        raise NotImplementedError(_SERIALIZATION_NOT_SUPPORTED_MSG)
def _serialization_unary_transform_serialize(self):     raise NotImplementedError(_SERIALIZATION_NOT_SUPPORTED_MSG)
def _serialization_unary_transform_deserialize(blob):   raise NotImplementedError(_SERIALIZATION_NOT_SUPPORTED_MSG)
def _serialization_binary_transform_serialize(self):    raise NotImplementedError(_SERIALIZATION_NOT_SUPPORTED_MSG)
def _serialization_binary_transform_deserialize(blob):  raise NotImplementedError(_SERIALIZATION_NOT_SUPPORTED_MSG)
def _serialization_histogram_serialize(self):           raise NotImplementedError(_SERIALIZATION_NOT_SUPPORTED_MSG)
def _serialization_histogram_deserialize(blob):         raise NotImplementedError(_SERIALIZATION_NOT_SUPPORTED_MSG)
def _serialization_binary_search_serialize(self):       raise NotImplementedError(_SERIALIZATION_NOT_SUPPORTED_MSG)
def _serialization_binary_search_deserialize(blob):     raise NotImplementedError(_SERIALIZATION_NOT_SUPPORTED_MSG)
def _serialization_three_way_partition_serialize(self): raise NotImplementedError(_SERIALIZATION_NOT_SUPPORTED_MSG)
def _serialization_three_way_partition_deserialize(blob): raise NotImplementedError(_SERIALIZATION_NOT_SUPPORTED_MSG)
def _serialization_segmented_sort_serialize(self):      raise NotImplementedError(_SERIALIZATION_NOT_SUPPORTED_MSG)
def _serialization_segmented_sort_deserialize(blob):    raise NotImplementedError(_SERIALIZATION_NOT_SUPPORTED_MSG)
