# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Mock tests to verify GDB and LLDB pretty-printer logic for cuda::std::optional."""

import sys
from pathlib import Path
import unittest

# Setup path so we can import cccl_common, optional, etc.
SCRIPT_DIR = Path(__file__).resolve().parent
GDB_DIR = SCRIPT_DIR.parent.parent / "share" / "libcudacxx" / "gdb"
LLDB_DIR = SCRIPT_DIR.parent.parent / "share" / "libcudacxx" / "lldb"

# --- Mock GDB Module ---
class MockGdb:
    TYPE_CODE_REF = 2
    TYPE_CODE_RVALUE_REF = 3

    class Type:
        def __init__(self, name, code=1):
            self.name = name
            self.code = code

        def __str__(self):
            return self.name

        def strip_typedefs(self):
            return self

        def unqualified(self):
            return self

        def fields(self):
            return []

    class Value:
        def __init__(self, type_name, engaged, val=None):
            self.type = MockGdb.Type(type_name)
            self._engaged = engaged
            self._val = val

        def __getitem__(self, key):
            if key == "__engaged_":
                return MockGdb.Value("bool", self._engaged)
            if key == "__storage_":
                return MockGdb.Value("storage", self._engaged, self._val)
            if key == "__val_":
                return self._val
            raise KeyError(key)

        def __bool__(self):
            return bool(self._engaged)

    class printing:
        class PrettyPrinter:
            def __init__(self, name):
                self.name = name

# Inject Mock GDB into sys.modules before importing gdb printers
sys.modules["gdb"] = MockGdb
sys.modules["gdb.printing"] = MockGdb.printing

# Load GDB printer
if str(GDB_DIR) not in sys.path:
    sys.path.insert(0, str(GDB_DIR))

import cccl_common as gdb_cccl_common
import optional as gdb_optional

# Clean up GDB import path and sys.modules for LLDB testing
sys.path.remove(str(GDB_DIR))
del sys.modules["cccl_common"]
del sys.modules["optional"]

# --- Mock LLDB Module ---
class MockLldb:
    class SBType:
        def __init__(self, name, code=1):
            self.name = name
            self.code = code

        def GetCanonicalType(self):
            return self

        def GetDereferencedType(self):
            return self

        def GetUnqualifiedType(self):
            return self

        def GetDisplayTypeName(self):
            return self.name

        def GetName(self):
            return self.name

        def IsReferenceType(self):
            return False

    class SBValue:
        def __init__(self, name, type_name, engaged, val=None):
            self.name = name
            self.type = MockLldb.SBType(type_name)
            self.engaged = engaged
            self.val = val

        def GetType(self):
            return self.type

        def GetNonSyntheticValue(self):
            return self

        def GetChildMemberWithName(self, name):
            if name == "__engaged_":
                return MockLldb.SBValue("__engaged_", "bool", self.engaged)
            if name == "__storage_":
                return MockLldb.SBValue("__storage_", "storage", self.engaged, self.val)
            if name == "__val_":
                return self.val
            return MockLldb.SBValue("invalid", "void", False)

        def GetValueAsUnsigned(self, default_val=0):
            if self.name == "__engaged_":
                return 1 if self.engaged else 0
            return default_val

        def IsValid(self):
            return self.name != "invalid"

        def Clone(self, new_name):
            return MockLldb.SBValue(new_name, self.type.name, self.engaged, self.val)

# Inject Mock LLDB into sys.modules before importing lldb printers
sys.modules["lldb"] = MockLldb

# Load LLDB printer
if str(LLDB_DIR) not in sys.path:
    sys.path.insert(0, str(LLDB_DIR))

import cccl_common as lldb_cccl_common
import optional as lldb_optional


class TestOptionalPrettyPrinters(unittest.TestCase):

    def test_gdb_optional_disengaged(self):
        # Create a disengaged mock optional<int>
        mock_val = MockGdb.Value("cuda::std::optional<int>", engaged=False)
        self.assertTrue(gdb_optional._is_cuda_optional(mock_val.type))

        printer = gdb_optional.OptionalPrinter(mock_val)
        self.assertFalse(printer.engaged)
        self.assertEqual(printer.to_string(), "cuda::std::nullopt")
        
        # Check children list is empty
        children = list(printer.children())
        self.assertEqual(len(children), 0)

    def test_gdb_optional_engaged(self):
        # Create an engaged mock optional<int> holding 42
        inner_val = MockGdb.Value("int", engaged=True) # inner value doesn't need engaged check but we reuse Value
        inner_val._val = 42

        mock_val = MockGdb.Value("cuda::std::optional<int>", engaged=True, val=inner_val)
        self.assertTrue(gdb_optional._is_cuda_optional(mock_val.type))

        printer = gdb_optional.OptionalPrinter(mock_val)
        self.assertTrue(printer.engaged)
        self.assertEqual(printer.to_string(), "cuda::std::optional<int>")

        # Check children list contains the single value
        children = list(printer.children())
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0][0], "value")
        self.assertEqual(children[0][1]._val, 42)

    def test_lldb_optional_disengaged(self):
        # Create a disengaged mock optional<int>
        mock_val = MockLldb.SBValue("opt", "cuda::std::optional<int>", engaged=False)
        self.assertTrue(lldb_optional.is_cuda_optional(mock_val.GetType(), {}))

        # Check summary
        summary = lldb_optional.optional_summary(mock_val, {})
        self.assertEqual(summary, "cuda::std::nullopt")

        # Check synthetic provider
        provider = lldb_optional.OptionalSyntheticProvider(mock_val, {})
        self.assertFalse(provider.engaged)
        self.assertEqual(provider.num_children(), 0)
        self.assertFalse(provider.has_children())

    def test_lldb_optional_engaged(self):
        # Create an engaged mock optional<int> holding 100
        inner_val = MockLldb.SBValue("val", "int", engaged=True)
        inner_val.val = 100

        mock_val = MockLldb.SBValue("opt", "cuda::std::optional<int>", engaged=True, val=inner_val)
        self.assertTrue(lldb_optional.is_cuda_optional(mock_val.GetType(), {}))

        # Check summary is empty (expansion is handled by LLDB UI synthetic children)
        summary = lldb_optional.optional_summary(mock_val, {})
        self.assertEqual(summary, "")

        # Check synthetic provider
        provider = lldb_optional.OptionalSyntheticProvider(mock_val, {})
        self.assertTrue(provider.engaged)
        self.assertEqual(provider.num_children(), 1)
        self.assertTrue(provider.has_children())
        
        child = provider.get_child_at_index(0)
        self.assertIsNotNone(child)
        self.assertEqual(child.name, "value")
        self.assertEqual(child.val, 100)
        self.assertEqual(provider.get_type_name(), "cuda::std::optional<int>")

    def test_lldb_optional_abi_namespace(self):
        inner_val = MockLldb.SBValue("val", "int", engaged=True)
        mock_val = MockLldb.SBValue("opt", "cuda::std::__version_bump_ver4_::optional<int>", engaged=True, val=inner_val)
        provider = lldb_optional.OptionalSyntheticProvider(mock_val, {})
        self.assertEqual(provider.get_type_name(), "cuda::std::optional<int>")


    def test_gdb_optional_ref_specialization(self):
        class MockField:
            def __init__(self, name):
                self.name = name

        class RefType(MockGdb.Type):
            def fields(self):
                return [MockField("__value_")]

        inner_val = MockGdb.Value("int", engaged=True)
        inner_val._val = 42

        class MockPointer:
            def __init__(self, target):
                self.target = target
            def __int__(self):
                return 0x1234
            def dereference(self):
                return self.target

        mock_val = MockGdb.Value("cuda::std::optional<int&>", engaged=True, val=inner_val)
        mock_val.type = RefType("cuda::std::optional<int&>")

        original_getitem = mock_val.__class__.__getitem__
        def custom_getitem(s, key):
            if key == "__value_":
                return MockPointer(inner_val)
            return original_getitem(s, key)
        mock_val.__class__.__getitem__ = custom_getitem

        try:
            printer = gdb_optional.OptionalPrinter(mock_val)
            self.assertTrue(printer.engaged)
            self.assertEqual(printer.to_string(), "cuda::std::optional<int&>")
            children = list(printer.children())
            self.assertEqual(len(children), 1)
            self.assertEqual(children[0][0], "value")
            self.assertEqual(children[0][1]._val, 42)
        finally:
            mock_val.__class__.__getitem__ = original_getitem

    def test_lldb_optional_ref_specialization(self):
        inner_val = MockLldb.SBValue("val", "int", engaged=True)
        inner_val.val = 100

        class MockPointerValue(MockLldb.SBValue):
            def Dereference(self):
                return inner_val

        ptr_val = MockPointerValue("__value_", "int*", engaged=True)
        ptr_val.val = 0x1234
        ptr_val.GetValueAsUnsigned = lambda default_val=0: 0x1234

        mock_val = MockLldb.SBValue("opt", "cuda::std::optional<int&>", engaged=True)

        def custom_getchild(name):
            if name == "__value_":
                return ptr_val
            return MockLldb.SBValue("invalid", "void", False)
        mock_val.GetChildMemberWithName = custom_getchild

        provider = lldb_optional.OptionalSyntheticProvider(mock_val, {})
        self.assertTrue(provider.engaged)
        self.assertEqual(provider.num_children(), 1)
        self.assertTrue(provider.has_children())

        child = provider.get_child_at_index(0)
        self.assertIsNotNone(child)
        self.assertEqual(child.name, "value")
        self.assertEqual(child.val, 100)


if __name__ == "__main__":
    unittest.main()
