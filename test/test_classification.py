"""Tests for :mod:`code_complexity.classification`."""

import pytest

from code_complexity.classification import (
    TokenClassifier,
    TokenCounts,
    extract_namespace_aliases,
)
from code_complexity.config import load_cpp_keywords, load_dialects
from code_complexity.tokenizer import tokenize


@pytest.fixture(scope="module")
def keywords():
    return load_cpp_keywords()


@pytest.fixture(scope="module")
def registry():
    return load_dialects()


def classify(code: str, keywords, dialects) -> TokenCounts:
    """Tokenises and classifies a snippet with the given dialects."""
    classifier = TokenClassifier(keywords, dialects, extract_namespace_aliases(code))
    return classifier.classify(tokenize(code))


class TestBaselineCpp:
    def test_simple_function(self, keywords):
        counts = classify("int main() { return 0; }", keywords, [])
        assert counts.operators == {"int": 1, "(": 1, "{": 1, "return": 1, ";": 1}
        assert counts.operands == {"main": 1, "0": 1}
        assert not counts.dialect_operators
        assert not counts.dialect_operands

    def test_closing_brackets_not_counted(self, keywords):
        counts = classify("f(a[i]);", keywords, [])
        for closer in (")", "]", "}"):
            assert closer not in counts.operators

    def test_operand_keywords(self, keywords):
        counts = classify("bool b = true; auto p = nullptr;", keywords, [])
        assert counts.operands["true"] == 1
        assert counts.operands["nullptr"] == 1
        assert counts.operators["bool"] == 1

    def test_qualified_name_counts_scope_operators(self, keywords):
        counts = classify("std::vector<int> v;", keywords, [])
        assert counts.operators["::"] == 1
        assert counts.operands["std"] == 1
        assert counts.operands["vector"] == 1

    def test_literals_are_operands(self, keywords):
        counts = classify('log("msg", 42, \'c\');', keywords, [])
        assert counts.operands['"msg"'] == 1
        assert counts.operands["42"] == 1
        assert counts.operands["'c'"] == 1

    def test_include_directive(self, keywords):
        counts = classify("#include <vector>\n", keywords, [])
        assert counts.operators["#include"] == 1
        assert counts.operands['"vector"'] == 1

    def test_pragma_once_is_not_dialect(self, keywords):
        counts = classify("#pragma once\n", keywords, [])
        assert counts.operators["#pragma"] == 1
        assert not counts.dialect_operators


class TestKokkos:
    def test_qualified_kokkos_call_is_one_operator(self, keywords, registry):
        code = 'Kokkos::parallel_for("loop", n, KOKKOS_LAMBDA(const int i) {});'
        counts = classify(code, keywords, [registry.resolve("kokkos")])
        assert counts.dialect_operators["Kokkos::parallel_for"] == 1
        assert counts.dialect_operators["KOKKOS_LAMBDA"] == 1
        # The C++ parts stay baseline:
        assert counts.operators["const"] == 1
        assert counts.operands["i"] == 1

    def test_deep_qualified_name(self, keywords, registry):
        counts = classify(
            "Kokkos::View<double*, Kokkos::LayoutRight> v;",
            keywords,
            [registry.resolve("kokkos")],
        )
        assert counts.dialect_operators["Kokkos::View"] == 1
        assert counts.dialect_operators["Kokkos::LayoutRight"] == 1

    def test_without_dialect_kokkos_is_plain_cpp(self, keywords):
        counts = classify("Kokkos::deep_copy(a, b);", keywords, [])
        assert not counts.dialect_operators
        assert counts.operands["Kokkos"] == 1
        assert counts.operators["::"] == 1


class TestNamespaceAliases:
    def test_alias_extraction(self):
        aliases = extract_namespace_aliases("namespace bc = boost::compute;\n")
        assert aliases == {"bc": ("boost", "compute")}

    def test_alias_resolved_to_dialect(self, keywords, registry):
        code = "namespace bc = boost::compute;\nbc::vector<float> v;"
        counts = classify(code, keywords, [registry.resolve("boost_compute")])
        assert counts.dialect_operators["boost::compute::vector"] == 1


class TestOpenMpPragmas:
    def test_pragma_clauses_are_dialect_operators(self, keywords, registry):
        code = "#pragma omp parallel for reduction(+ : sum)\nfor (int i = 0; i < n; ++i) sum += i;"
        counts = classify(code, keywords, [registry.resolve("openmp")])
        assert counts.dialect_operators["#pragma omp"] == 1
        assert counts.dialect_operators["parallel"] == 1
        assert counts.dialect_operators["for"] == 1
        assert counts.dialect_operators["reduction"] == 1
        # Variables referenced inside the pragma are dialect operands:
        assert counts.dialect_operands["sum"] == 1
        # ... while the loop itself stays baseline C++:
        assert counts.operators["for"] == 1
        assert counts.operands["sum"] == 1

    def test_omp_api_calls_are_dialect_operators(self, keywords, registry):
        counts = classify(
            "int t = omp_get_thread_num();", keywords, [registry.resolve("openmp")]
        )
        assert counts.dialect_operators["omp_get_thread_num"] == 1


class TestCudaAndHip:
    def test_kernel_launch_counts_once(self, keywords, registry):
        counts = classify(
            "compute<<<blocks, threads>>>(data);", keywords, [registry.resolve("cuda")]
        )
        assert counts.dialect_operators["<<<"] == 1
        assert ">>>" not in counts.dialect_operators

    def test_launch_brackets_split_without_cuda(self, keywords):
        counts = classify("compute<<<blocks, threads>>>(data);", keywords, [])
        assert counts.operators["<<"] == 1
        assert counts.operators["<"] == 1
        assert counts.operators[">>"] == 1
        assert counts.operators[">"] == 1

    def test_cuda_builtins_and_api(self, keywords, registry):
        code = "__global__ void k() { int i = threadIdx.x; } cudaMalloc(&p, n);"
        counts = classify(code, keywords, [registry.resolve("cuda")])
        assert counts.dialect_operators["__global__"] == 1
        assert counts.dialect_operators["threadIdx"] == 1
        assert counts.dialect_operators["cudaMalloc"] == 1
        assert counts.operands["x"] == 1

    def test_pcuda_native_and_compat_api(self, keywords, registry):
        code = (
            "pcudaError_t e = pcudaMalloc(&p, n);\n"
            "pcudaParallelFor(grid, block, [=]() { int i = threadIdx.x; });\n"
            "cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost);\n"
            "k<<<grid, block>>>(x);\n"
        )
        counts = classify(code, keywords, [registry.resolve("pcuda")])
        assert counts.dialect_operators["pcudaError_t"] == 1
        assert counts.dialect_operators["pcudaMalloc"] == 1
        assert counts.dialect_operators["pcudaParallelFor"] == 1
        assert counts.dialect_operators["threadIdx"] == 1
        # The cuda*/hip* compatibility APIs belong to pCUDA as well:
        assert counts.dialect_operators["cudaMemcpy"] == 1
        assert counts.dialect_operators["<<<"] == 1

    def test_pcuda_device_branch_builtins(self, keywords, registry):
        counts = classify(
            "if (__acpp_sscp_is_device) { work(); }", keywords, [registry.resolve("pcuda")]
        )
        assert counts.dialect_operators["__acpp_sscp_is_device"] == 1

    def test_hip_api(self, keywords, registry):
        counts = classify("hipMemcpy(dst, src, n, hipMemcpyHostToDevice);",
                          keywords, [registry.resolve("hip")])
        assert counts.dialect_operators["hipMemcpy"] == 1
        assert counts.dialect_operators["hipMemcpyHostToDevice"] == 1


class TestOtherDialects:
    def test_opencl_identifiers(self, keywords, registry):
        code = "cl_int err; cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sz, p, &err);"
        counts = classify(code, keywords, [registry.resolve("opencl")])
        assert counts.dialect_operators["cl_int"] == 1
        assert counts.dialect_operators["cl_mem"] == 1
        assert counts.dialect_operators["clCreateBuffer"] == 1
        assert counts.dialect_operators["CL_MEM_READ_ONLY"] == 1

    def test_sycl_namespace(self, keywords, registry):
        counts = classify(
            "sycl::queue q{sycl::default_selector_v};", keywords, [registry.resolve("sycl")]
        )
        assert counts.dialect_operators["sycl::queue"] == 1
        assert counts.dialect_operators["sycl::default_selector_v"] == 1

    def test_full_counters_merge_baseline_and_dialect(self, keywords, registry):
        counts = classify("Kokkos::fence(); int a = 1;", keywords, [registry.resolve("kokkos")])
        assert counts.full_operators["Kokkos::fence"] == 1
        assert counts.full_operators["int"] == 1
        assert counts.full_operands["a"] == 1
