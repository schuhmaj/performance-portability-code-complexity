"""Tests for :mod:`ppbcc.code_complexity.detection`."""

from pathlib import Path

import pytest

from ppbcc.code_complexity.config import load_dialects
from ppbcc.code_complexity.detection import detect_dialects, extract_includes


@pytest.fixture(scope="module")
def registry():
    return load_dialects()


def detected_names(code: str, path: str | None, registry) -> list[str]:
    """Runs the detection and returns the dialect names."""
    return [
        spec.name
        for spec in detect_dialects(code, Path(path) if path else None, registry)
    ]


class TestExtractIncludes:
    def test_angle_and_quoted(self):
        code = '#include <vector>\n#include "my/header.h"\n  #  include <omp.h>\n'
        assert extract_includes(code) == ["vector", "my/header.h", "omp.h"]


class TestDetection:
    def test_plain_cpp_detects_nothing(self, registry):
        code = "#include <vector>\nint main() { return 0; }\n"
        assert detected_names(code, "main.cpp", registry) == []

    def test_kokkos_via_header(self, registry):
        code = "#include <Kokkos_Core.hpp>\n"
        assert detected_names(code, "impl.cpp", registry) == ["kokkos"]

    def test_kokkos_via_namespace(self, registry):
        code = "void f() { Kokkos::fence(); }\n"
        assert detected_names(code, "impl.cpp", registry) == ["kokkos"]

    def test_cuda_via_extension(self, registry):
        assert "cuda" in detected_names("__global__ void k() {}\n", "kernel.cu", registry)

    def test_openmp_via_pragma(self, registry):
        code = "#pragma omp parallel for\nfor(;;) {}\n"
        assert detected_names(code, "impl.cpp", registry) == ["openmp"]

    def test_sycl_via_header_and_namespace(self, registry):
        code = "#include <sycl/sycl.hpp>\nsycl::queue q;\n"
        assert detected_names(code, "impl.cpp", registry) == ["sycl"]

    def test_boost_compute_via_namespace_alias(self, registry):
        code = '#include "boost/compute.hpp"\nnamespace bc = boost::compute;\n'
        assert "boost_compute" in detected_names(code, "impl.cpp", registry)

    def test_opencl_kernel_file_via_extension(self, registry):
        code = "__kernel void add(__global float* a) {}\n"
        assert "opencl" in detected_names(code, "add.cl", registry)

    def test_opencl_host_code_via_pattern_hits(self, registry):
        code = "cl_int err; cl_mem buf = clCreateBuffer(c, CL_MEM_READ_ONLY, s, p, &err);\n"
        assert "opencl" in detected_names(code, "host.cpp", registry)

    def test_hip_file_does_not_also_detect_cuda(self, registry):
        code = (
            "#include <hip/hip_runtime.h>\n"
            "__global__ void k() { int i = threadIdx.x + blockIdx.x * blockDim.x; }\n"
        )
        assert detected_names(code, "impl.hip", registry) == ["hip"]

    def test_shared_gpu_markers_alone_still_detect_cuda(self, registry):
        code = "__host__ __device__ int f(); __host__ __device__ int g();\n"
        assert detected_names(code, "defs.h", registry) == ["cuda"]

    def test_pcuda_via_header_suppresses_cuda(self, registry):
        code = (
            "#include <pcuda.hpp>\n"
            "__global__ void k() { int i = threadIdx.x + blockIdx.x * blockDim.x; }\n"
            "void run() { pcudaLaunchKernelGGL(k, grid, block, 0, 0); }\n"
        )
        assert detected_names(code, "impl.cpp", registry) == ["pcuda"]

    def test_pcuda_via_native_api_calls(self, registry):
        code = "pcudaMalloc(&p, n); pcudaMemcpy(d, s, n); pcudaFree(p);\n"
        assert "pcuda" in detected_names(code, "impl.cpp", registry)

    def test_multiple_dialects(self, registry):
        code = "#include <Kokkos_Core.hpp>\n#pragma omp parallel\n{}\n"
        assert detected_names(code, "impl.cpp", registry) == ["openmp", "kokkos"]

    def test_glsl_shader_via_extension(self, registry):
        code = "layout(std430, binding = 0) buffer B { float a[]; };\n"
        assert "glsl" in detected_names(code, "shader.comp", registry)

    def test_wgsl_via_extension(self, registry):
        code = "@group(0) @binding(0) var<storage> data: array<f32>;\n"
        assert "wgsl" in detected_names(code, "kernel.wgsl", registry)
