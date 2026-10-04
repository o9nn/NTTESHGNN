// F32 Torch7 nn.Linear QKV -> NTTESHGNN witness -> real GGML CPU graph.
// Projection-level experiment only: not a GGUF loader or a decoder KV cache.
#include "nt_bridge.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static std::map<std::string, std::string> read_manifest(const std::string & path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("missing witness.txt");
    std::map<std::string, std::string> m;
    std::string line;
    while (std::getline(in, line)) {
        auto pos = line.find('=');
        if (pos == std::string::npos || !pos || pos + 1 == line.size() ||
            !m.emplace(line.substr(0, pos), line.substr(pos + 1)).second)
            throw std::runtime_error("invalid or duplicate witness field");
    }
    return m;
}

static int dimension(const std::map<std::string, std::string> & m,
                     const std::string & name, int max_value) {
    const std::string & value = m.at(name);
    size_t used = 0;
    long n = std::stol(value, &used);
    if (used != value.size() || n <= 0 || n > max_value)
        throw std::runtime_error("invalid " + name);
    return static_cast<int>(n);
}

static std::vector<float> read_f32(const std::string & path, size_t count) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in || in.tellg() != static_cast<std::streamoff>(count * sizeof(float)))
        throw std::runtime_error("wrong or missing F32 file size: " + path);
    in.seekg(0);
    std::vector<float> values(count);
    if (!in.read(reinterpret_cast<char *>(values.data()), count * sizeof(float)))
        throw std::runtime_error("short read: " + path);
    return values;
}

static bool close_enough(float a, float b) {
    return std::isfinite(a) && std::isfinite(b) &&
           std::abs(a-b) <= 1e-4f * std::max(1.0f, std::abs(b));
}

int main(int argc, char ** argv) {
  try {
    if (argc != 2) throw std::runtime_error("usage: fiber_qkv_graph OUT_DIR");
    const std::string dir = argv[1];
    const auto m = read_manifest(dir + "/witness.txt");
    if (m.at("schema") != "fiber-qkv-v1" || m.at("dtype") != "f32-le" ||
        m.at("role_order") != "QKV" || m.at("source_axes") != "out,in")
        throw std::runtime_error("unsupported schema, dtype, source axes, or QKV role order");
    const uint16_t endian = 1;
    if (*reinterpret_cast<const uint8_t *>(&endian) != 1 || sizeof(float) != 4 ||
        !std::numeric_limits<float>::is_iec559)
        throw std::runtime_error("F32 little-endian IEEE host required");
    const int D = dimension(m, "width", 1024);
    const int H = dimension(m, "heads", 1024);
    const int T = dimension(m, "tokens", 256);
    if (D % H || dimension(m, "ggml_weight_ne0", 1024) != D ||
        dimension(m, "ggml_weight_ne1", 3072) != 3*D)
        throw std::runtime_error("incompatible head count or GGML weight metadata");
    const int d = D/H;
    const auto w = read_f32(dir + "/weight.f32", 3ull*D*D);
    const auto b = read_f32(dir + "/bias.f32", 3ull*D);
    const auto x = read_f32(dir + "/input.f32", 1ull*T*D);
    const auto ref = read_f32(dir + "/expected.f32", 3ull*T*D);

    // Materialize the user-owned NTTESHGNN tensor family's [role,in,out]
    // ordering and prove it equals Torch's source [out,in] scalar access.
    std::vector<float> role_first(3ull*D*D);
    if (nt_materialize_role_first_f32(w.data(), D, role_first.data(), role_first.size()))
        throw std::runtime_error("NTTESHGNN role-first materialization rejected");
    for (int o=0; o<D; ++o) for (int i=0; i<D; ++i) for (int r=0; r<3; ++r) {
        const size_t src = (1ull*r*D + o)*D + i;
        const size_t dst = r + 3ull*(i + 1ull*D*o);
        if (role_first[dst] != w[src])
            throw std::runtime_error("role-first NTT fiber disagrees with source");
    }

    // Graph allocation owns all leaf data; no cross-runtime pointer aliases.
    const size_t memory = (w.size()+b.size()+x.size()+ref.size()*4)*sizeof(float)
                        + 4u*1024u*1024u;
    ggml_init_params params = {memory, nullptr, false};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), &ggml_free);
    if (!ctx) throw std::runtime_error("ggml_init failed");
    ggml_tensor *gw = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, D, 3*D);
    ggml_tensor *gx = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, D, T);
    ggml_tensor *gb = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 3*D);
    if (!gw || !gx || !gb || gw->nb[0] != 4 || gw->nb[1] != 4ull*D ||
        gx->nb[0] != 4 || gx->nb[1] != 4ull*D)
        throw std::runtime_error("unexpected GGML leaf byte strides");
    std::memcpy(gw->data, w.data(), w.size()*sizeof(float));
    std::memcpy(gx->data, x.data(), x.size()*sizeof(float));
    std::memcpy(gb->data, b.data(), b.size()*sizeof(float));

    ggml_tensor *projected = ggml_mul_mat(ctx.get(), gw, gx);  // [3D,T]
    ggml_tensor *bias_2d = ggml_reshape_2d(ctx.get(), gb, 3*D, 1);
    ggml_tensor *repeated_bias = ggml_repeat(ctx.get(), bias_2d, projected);
    ggml_tensor *y = ggml_add(ctx.get(), projected, repeated_bias);
    ggml_tensor *packed = ggml_reshape_4d(ctx.get(), y, d, H, 3, T);
    // Pinned GGML uses axis_[source_axis] = output_axis; unlike the NTT API.
    ggml_tensor *permuted = ggml_permute(ctx.get(), packed, 2, 1, 0, 3);
    ggml_tensor *materialized = ggml_cont(ctx.get(), permuted); // [role,head,c,token]
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, materialized);
    if (ggml_graph_compute_with_ctx(ctx.get(), graph, 2) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("GGML CPU graph computation failed");
    if (y->ne[0] != 3*D || y->ne[1] != T ||
        materialized->ne[0] != 3 || materialized->ne[1] != H ||
        materialized->ne[2] != d || materialized->ne[3] != T ||
        !ggml_is_contiguous(materialized))
        throw std::runtime_error("computed graph shape/layout witness mismatch");
    float max_abs_error = 0;
    size_t compared = 0;
    for (int t=0; t<T; ++t) for (int r=0; r<3; ++r)
    for (int h=0; h<H; ++h) for (int c=0; c<d; ++c) {
        const size_t torch_idx = 1ull*t*(3*D) + r*D + h*d + c;
        const size_t ggml_idx = r + 3ull*(h + H*(c + d*t));
        const float expected = ref[torch_idx];
        const float projected_value = ((const float *)y->data)[torch_idx];
        const float fiber_value = ((const float *)materialized->data)[ggml_idx];
        if (!close_enough(projected_value, expected) ||
            !close_enough(fiber_value, expected))
            throw std::runtime_error("Torch7 reference / GGML projection / fiber parity mismatch");
        max_abs_error = std::max(max_abs_error, std::abs(fiber_value-expected));
        ++compared;
    }
    std::cout << "PASS: " << compared << " QKV scalars; source=" << m.at("origin")
              << "; ne(W)=[" << gw->ne[0] << "," << gw->ne[1] << "]"
              << "; ne(fiber)=[3," << H << "," << d << "," << T << "]"
              << "; max_abs_error=" << max_abs_error << '\n';
    return 0;
  } catch (const std::exception & e) {
    std::cerr << "fiber boundary rejected: " << e.what() << '\n';
    return 1;
  }
}
